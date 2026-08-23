// potluck-server admission, discovery, profiling, and route planning.

#include "internal.h"

#include <cmath>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static uint64_t saturating_add(uint64_t left, uint64_t right) {
    if (left > std::numeric_limits<uint64_t>::max() - right) {
        return std::numeric_limits<uint64_t>::max();
    }
    return left + right;
}

static uint64_t saturating_mul(uint64_t left, uint64_t right) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        return std::numeric_limits<uint64_t>::max();
    }
    return left * right;
}

uint64_t route_layer_cost(uint32_t n_layer, uint64_t model_bytes, uint32_t n_head_kv,
                          uint32_t head_dim, uint32_t n_ctx) {
    if (n_layer == 0) {
        return 0;
    }
    const uint64_t weights_per_layer = std::max<uint64_t>(1, model_bytes / n_layer);
    const uint64_t kv_per_layer = saturating_mul(
        saturating_mul(saturating_mul(n_head_kv, head_dim), n_ctx), 4);
    return std::max<uint64_t>(1, saturating_add(weights_per_layer, kv_per_layer));
}

uint64_t route_needed_bytes(uint32_t n_layer, uint64_t model_bytes, uint32_t n_head_kv,
                            uint32_t head_dim, uint32_t n_ctx) {
    return saturating_mul(static_cast<uint64_t>(n_layer),
                          route_layer_cost(n_layer, model_bytes, n_head_kv, head_dim, n_ctx));
}

static bool allocate_round_spans(uint32_t total_layers, const std::vector<size_t> & owners,
                          const std::vector<uint64_t> & usable,
                          const std::vector<uint32_t> & max_layers,
                          std::vector<uint32_t> & spans) {
    if (owners.empty() || total_layers < owners.size() ||
        usable.size() != max_layers.size()) {
        return false;
    }
    uint64_t capacity = 0;
    for (const size_t owner : owners) {
        if (owner >= usable.size() || max_layers[owner] == 0) {
            return false;
        }
        capacity = saturating_add(capacity, max_layers[owner]);
    }
    if (capacity < total_layers) {
        return false;
    }

    spans.assign(usable.size(), 0);
    for (const size_t owner : owners) {
        spans[owner] = 1;
    }
    uint64_t remaining = total_layers - owners.size();
    std::vector<long double> fractional(owners.size(), 0.0L);
    while (remaining != 0) {
        long double weight_sum = 0.0L;
        for (const size_t owner : owners) {
            if (spans[owner] < max_layers[owner]) {
                weight_sum += static_cast<long double>(usable[owner]);
            }
        }
        if (weight_sum <= 0.0L) {
            return false;
        }

        uint64_t assigned = 0;
        for (size_t index = 0; index < owners.size(); ++index) {
            const size_t owner = owners[index];
            if (spans[owner] >= max_layers[owner]) {
                fractional[index] = -1.0L;
                continue;
            }
            const long double ideal = static_cast<long double>(remaining) *
                                      static_cast<long double>(usable[owner]) / weight_sum;
            const uint64_t floor_extra = ideal >= static_cast<long double>(
                std::numeric_limits<uint64_t>::max())
                ? remaining : static_cast<uint64_t>(ideal);
            const uint64_t room = max_layers[owner] - spans[owner];
            const uint64_t extra = std::min({ floor_extra, room, remaining - assigned });
            spans[owner] += static_cast<uint32_t>(extra);
            assigned += extra;
            fractional[index] = std::max<long double>(
                0.0L, ideal - static_cast<long double>(floor_extra));
        }
        if (assigned != 0) {
            remaining -= assigned;
            continue;
        }

        size_t best = owners.size();
        for (size_t index = 0; index < owners.size(); ++index) {
            const size_t owner = owners[index];
            if (spans[owner] >= max_layers[owner]) {
                continue;
            }
            if (best == owners.size() || fractional[index] > fractional[best] ||
                (fractional[index] == fractional[best] && owner < owners[best])) {
                best = index;
            }
        }
        if (best == owners.size()) {
            return false;
        }
        ++spans[owners[best]];
        --remaining;
    }
    return true;
}
std::vector<device_probe> admit_devices(std::vector<device_probe> candidates,
                                         uint32_t n_layer, uint64_t model_bytes,
                                         uint32_t n_head_kv, uint32_t head_dim,
                                         uint32_t n_ctx, bool reserve_owner_slot) {
    const uint64_t layer_cost = route_layer_cost(n_layer, model_bytes, n_head_kv,
                                                 head_dim, n_ctx);
    const uint64_t needed = route_needed_bytes(n_layer, model_bytes, n_head_kv,
                                               head_dim, n_ctx);
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const device_probe & left, const device_probe & right) {
                         return left.usable_bytes() > right.usable_bytes();
                     });
    std::vector<device_probe> positive;
    positive.reserve(candidates.size());
    for (device_probe & candidate : candidates) {
        if (!candidate.ok) {
            continue;
        }
        if (candidate.usable_bytes() < layer_cost) {
            std::fprintf(stderr,
                         "potluck-server: excluding %s: insufficient memory for one layer\n",
                         candidate.host.c_str());
            continue;
        }
        positive.push_back(std::move(candidate));
    }
    if (positive.empty()) {
        throw std::runtime_error(
            "cluster has no device able to hold one active layer (need " +
            std::to_string(layer_cost / (1024ull * 1024ull)) + " MiB)");
    }

    // At least two rounds are required for every multi-layer model. Reserve
    // one owner position when the head will be appended after remote admission.
    const size_t max_owners = n_layer == 1
        ? 1 : std::max<size_t>(1, n_layer / 2);
    const size_t reserved_owners = reserve_owner_slot && max_owners > 1 ? 1 : 0;
    const size_t admitted_count =
        std::min(max_owners - reserved_owners, positive.size());
    for (size_t i = admitted_count; i < positive.size(); ++i) {
        std::fprintf(stderr, "potluck-server: excluding %s: no layer slot remains\n",
                     positive[i].host.c_str());
    }
    positive.resize(admitted_count);

    uint64_t capacity = 0;
    for (const device_probe & candidate : positive) {
        capacity = saturating_add(capacity, candidate.usable_bytes());
    }
    std::printf("potluck-server: admitted %zu devices (%llu MiB usable, need %llu MiB)\n",
                positive.size(), static_cast<unsigned long long>(capacity / (1024ull * 1024ull)),
                static_cast<unsigned long long>(needed / (1024ull * 1024ull)));
    return positive;
}

std::vector<potluck::ring_window> build_ring_route(const std::vector<device_probe> & devices,
                                                   uint32_t n_layer, uint64_t model_bytes,
                                                   uint32_t n_head_kv, uint32_t head_dim,
                                                   uint32_t n_ctx) {
    if (devices.empty()) {
        throw std::runtime_error("ring needs at least one admitted device");
    }
    if (n_layer == 0) {
        throw std::runtime_error("model reports zero layers");
    }
    const uint64_t layer_cost = route_layer_cost(n_layer, model_bytes, n_head_kv,
                                                 head_dim, n_ctx);
    std::vector<uint64_t> usable;
    std::vector<uint32_t> max_layers;
    usable.reserve(devices.size());
    max_layers.reserve(devices.size());
    for (const device_probe & device : devices) {
        const uint64_t bytes = device.usable_bytes();
        if (bytes < layer_cost) {
            throw std::runtime_error("ring device cannot hold one active layer");
        }
        usable.push_back(bytes);
        max_layers.push_back(static_cast<uint32_t>(std::min<uint64_t>(
            n_layer, bytes / layer_cost)));
    }
    if (n_layer > 1 && devices.size() > n_layer / 2) {
        throw std::runtime_error("ring has too many owners for repeated windows");
    }

    std::vector<size_t> owners;
    owners.reserve(devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        owners.push_back(i);
    }

    uint32_t selected_rounds = 0;
    uint32_t larger_rounds = 0;
    std::vector<uint32_t> smaller_spans;
    std::vector<uint32_t> larger_spans;
    for (uint32_t rounds = n_layer == 1 ? 1 : 2; rounds <= n_layer; ++rounds) {
        const uint32_t smaller_layers = n_layer / rounds;
        const uint32_t remainder = n_layer % rounds;
        if (smaller_layers < owners.size()) {
            continue;
        }
        std::vector<uint32_t> candidate_smaller;
        if (!allocate_round_spans(smaller_layers, owners, usable, max_layers,
                                  candidate_smaller)) {
            continue;
        }
        std::vector<uint32_t> candidate_larger = candidate_smaller;
        if (remainder != 0 &&
            !allocate_round_spans(smaller_layers + 1, owners, usable, max_layers,
                                  candidate_larger)) {
            continue;
        }
        selected_rounds = rounds;
        larger_rounds = remainder;
        smaller_spans = std::move(candidate_smaller);
        larger_spans = std::move(candidate_larger);
        break;
    }
    if (selected_rounds == 0) {
        throw std::runtime_error("no repeated ring route fits admitted device memory");
    }

    std::vector<potluck::ring_window> windows;
    windows.reserve(static_cast<size_t>(selected_rounds) * devices.size());
    uint32_t next_layer = 0;
    for (uint32_t round = 0; round < selected_rounds; ++round) {
        const std::vector<uint32_t> & spans =
            round < larger_rounds ? larger_spans : smaller_spans;
        for (size_t i = 0; i < owners.size(); ++i) {
            const uint32_t span = spans[owners[i]];
            if (span == 0) {
                throw std::runtime_error("ring route contains a zero-width owner");
            }
            windows.push_back({ static_cast<uint32_t>(owners[i]), next_layer,
                                next_layer + span, 0 });
            next_layer += span;
        }
    }
    if (next_layer != n_layer) {
        throw std::runtime_error("capability route does not cover the model");
    }
    return windows;
}

static std::string basename_of(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}
std::string shell_quote(const std::string & value) {
    std::string quoted = "'";
    for (const char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += '\'';
    return quoted;
}
static bool valid_ssh_component(const std::string & value) {
    if (value.empty() || value.front() == '-') {
        return false;
    }
    for (const unsigned char character : value) {
        if (character <= 0x20 || character == 0x7f) {
            return false;
        }
    }
    return true;
}

void validate_ssh_target(const std::string & target) {
    if (!valid_ssh_component(target)) {
        throw std::runtime_error("invalid SSH target");
    }
    const size_t at = target.rfind('@');
    const std::string user = at == std::string::npos
        ? std::string() : target.substr(0, at);
    const std::string host = at == std::string::npos
        ? target : target.substr(at + 1);
    if (at != std::string::npos &&
        (at == 0 || target.find('@') != at || !valid_ssh_component(user))) {
        throw std::runtime_error("invalid SSH user in target");
    }
    if (!valid_ssh_component(host)) {
        throw std::runtime_error("invalid SSH host in target");
    }
}

static std::string discovery_known_hosts_file() {
    const char * xdg = std::getenv("XDG_CONFIG_HOME");
    const char * home = std::getenv("HOME");
    std::filesystem::path directory;
    if (xdg != nullptr && xdg[0] != '\0') {
        directory = std::filesystem::path(xdg) / "potluck";
    } else if (home != nullptr && home[0] != '\0') {
        directory = std::filesystem::path(home) / ".config" / "potluck";
    } else {
        throw std::runtime_error("automatic SSH trust needs XDG_CONFIG_HOME or HOME");
    }
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        throw std::runtime_error("cannot create Potluck config directory: " + error.message());
    }
    return (directory / "known_hosts").string();
}

const char * accel_kind_name(potluck::accel_kind kind) {
    switch (kind) {
        case potluck::accel_kind::metal:
            return "metal";
        case potluck::accel_kind::cuda:
            return "cuda";
        case potluck::accel_kind::other:
            return "accelerator";
        default:
            return "none";
    }
}
std::string ssh_options(const bootstrap_node & bootstrap) {
    validate_ssh_target(bootstrap.ssh_target);
    const std::string trust = bootstrap.known_hosts_file.empty()
        ? std::string()
        : " -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=" +
          shell_quote(bootstrap.known_hosts_file);
    const std::string port = bootstrap.ssh_port == 22
        ? std::string() : " -p " + std::to_string(bootstrap.ssh_port);
    return "ssh -o BatchMode=yes -o ConnectTimeout=5"
           " -o ServerAliveInterval=2 -o ServerAliveCountMax=2" + trust + port;
}
std::vector<bootstrap_node> discover_bootstrap_nodes() {
    std::string discovery_error;
    const std::vector<potluck::discovered_node> discovered =
        potluck::discover_nodes(3000, discovery_error);
    if (!discovery_error.empty()) {
        throw std::runtime_error("mDNS discovery failed: " + discovery_error);
    }
    const std::string known_hosts_file = discovery_known_hosts_file();
    std::vector<bootstrap_node> result;
    result.reserve(discovered.size());
    for (const potluck::discovered_node & node : discovered) {
        const std::string label = !node.id.empty()
            ? node.id : (node.instance.empty() ? std::string("<unnamed>") : node.instance);
        const auto skip = [&](const char * reason) {
            std::fprintf(stderr, "potluck-server: skipping incompatible discovered node '%s': %s\n",
                         label.c_str(), reason);
        };
        if (node.protocol_version != potluck::discovery_protocol_version) {
            skip("unsupported protocol version");
            continue;
        }
        if (!node.available) {
            skip("node is not available");
            continue;
        }
        if (node.id.empty()) {
            skip("missing id");
            continue;
        }
        if (!valid_ssh_component(node.host)) {
            skip("invalid SSH host");
            continue;
        }
        if (!valid_ssh_component(node.ssh_user) ||
            node.ssh_user.find('@') != std::string::npos) {
            skip("invalid SSH user");
            continue;
        }
        if (node.ssh_port == 0 || node.ring_port == 0) {
            skip("missing SSH or ring port");
            continue;
        }
        bootstrap_node bootstrap;
        bootstrap.ssh_target = node.ssh_user + "@" + node.host;
        bootstrap.ssh_port = node.ssh_port;
        bootstrap.ring_host = node.host;
        bootstrap.ring_port = node.ring_port;
        bootstrap.known_hosts_file = known_hosts_file;
        try {
            validate_ssh_target(bootstrap.ssh_target);
        } catch (const std::exception &) {
            skip("invalid SSH target");
            continue;
        }
        result.push_back(std::move(bootstrap));
    }
    if (result.empty()) {
        throw std::runtime_error("mDNS discovery returned no eligible nodes");
    }
    return result;
}

static bool parse_probe_line(const std::string & line, potluck::accel_profile & profile,
                      std::string & error) {
    char kind[32] = {};
    unsigned long long accel_free = 0;
    unsigned long long accel_total = 0;
    unsigned long long host_free = 0;
    unsigned long long host_total = 0;
    if (std::sscanf(line.c_str(),
                    "potluck-probe kind=%31s accel_free=%llu accel_total=%llu host_free=%llu host_total=%llu",
                    kind, &accel_free, &accel_total, &host_free, &host_total) != 5) {
        error = "invalid probe output";
        return false;
    }
    potluck::accel_kind accel_kind = potluck::accel_kind::none;
    if (std::strcmp(kind, "metal") == 0) {
        accel_kind = potluck::accel_kind::metal;
    } else if (std::strcmp(kind, "cuda") == 0) {
        accel_kind = potluck::accel_kind::cuda;
    } else if (std::strcmp(kind, "other") == 0) {
        accel_kind = potluck::accel_kind::other;
    } else if (std::strcmp(kind, "none") != 0) {
        error = "unknown probe accelerator kind";
        return false;
    }
    profile.kind = accel_kind;
    profile.free_bytes = accel_free;
    profile.total_bytes = accel_total;
    profile.host_free_bytes = host_free;
    profile.host_total_bytes = host_total;
    return true;
}

static device_probe run_probe_command(const std::string & command, const std::string & host) {
    device_probe result;
    result.host = host;
    int output_pipe[2] = {};
    if (pipe(output_pipe) != 0) {
        result.error = "cannot start probe pipe";
        return result;
    }
    const pid_t pid = fork();
    if (pid < 0) {
        close(output_pipe[0]);
        close(output_pipe[1]);
        result.error = "cannot fork probe";
        return result;
    }
    if (pid == 0) {
        close(output_pipe[0]);
        (void) setpgid(0, 0);
        if (dup2(output_pipe[1], STDOUT_FILENO) < 0) {
            std::_Exit(127);
        }
        close(output_pipe[1]);
        execl("/bin/sh", "sh", "-c", command.c_str(), static_cast<char *>(nullptr));
        std::_Exit(127);
    }
    close(output_pipe[1]);
    const int flags = fcntl(output_pipe[0], F_GETFL, 0);
    if (flags >= 0) {
        (void) fcntl(output_pipe[0], F_SETFL, flags | O_NONBLOCK);
    }
    std::string output;
    int status = 0;
    bool timed_out = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        char buffer[512] = {};
        for (;;) {
            const ssize_t count = read(output_pipe[0], buffer, sizeof(buffer));
            if (count > 0) {
                if (output.size() < 4096) {
                    output.append(buffer, static_cast<size_t>(
                        std::min<ssize_t>(count, 4096 - output.size())));
                }
                continue;
            }
            if (count == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
                break;
            }
            break;
        }
        const pid_t waited = waitpid(pid, &status, WNOHANG);
        if (waited == pid) {
            for (;;) {
                const ssize_t count = read(output_pipe[0], buffer, sizeof(buffer));
                if (count > 0) {
                    if (output.size() < 4096) {
                        output.append(buffer, static_cast<size_t>(
                            std::min<ssize_t>(count, 4096 - output.size())));
                    }
                    continue;
                }
                break;
            }
            break;
        }
        if (waited < 0) {
            result.error = "cannot wait for probe";
            break;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            (void) kill(-pid, SIGKILL);
            (void) kill(pid, SIGKILL);
            (void) waitpid(pid, &status, 0);
            timed_out = true;
            break;
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - now);
        pollfd descriptor = { output_pipe[0], POLLIN, 0 };
        (void) poll(&descriptor, 1, static_cast<int>(
            std::min<int64_t>(100, std::max<int64_t>(1, remaining.count()))));
    }
    close(output_pipe[0]);
    if (timed_out) {
        result.error = "probe timed out after 30 seconds";
        return result;
    }
    if (!result.error.empty() || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        if (result.error.empty()) {
            result.error = "probe command failed";
        }
        return result;
    }
    const size_t newline = output.find('\n');
    const std::string line = output.substr(0, newline);
    result.profile.rank = 0;
    result.ok = parse_probe_line(line, result.profile, result.error);
    return result;
}

device_probe probe_local_worker(const std::string & worker_path) {
    return run_probe_command(shell_quote(worker_path) + " --probe", "127.0.0.1");
}

device_probe probe_remote_worker(const bootstrap_node & bootstrap) {
    const std::string command = ssh_options(bootstrap) + " " + shell_quote(bootstrap.ssh_target) +
        " " + shell_quote("cd ~/potluck || exit 1; LD_LIBRARY_PATH=.${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} ./potluck-worker --probe");
    return run_probe_command(command, bootstrap.ring_host);
}

std::vector<device_probe> probe_remote_candidates(
    const std::vector<bootstrap_node> & candidates) {
    struct probe_task {
        std::future<device_probe> future;
        std::thread thread;
        std::chrono::steady_clock::time_point started;
    };
    std::vector<probe_task> tasks;
    tasks.reserve(candidates.size());
    for (const bootstrap_node & bootstrap : candidates) {
        const auto started = std::chrono::steady_clock::now();
        std::promise<device_probe> promise;
        std::future<device_probe> future = promise.get_future();
        std::thread thread([bootstrap, promise = std::move(promise)]() mutable {
            try {
                promise.set_value(probe_remote_worker(bootstrap));
            } catch (const std::exception & exception) {
                device_probe failure;
                failure.host = bootstrap.ring_host;
                failure.bootstrap = bootstrap;
                failure.error = exception.what();
                promise.set_value(std::move(failure));
            } catch (...) {
                device_probe failure;
                failure.host = bootstrap.ring_host;
                failure.bootstrap = bootstrap;
                failure.error = "probe failed with an unknown error";
                promise.set_value(std::move(failure));
            }
        });
        tasks.push_back({ std::move(future), std::move(thread), started });
    }
    std::vector<device_probe> results;
    results.reserve(candidates.size());
    for (size_t index = 0; index < tasks.size(); ++index) {
        device_probe result;
        const auto now = std::chrono::steady_clock::now();
        const auto task_deadline = tasks[index].started + std::chrono::seconds(30);
        const auto timeout = now < task_deadline
            ? task_deadline - now : std::chrono::steady_clock::duration::zero();
        if (tasks[index].future.wait_for(timeout) == std::future_status::ready) {
            result = tasks[index].future.get();
            tasks[index].thread.join();
        } else {
            result.host = candidates[index].ring_host;
            result.bootstrap = candidates[index];
            result.error = "probe timed out after 30 seconds";
            tasks[index].thread.detach();
        }
        result.bootstrap = candidates[index];
        results.push_back(std::move(result));
    }
    return results;
}

static std::string local_sha256(const std::string & path) {
    const std::string command = "(command -v sha256sum >/dev/null && sha256sum " +
        shell_quote(path) + " || shasum -a 256 " + shell_quote(path) + ") 2>/dev/null";
    FILE * pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("cannot compute model checksum");
    }
    char digest[129] = {};
    const bool read = std::fscanf(pipe, "%128s", digest) == 1;
    const int status = pclose(pipe);
    if (!read || status != 0) {
        throw std::runtime_error("cannot compute model checksum");
    }
    return digest;
}
std::filesystem::path canonical_model_path(const std::filesystem::path & model_path) {
    std::error_code error;
    const std::filesystem::path absolute = std::filesystem::absolute(model_path, error);
    if (error) {
        throw std::runtime_error("cannot canonicalize model path: " + error.message());
    }
    error.clear();
    const std::filesystem::path canonical =
        std::filesystem::weakly_canonical(absolute, error);
    if (error) {
        throw std::runtime_error("cannot canonicalize model path: " + error.message());
    }
    return canonical;
}

static std::string model_path_key(const std::filesystem::path & canonical_path) {
    static constexpr char hex[] = "0123456789abcdef";
    const std::string text = canonical_path.string();
    std::string key;
    key.reserve(text.size() * 2);
    for (const unsigned char character : text) {
        key += hex[character >> 4];
        key += hex[character & 0x0f];
    }
    return key;
}

static bool valid_sha256_digest(const std::string & digest) {
    if (digest.size() != 64) {
        return false;
    }
    for (const unsigned char character : digest) {
        if (!std::isxdigit(character)) {
            return false;
        }
    }
    return true;
}


std::filesystem::path model_digest_cache::persistent_path(
    const std::filesystem::path & canonical_path) {
    const char * home = std::getenv("HOME");
    if (home == nullptr || home[0] == '\0') {
        return {};
    }
    return std::filesystem::path(home) / ".cache" / "potluck" /
           (model_path_key(canonical_path) + ".sha256");
}

std::string model_digest_cache::get(const std::filesystem::path & model_path) {
    const std::filesystem::path canonical = canonical_model_path(model_path);
    const uintmax_t current_size = std::filesystem::file_size(canonical);
    const auto current_mtime = std::filesystem::last_write_time(canonical);
    const long long current_mtime_ticks =
        static_cast<long long>(current_mtime.time_since_epoch().count());
    const std::string key = model_path_key(canonical);
    std::lock_guard<std::mutex> lock(mutex);
    if (valid && path == canonical && size == current_size && mtime == current_mtime) {
        return digest;
    }

    const std::filesystem::path cache_path = persistent_path(canonical);
    if (!cache_path.empty()) {
        std::ifstream input(cache_path);
        std::string cached_key;
        uintmax_t cached_size = 0;
        long long cached_mtime_ticks = 0;
        std::string cached_digest;
        if (input >> cached_key >> cached_size >> cached_mtime_ticks >> cached_digest &&
            cached_key == key && cached_size == current_size &&
            cached_mtime_ticks == current_mtime_ticks &&
            valid_sha256_digest(cached_digest)) {
            digest = std::move(cached_digest);
            path = canonical;
            size = current_size;
            mtime = current_mtime;
            valid = true;
            return digest;
        }
    }

    digest = local_sha256(canonical.string());
    path = canonical;
    size = current_size;
    mtime = current_mtime;
    valid = true;
    if (!cache_path.empty()) {
        std::error_code error;
        std::filesystem::create_directories(cache_path.parent_path(), error);
        if (!error) {
            std::ofstream output(cache_path, std::ios::trunc);
            if (output) {
                output << key << ' ' << current_size << ' '
                       << current_mtime_ticks << ' ' << digest << '\n';
            }
        }
    }
    return digest;
}

static std::string command_output(const std::string & command) {
    FILE * pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return {};
    }
    std::string output;
    char buffer[512] = {};
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    const int status = pclose(pipe);
    return status == 0 ? output : std::string();
}

static std::string file_contents(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        return {};
    }
    std::string output;
    char buffer[512] = {};
    while (input.read(buffer, sizeof(buffer)) || input.gcount() != 0) {
        output.append(buffer, static_cast<size_t>(input.gcount()));
    }
    return output;
}

std::string first_command_line(const std::string & command) {
    FILE * pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return {};
    }
    char buffer[512] = {};
    std::string line;
    if (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        line = buffer;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
    }
    const int status = pclose(pipe);
    return status == 0 ? line : std::string();
}

static std::string first_file_line(const std::filesystem::path & path) {
    std::ifstream input(path);
    std::string line;
    std::getline(input, line);
    return line;
}

bool refresh_remote_binaries(const bootstrap_node & bootstrap,
                             const std::filesystem::path & stage_dir,
                             const std::string & local_platform) {
    const std::filesystem::path build_id_path = stage_dir / "potluck-build-id";
    if (!std::filesystem::is_regular_file(build_id_path)) {
        return true;
    }
    const std::string ssh = ssh_options(bootstrap);
    const std::string target = shell_quote(bootstrap.ssh_target);
    const std::string remote_platform = first_command_line(
        ssh + " " + target + " " + shell_quote("uname -sm 2>/dev/null"));
    if (remote_platform.empty()) {
        std::fprintf(stderr, "potluck-server: cannot inspect platform on %s\n",
                     bootstrap.ssh_target.c_str());
        return false;
    }
    if (remote_platform != local_platform) {
        std::printf("potluck-server: %s manages its own build\n",
                    bootstrap.ring_host.c_str());
        return true;
    }
    const std::string local_build_id = file_contents(build_id_path);
    const std::string remote_build_id = command_output(
        ssh + " " + target + " " +
        shell_quote("cat ~/potluck/potluck-build-id 2>/dev/null"));
    if (!local_build_id.empty() && local_build_id == remote_build_id) {
        return true;
    }
    const std::string mkdir = ssh + " " + target + " " +
        shell_quote("mkdir -p ~/potluck");
    if (std::system(mkdir.c_str()) != 0) {
        std::fprintf(stderr, "potluck-server: cannot prepare payload directory on %s\n",
                     bootstrap.ssh_target.c_str());
        return false;
    }
    const std::string rsync = "rsync -a --whole-file -e " + shell_quote(ssh) + " " +
        shell_quote((stage_dir.string() + "/")) + " " +
        shell_quote(bootstrap.ssh_target + ":potluck/");
    if (std::system(rsync.c_str()) != 0) {
        std::fprintf(stderr, "potluck-server: cannot refresh worker binaries on %s\n",
                     bootstrap.ssh_target.c_str());
        return false;
    }
    std::printf("potluck-server: refreshed worker binaries on %s\n",
                bootstrap.ring_host.c_str());
    std::fflush(stdout);
    return true;
}


bool ensure_remote_artifact(const bootstrap_node & bootstrap,
                            const std::filesystem::path & local_path,
                            const std::filesystem::path & remote_path,
                            const std::string & digest, std::string & error) {
    error.clear();
    if (remote_path.empty() || remote_path.is_absolute() || remote_path.filename().empty()) {
        error = "remote artifact path must be a relative file";
        return false;
    }
    for (const std::filesystem::path & part : remote_path) {
        if (part == "..") {
            error = "remote artifact path cannot contain '..'";
            return false;
        }
    }
    const std::string source = canonical_model_path(local_path).string();
    const std::string remote = remote_path.generic_string();
    const std::string ssh = ssh_options(bootstrap);
    const std::string remote_check =
        "(cd ~/potluck && (sha256sum " + shell_quote(remote) +
        " 2>/dev/null || shasum -a 256 " + shell_quote(remote) +
        " 2>/dev/null) | cut -d' ' -f1)";
    const std::string check = ssh + " " + shell_quote(bootstrap.ssh_target) + " " +
        shell_quote(remote_check);
    const auto remote_digest = [&]() {
        FILE * pipe = popen(check.c_str(), "r");
        if (pipe == nullptr) {
            return std::string();
        }
        char buffer[256] = {};
        std::string actual;
        if (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            actual = buffer;
            while (!actual.empty() &&
                   (actual.back() == '\n' || actual.back() == '\r')) {
                actual.pop_back();
            }
        }
        const int status = pclose(pipe);
        return status == 0 ? actual : std::string();
    };

    const std::string actual = remote_digest();
    if (actual == digest) {
        std::printf("potluck-server: %s already has %s\n",
                    bootstrap.ring_host.c_str(), remote.c_str());
        return true;
    }
    std::printf("potluck-server: sending %s to %s (%llu MiB)\n", remote.c_str(),
                bootstrap.ring_host.c_str(),
                static_cast<unsigned long long>(
                    std::filesystem::file_size(source) / (1024ull * 1024ull)));
    std::fflush(stdout);
    const std::filesystem::path parent = remote_path.parent_path();
    const std::string directory = "mkdir -p ~/potluck && cd ~/potluck && mkdir -p " +
        shell_quote(parent.empty() ? "." : parent.generic_string());
    const std::string mkdir = ssh + " " + shell_quote(bootstrap.ssh_target) +
        " " + shell_quote(directory);
    if (std::system(mkdir.c_str()) != 0) {
        error = "cannot prepare remote artifact directory";
        return false;
    }
    const std::string rsync = "rsync -a --whole-file --partial --inplace -e " +
        shell_quote(ssh) + " " + shell_quote(source) + " " +
        shell_quote(bootstrap.ssh_target + ":potluck/" + remote);
    if (std::system(rsync.c_str()) != 0) {
        error = "artifact transfer failed";
        return false;
    }
    const std::string transferred = remote_digest();
    if (transferred != digest) {
        error = "remote checksum mismatch (expected " + digest + ", got " +
            (transferred.empty() ? std::string("<unavailable>") : transferred) + ")";
        return false;
    }
    return true;
}

std::string pinned_model_digest(const std::filesystem::path & model_path,
                                const std::filesystem::path & repo_root) {
    const std::filesystem::path script = repo_root / "scripts" / "potluck-model.sh";
    if (!std::filesystem::is_regular_file(script) || model_path.filename().empty()) {
        return {};
    }
    const std::string script_command =
        "source " + shell_quote(script.string()) +
        " && if [[ " + shell_quote(model_path.filename().string()) +
        " == \"$POTLUCK_MODEL_FILE\" ]]; then "
        "printf '%s' \"$POTLUCK_MODEL_SHA256\"; fi";
    const std::string digest = command_output("bash -c " + shell_quote(script_command));
    if (!valid_sha256_digest(digest)) {
        return {};
    }
    const std::string actual = local_sha256(canonical_model_path(model_path).string());
    return actual == digest ? digest : std::string();
}


// Workers report their accelerator before asking for a schedule. Missing or
// timed-out profiles mean CPU-only execution for that worker.
std::vector<potluck::accel_profile> collect_accel_profiles(ServerRing & ring, uint32_t n_workers) {
    constexpr int profile_timeout_ms = 120000;
    if (!ring.result.set_receive_timeout(profile_timeout_ms, ring.error)) {
        throw std::runtime_error("cannot set ring profile timeout: " + ring.error);
    }
    std::vector<potluck::accel_profile> profiles(n_workers);
    std::vector<bool> seen(n_workers, false);
    for (uint32_t received = 0; received < n_workers; ++received) {
        potluck::message message;
        if (!ring.result.receive(message, ring.error)) {
            throw std::runtime_error("accelerator profile collection stopped at " +
                                     std::to_string(received) + " of " +
                                     std::to_string(n_workers) + " workers: " + ring.error);
        }
        if (message.type != potluck::message_type::profile_result ||
            message.rank >= n_workers || seen[message.rank]) {
            throw std::runtime_error("ring worker sent an unexpected accelerator profile");
        }
        if (!potluck::decode_accel_profile(message.payload.data(), message.payload.size(),
                                           profiles[message.rank], ring.error)) {
            throw std::runtime_error("cannot decode accelerator profile: " + ring.error);
        }
        if (profiles[message.rank].rank != message.rank) {
            throw std::runtime_error("accelerator profile rank mismatch");
        }
        seen[message.rank] = true;
        const potluck::accel_profile & profile = profiles[message.rank];
        std::printf("potluck-server: worker %u accelerator %s free %llu MiB total %llu MiB\n",
                    message.rank, accel_kind_name(profile.kind),
                    static_cast<unsigned long long>(profile.free_bytes / (1024ull * 1024ull)),
                    static_cast<unsigned long long>(profile.total_bytes / (1024ull * 1024ull)));
    }
    std::fflush(stdout);
    return profiles;
}

// Spend each worker's usable accelerator memory across its own windows in ring
// order. Layers stay on CPU when the device is absent, small, or busy.
void assign_gpu_layers(std::vector<potluck::ring_window> & windows,
                       const std::vector<potluck::accel_profile> & profiles,
                       uint32_t n_workers, uint64_t model_bytes, uint32_t n_layer,
                       uint32_t n_head_kv, uint32_t head_dim, uint32_t n_ctx) {
    constexpr uint64_t mib = 1024ull * 1024ull;
    constexpr uint64_t window_slack_bytes = 64 * mib;
    constexpr uint64_t min_device_reserve_bytes = 512 * mib;
    if (windows.empty() || profiles.size() != n_workers || n_layer == 0) {
        throw std::runtime_error("cannot plan placement without a route and profiles");
    }
    std::vector<int64_t> budget(n_workers, 0);
    for (uint32_t rank = 0; rank < n_workers; ++rank) {
        const potluck::accel_profile & profile = profiles[rank];
        if (profile.kind == potluck::accel_kind::none || profile.total_bytes == 0) {
            continue;
        }
        // Keep desktop memory responsive on unified-memory hosts.
        const uint64_t reserve = std::max(min_device_reserve_bytes, profile.total_bytes / 8);
        budget[rank] = profile.free_bytes > reserve && profile.free_bytes - reserve > 0
            ? static_cast<int64_t>(profile.free_bytes - reserve) : 0;
    }
    const uint64_t weights_per_layer = model_bytes / n_layer;
    const uint64_t kv_per_layer = 2ull * n_head_kv * head_dim * 2ull * n_ctx; // K and V, f16
    const uint64_t layer_cost = weights_per_layer + kv_per_layer;
    for (potluck::ring_window & window : windows) {
        int64_t affordable = static_cast<int64_t>(layer_cost) == 0
            ? 0 : (budget[window.owner] - static_cast<int64_t>(window_slack_bytes)) /
                      static_cast<int64_t>(layer_cost);
        affordable = std::max<int64_t>(affordable, 0);
        const int64_t span = window.end - window.start;
        const int32_t n_gpu_layers = static_cast<int32_t>(std::min(affordable, span));
        window.n_gpu_layers = n_gpu_layers;
        budget[window.owner] -= static_cast<int64_t>(n_gpu_layers) * static_cast<int64_t>(layer_cost);
    }
}
