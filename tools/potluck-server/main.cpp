// potluck-server: an OpenAI-compatible HTTP server around a direct
// ZeroMQ piped ring. Requests traverse the configured ring route.

#include "llama.h"
#include "common.h"
#include "chat.h"
#include "nlohmann/json.hpp"
#include "potluck-discovery.h"
#include "potluck-transport.h"
#include "potluck_runtime.h"
#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <sys/resource.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <limits.h>
#include <sys/types.h>
#endif

using json = nlohmann::ordered_json;

namespace {

[[noreturn]] void fail(const std::string & message) {
    std::fprintf(stderr, "potluck-server: %s\n", message.c_str());
    std::fflush(nullptr);
    std::_Exit(1); // startup failures skip the known ggml-metal teardown assert
}

std::string exe_dir(const char * argv0) {
    std::string path;
#if defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size + 1, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) == 0) {
        path = buffer.data();
    }
#elif defined(__linux__)
    char buffer[PATH_MAX + 1] = {};
    const ssize_t n = readlink("/proc/self/exe", buffer, PATH_MAX);
    if (n > 0) {
        buffer[n] = '\0';
        path = buffer;
    }
#endif
    if (path.empty()) {
        path = argv0 == nullptr ? std::string() : std::string(argv0);
    }
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : path.substr(0, slash);
}

std::string basename_of(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string model_stem(const std::string & path) {
    const std::string base = basename_of(path);
    const size_t dot = base.rfind('.');
    return dot == std::string::npos ? base : base.substr(0, dot);
}

std::vector<std::string> split_csv(const std::string & text) {
    std::vector<std::string> values;
    size_t at = 0;
    for (;;) {
        const size_t comma = text.find(',', at);
        const std::string value = text.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        if (value.empty()) {
            throw std::runtime_error("comma-separated values cannot be empty");
        }
        values.push_back(value);
        if (comma == std::string::npos) {
            return values;
        }
        at = comma + 1;
    }
}

std::vector<std::string> parse_hosts(const std::string & text) {
    return split_csv(text);
}

std::string ring_host(const std::string & bootstrap_host) {
    const size_t user = bootstrap_host.rfind('@');
    const std::string host = user == std::string::npos ? bootstrap_host : bootstrap_host.substr(user + 1);
    if (host.empty()) {
        throw std::runtime_error("bootstrap host has no ring address");
    }
    return host;
}

uint32_t json_u32(const json & object, const char * key, uint32_t fallback) {
    if (!object.contains(key)) {
        return fallback;
    }
    const json & value = object.at(key);
    uint64_t number = 0;
    if (value.is_number_unsigned()) {
        number = value.get<uint64_t>();
    } else if (value.is_number_integer()) {
        const int64_t signed_number = value.get<int64_t>();
        if (signed_number < 0) {
            throw std::runtime_error(std::string("invalid ") + key + ": expected a non-negative integer");
        }
        number = static_cast<uint64_t>(signed_number);
    } else {
        throw std::runtime_error(std::string("invalid ") + key + ": expected a non-negative integer");
    }
    if (number > UINT32_MAX) {
        throw std::runtime_error(std::string("invalid ") + key + ": value is too large");
    }
    return static_cast<uint32_t>(number);
}

bool json_bool(const json & object, const char * key, bool fallback) {
    if (!object.contains(key)) {
        return fallback;
    }
    if (!object.at(key).is_boolean()) {
        throw std::runtime_error(std::string("invalid ") + key + ": expected a boolean");
    }
    return object.at(key).get<bool>();
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
std::string discovery_known_hosts_file() {
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


uint16_t free_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("cannot create worker port probe");
    }
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(fd, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) < 0) {
        close(fd);
        throw std::runtime_error("cannot bind worker port probe");
    }
    socklen_t length = sizeof(address);
    if (getsockname(fd, reinterpret_cast<sockaddr *>(&address), &length) < 0) {
        close(fd);
        throw std::runtime_error("cannot inspect worker port probe");
    }
    const uint16_t port = ntohs(address.sin_port);
    close(fd);
    return port;
}
std::string local_address_for_peer(const std::string & host, uint16_t port) {
    addrinfo hints = {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    const std::string service = std::to_string(port);
    addrinfo * addresses = nullptr;
    const int resolve = getaddrinfo(host.c_str(), service.c_str(), &hints, &addresses);
    if (resolve != 0) {
        throw std::runtime_error("cannot resolve discovered node " + host + ": " + gai_strerror(resolve));
    }
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> guard(addresses, freeaddrinfo);
    for (const addrinfo * address = addresses; address != nullptr; address = address->ai_next) {
        const int fd = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (fd < 0) {
            continue;
        }
        const int connected = connect(fd, address->ai_addr, address->ai_addrlen);
        sockaddr_in local = {};
        socklen_t local_length = sizeof(local);
        const bool found = connected == 0 &&
                           getsockname(fd, reinterpret_cast<sockaddr *>(&local), &local_length) == 0;
        close(fd);
        if (found) {
            char text[INET_ADDRSTRLEN] = {};
            if (inet_ntop(AF_INET, &local.sin_addr, text, sizeof(text)) != nullptr) {
                return text;
            }
        }
    }
    throw std::runtime_error("cannot select a local address for discovered node " + host);
}


std::string tcp_endpoint(const std::string & host, uint16_t port) {
    return "tcp://" + host + ":" + std::to_string(port);
}

std::string endpoint_host(const std::string & endpoint, const std::string & host) {
    const size_t scheme = endpoint.find("://");
    const size_t port = endpoint.rfind(':');
    if (scheme == std::string::npos || port == std::string::npos || port <= scheme + 3) {
        throw std::runtime_error("invalid ZeroMQ endpoint: " + endpoint);
    }
    return endpoint.substr(0, scheme + 3) + host + endpoint.substr(port);
}

std::string local_shard_path(const std::string & directory, const std::string & source,
                             uint32_t index, uint32_t count) {
    return directory + "/" + model_stem(source) + ".potluck-" + std::to_string(index) +
           "of" + std::to_string(count) + ".gguf";
}

struct ring_worker {
    potluck::node_addr address;
    std::string bind_endpoint;
    std::string connect_endpoint;
    std::string next_endpoint;
};

struct bootstrap_node {
    std::string ssh_target;
    uint16_t ssh_port = 22;
    std::string ring_host;
    uint16_t ring_port = 40001;
    std::string known_hosts_file;
};

struct ServerRing {
    std::vector<ring_worker> workers;
    std::vector<potluck::ring_window> windows;
    potluck::ring_receiver result;
    std::string result_endpoint;
    std::vector<potluck::ring_sender> controls;
    potluck::ring_sender ingress;
    std::string error;
};

struct serve_stats {
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    uint64_t head_payload_bytes = 0;
};

std::vector<potluck::ring_window> build_ring_route(uint32_t n_workers, uint32_t n_layer) {
    if (n_workers == 0) {
        throw std::runtime_error("ring needs at least one worker");
    }
    if (n_layer < n_workers) {
        throw std::runtime_error("ring needs at least one layer per worker");
    }
    constexpr uint32_t cycles = 2;
    const uint32_t n_windows = std::min(n_layer, n_workers * cycles);
    std::vector<potluck::ring_window> windows;
    windows.reserve(n_windows);
    for (uint32_t index = 0; index < n_windows; ++index) {
        const uint32_t start = static_cast<uint32_t>(
            (static_cast<uint64_t>(n_layer) * index) / n_windows);
        const uint32_t end = static_cast<uint32_t>(
            (static_cast<uint64_t>(n_layer) * (index + 1)) / n_windows);
        windows.push_back({ index % n_workers, start, end, 0 });
    }
    return windows;
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

// Workers report their accelerator before asking for a schedule. Missing or
// timed-out profiles mean CPU-only execution for that worker.
std::vector<potluck::accel_profile> collect_accel_profiles(ServerRing & ring, uint32_t n_workers) {
    constexpr int profile_timeout_ms = 60000;
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

bool launch_remote_worker(const bootstrap_node & bootstrap, const std::string & model,
                          const ring_worker & worker, const std::string & result_endpoint,
                          uint32_t index) {
    const std::string log = "worker-" + std::to_string(index) + ".log";
    const std::string remote = "cd ~/potluck || exit 1; nohup ./potluck-worker " + shell_quote(model) +
                               " --bind " + shell_quote(worker.bind_endpoint) +
                               " --next " + shell_quote(worker.next_endpoint) +
                               " --result " + shell_quote(result_endpoint) +
                               " --rank " + std::to_string(index) +
                               " >" + shell_quote(log) + " 2>&1 < /dev/null &";
    const std::string ssh_port = bootstrap.ssh_port == 22
        ? std::string() : " -p " + std::to_string(bootstrap.ssh_port);
    const std::string ssh_trust = bootstrap.known_hosts_file.empty()
        ? std::string()
        : " -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=" +
              shell_quote(bootstrap.known_hosts_file);
    const std::string ssh = "ssh -o BatchMode=yes" + ssh_trust + ssh_port;
    const std::string command = ssh + " " + shell_quote(bootstrap.ssh_target) + " " + shell_quote(remote);
    std::printf("potluck-server: launch[%u] %s\n", index, command.c_str());
    std::fflush(stdout);
    const int rc = std::system(command.c_str());
    if (rc == 0) {
        return true;
    }
    const std::string tail_command = ssh + " " + shell_quote(bootstrap.ssh_target) + " " +
                                     shell_quote("tail -n 40 ~/potluck/" + log);
    std::fprintf(stderr, "potluck-server: SSH launch failed (exit %d): %s\n", rc, command.c_str());
    std::fprintf(stderr, "potluck-server: remote log tail command: %s\n", tail_command.c_str());
    std::system(tail_command.c_str());
    return false;
}

void launch_local_workers(const std::string & worker_path,
                          const std::vector<std::string> & models,
                          const std::vector<ring_worker> & workers,
                          const std::string & result_endpoint) {
    for (uint32_t index = 0; index < models.size(); ++index) {
        const pid_t pid = fork();
        if (pid < 0) {
            throw std::runtime_error("cannot fork potluck-worker");
        }
        if (pid == 0) {
            std::vector<std::string> args = {
                worker_path, models[index],
                "--bind", workers[index].bind_endpoint,
                "--next", workers[index].next_endpoint,
                "--result", result_endpoint,
                "--rank", std::to_string(index)
            };
            std::vector<char *> argv;
            argv.reserve(args.size() + 1);
            for (std::string & arg : args) {
                argv.push_back(arg.data());
            }
            argv.push_back(nullptr);
            execv(worker_path.c_str(), argv.data());
            std::_Exit(127);
        }
    }
}

void configure_ring(ServerRing & ring, uint32_t n_layer, uint32_t n_ctx,
                    uint32_t n_seq_max, uint32_t n_ubatch, uint32_t seed,
                    float temp, float top_p) {
    const uint32_t n_workers = static_cast<uint32_t>(ring.workers.size());
    if (n_workers == 0 || ring.windows.empty()) {
        throw std::runtime_error("cannot configure an empty ring");
    }
    constexpr int handshake_timeout_ms = 120000;
    constexpr int decode_timeout_ms = 120000;

    ring.controls.reserve(n_workers);
    for (uint32_t index = 0; index < n_workers; ++index) {
        potluck::ring_sender sender = potluck::ring_sender::connect(
            ring.workers[index].connect_endpoint, ring.error);
        if (!sender.valid()) {
            throw std::runtime_error("cannot connect ring control sender for worker " +
                                     std::to_string(index) + ": " + ring.error);
        }
        if (!sender.set_send_timeout(handshake_timeout_ms, ring.error)) {
            throw std::runtime_error("cannot set ring control timeout: " + ring.error);
        }
        ring.controls.push_back(std::move(sender));
    }
    ring.ingress = potluck::ring_sender::connect(ring.workers.front().connect_endpoint, ring.error);
    if (!ring.ingress.valid()) {
        throw std::runtime_error("cannot connect ring ingress sender: " + ring.error);
    }
    if (!ring.ingress.set_send_timeout(decode_timeout_ms, ring.error)) {
        throw std::runtime_error("cannot set ring ingress timeout: " + ring.error);
    }

    for (uint32_t index = 0; index < n_workers; ++index) {
        potluck::node_config config;
        config.n_workers = n_workers;
        config.index = index;
        config.n_layer = n_layer;
        config.n_ctx = n_ctx;
        config.n_seq_max = n_seq_max;
        config.n_ubatch = n_ubatch;
        config.seed = seed;
        config.temp = temp;
        config.top_p = top_p;
        config.windows = ring.windows;
        std::vector<uint8_t> payload;
        if (!potluck::encode_config(config, payload)) {
            throw std::runtime_error("cannot encode ring worker configuration");
        }
        potluck::message message;
        message.type = potluck::message_type::node_config;
        message.rank = index;
        message.payload = std::move(payload);
        if (!ring.controls[index].send(message, ring.error)) {
            throw std::runtime_error("cannot send ring worker configuration: " + ring.error);
        }
    }

    if (!ring.result.set_receive_timeout(handshake_timeout_ms, ring.error)) {
        throw std::runtime_error("cannot set ring ready timeout: " + ring.error);
    }
    for (uint32_t ready_count = 0; ready_count < n_workers; ++ready_count) {
        potluck::message ready;
        if (!ring.result.receive(ready, ring.error)) {
            throw std::runtime_error("ring worker readiness failed: " + ring.error);
        }
        if (ready.type != potluck::message_type::ready) {
            throw std::runtime_error("ring worker sent unexpected readiness message");
        }
    }
    if (!ring.result.set_receive_timeout(decode_timeout_ms, ring.error)) {
        throw std::runtime_error("cannot set ring result timeout: " + ring.error);
    }
    std::printf("potluck-server: ring ready (%u workers, %zu windows)\n",
                n_workers, ring.windows.size());
    for (size_t i = 0; i < ring.windows.size(); ++i) {
        const potluck::ring_window & window = ring.windows[i];
        std::printf("potluck-server: ring window %zu owner=%u layers=[%u,%u) n_gpu_layers=%d\n",
                    i, window.owner, window.start, window.end, window.n_gpu_layers);
    }
    std::fflush(stdout);
}

std::vector<int32_t> drive_batch(ServerRing & ring,
                                 const std::vector<int32_t> & positions,
                                 const std::vector<int32_t> & sequences,
                                 const std::vector<int32_t> & tokens,
                                 int32_t clear_seq, int32_t trim_seq, int32_t trim_to,
                                 uint32_t n_logits,
                                 serve_stats * stats = nullptr) {
    if (positions.empty() || positions.size() != sequences.size() || positions.size() != tokens.size()) {
        throw std::runtime_error("invalid batch dimensions");
    }
    potluck::message input;
    input.type = potluck::message_type::batch_decode;
    input.flags = 0;
    input.rank = 0;
    input.sequence = static_cast<uint64_t>(positions.back());
    if (!potluck::encode_batch_payload(positions, sequences, tokens, nullptr, 0,
                                       clear_seq, trim_seq, trim_to, n_logits, input.payload)) {
        throw std::runtime_error("cannot encode ring batch");
    }
    if (stats != nullptr) {
        stats->head_payload_bytes += input.payload.size();
    }
    if (!ring.ingress.send(input, ring.error)) {
        throw std::runtime_error("cannot inject ring batch: " + ring.error);
    }
    potluck::message output;
    if (!ring.result.receive(output, ring.error)) {
        throw std::runtime_error("ring result receiver closed: " + ring.error);
    }
    if (stats != nullptr) {
        stats->head_payload_bytes += output.payload.size();
    }
    if (output.type != potluck::message_type::batch_result) {
        throw std::runtime_error("unexpected ring result message");
    }
    if (output.flags != ring.windows.size()) {
        throw std::runtime_error("ring result stopped before completing its route");
    }
    std::vector<int32_t> result_positions, result_sequences, result_tokens;
    std::vector<float> result_hidden;
    int32_t ignored_clear = -1, ignored_trim_seq = -1, ignored_trim = -1;
    uint32_t ignored_logits = 0;
    if (!potluck::decode_batch_payload(output.payload.data(), output.payload.size(), 0,
                                       ignored_clear, ignored_trim_seq, ignored_trim, ignored_logits,
                                       result_positions, result_sequences, result_tokens,
                                       result_hidden, ring.error)) {
        throw std::runtime_error("cannot decode ring result: " + ring.error);
    }
    if (result_positions != positions || result_sequences != sequences || result_tokens.size() != positions.size()) {
        throw std::runtime_error("ring result entries do not match the request");
    }
    return result_tokens;
}

std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & text) {
    if (text.empty()) {
        throw std::runtime_error("prompt is empty");
    }
    return common_tokenize(vocab, text, true, true);
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    return common_token_to_piece(vocab, token, true);
}

std::string render_tokens(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string text;
    for (const llama_token token : tokens) {
        text += token_piece(vocab, token);
    }
    return text;
}
double peak_rss_mb() {
    rusage usage = {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0.0;
    }
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
}

std::vector<llama_token> serve(ServerRing & ring, const llama_vocab * vocab,
                               const std::vector<llama_token> & prompt,
                               uint32_t n_predict,
                               const std::function<void(const std::string &)> & emit,
                               serve_stats * stats = nullptr) {
    if (prompt.empty()) {
        throw std::runtime_error("prompt is empty");
    }
    std::vector<int32_t> positions(prompt.size());
    std::vector<int32_t> sequences(prompt.size(), 0);
    std::vector<int32_t> tokens(prompt.size());
    for (size_t i = 0; i < prompt.size(); ++i) {
        positions[i] = static_cast<int32_t>(i);
        tokens[i] = static_cast<int32_t>(prompt[i]);
    }
    const auto prefill_start = std::chrono::steady_clock::now();
    (void) drive_batch(ring, positions, sequences, tokens, -2, -1, -1, 1, stats);
    if (stats != nullptr) {
        stats->prefill_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - prefill_start).count();
    }

    std::vector<llama_token> generated;
    generated.reserve(n_predict);
    llama_token previous = prompt.back();
    uint32_t position = static_cast<uint32_t>(prompt.size());
    for (uint32_t i = 0; i < n_predict; ++i) {
        const auto decode_start = std::chrono::steady_clock::now();
        const std::vector<int32_t> result = drive_batch(ring,
            { static_cast<int32_t>(position) }, { 0 }, { static_cast<int32_t>(previous) }, -1, -1, -1, 1, stats);
        if (stats != nullptr) {
            stats->decode_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - decode_start).count();
        }
        if (result.empty()) {
            break;
        }
        const llama_token next = static_cast<llama_token>(result.front());
        if (llama_vocab_is_eog(vocab, next)) {
            break;
        }
        generated.push_back(next);
        if (emit) {
            emit(token_piece(vocab, next));
        }
        previous = next;
        ++position;
    }
    return generated;
}

std::string now_id() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::to_string(std::chrono::duration_cast<std::chrono::seconds>(now).count());
}

json error_json(const std::string & message) {
    return json{ { "error", message } };
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string model_path;
        std::string shard_dir;
        std::string host = "127.0.0.1";
        std::string hosts_spec;
        std::string launch;
        uint16_t http_port = 8080;
        uint32_t worker_local = 0;
        uint32_t n_predict_default = 24;
        uint32_t n_ctx = 4096;
        uint32_t n_seq_max = 1;
        uint32_t n_ubatch = 1; // one-token internal ubatches preserve llama-cli numerics
        uint32_t seed = 0;
        float temp = 0.0f;
        float top_p = 0.0f;
        bool bench = false;
        bool workers_option = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto take = [&](const char * option) -> std::string {
                if (++i >= argc) {
                    throw std::runtime_error(std::string("missing value for ") + option);
                }
                return argv[i];
            };
            if (arg == "-m" || arg == "--model") model_path = take("--model");
            else if (arg == "--shard-dir" || arg == "--shards") shard_dir = take("--shard-dir");
            else if (arg == "--host") host = take("--host");
            else if (arg == "--port") http_port = static_cast<uint16_t>(std::stoul(take("--port")));
            else if (arg == "--workers") {
                worker_local = static_cast<uint32_t>(std::stoul(take("--workers")));
                workers_option = true;
            }
            else if (arg == "--hosts") hosts_spec = take("--hosts");
            else if (arg == "--launch") launch = take("--launch");
            else if (arg == "--ctx") n_ctx = static_cast<uint32_t>(std::stoul(take("--ctx")));
            else if (arg == "--batch") n_seq_max = std::max<uint32_t>(1, static_cast<uint32_t>(std::stoul(take("--batch"))));
            else if (arg == "--temp") temp = std::stof(take("--temp"));
            else if (arg == "--top-p") top_p = std::stof(take("--top-p"));
            else if (arg == "--seed") seed = static_cast<uint32_t>(std::stoul(take("--seed")));
            else if (arg == "--n-predict") n_predict_default = static_cast<uint32_t>(std::stoul(take("--n-predict")));
            else if (arg == "--bench") bench = true;
            else throw std::runtime_error(
                "usage: potluck-server -m model.gguf [--workers N] [--shard-dir DIR] "
                "[--hosts a,b,c (internal discovery bootstrap)] [--launch ssh]");
        }
        if (model_path.empty()) {
            throw std::runtime_error("need -m model.gguf");
        }
        if (!launch.empty() && launch != "ssh") {
            throw std::runtime_error("--launch supports only ssh for internal discovery bootstrap");
        }
        if (!hosts_spec.empty() && workers_option) {
            throw std::runtime_error("--hosts is internal discovery-bootstrap input and cannot be combined with --workers");
        }
        if (hosts_spec.empty() && !launch.empty()) {
            throw std::runtime_error("--launch ssh requires internal discovery-bootstrap --hosts input");
        }
        if (!hosts_spec.empty() && launch != "ssh") {
            throw std::runtime_error("--hosts is internal discovery-bootstrap input and requires --launch ssh");
        }

        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.no_alloc = true;
        model_params.load_mode = LLAMA_LOAD_MODE_NONE;
        model_params.check_tensors = false;
        llama_model * meta = llama_model_load_from_file(model_path.c_str(), model_params);
        if (meta == nullptr) {
            throw std::runtime_error("cannot load model metadata");
        }
        const llama_vocab * vocab = llama_model_get_vocab(meta);
        const uint32_t n_layer = static_cast<uint32_t>(llama_model_n_layer(meta));
        if (n_layer == 0) {
            throw std::runtime_error("model metadata reports zero layers");
        }
        uint32_t head_dim = 0;
        char key_length_buf[64] = {0};
        if (llama_model_meta_val_str(meta, "attention.key_length", key_length_buf,
                                     sizeof(key_length_buf)) > 0) {
            head_dim = static_cast<uint32_t>(std::atof(key_length_buf));
        }
        if (head_dim == 0) {
            const int32_t n_head = llama_model_n_head(meta);
            head_dim = n_head > 0 ? static_cast<uint32_t>(llama_model_n_embd(meta) / n_head) : 0;
        }
        const std::string model_name = basename_of(model_path);
        common_chat_templates_ptr chat_templates = common_chat_templates_init(meta, "");

        std::vector<bootstrap_node> bootstrap_nodes;
        if (!hosts_spec.empty()) {
            for (const std::string & value : parse_hosts(hosts_spec)) {
                bootstrap_node bootstrap;
                bootstrap.ssh_target = value;
                bootstrap.ring_host = ring_host(value);
                bootstrap_nodes.push_back(std::move(bootstrap));
            }
        } else if (!workers_option) {
            std::string discovery_error;
            const std::vector<potluck::discovered_node> discovered =
                potluck::discover_nodes(3000, discovery_error);
            if (!discovery_error.empty()) {
                throw std::runtime_error("mDNS discovery failed: " + discovery_error);
            }
            const std::string known_hosts_file = discovery_known_hosts_file();
            for (const potluck::discovered_node & node : discovered) {
                const std::string label = !node.id.empty()
                    ? node.id : (node.instance.empty() ? std::string("<unnamed>") : node.instance);
                const auto skip = [&](const char * reason) {
                    std::fprintf(stderr,
                                 "potluck-server: skipping incompatible discovered node '%s': %s\n",
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
                if (node.host.empty()) {
                    skip("missing host");
                    continue;
                }
                if (node.ssh_user.empty()) {
                    skip("missing SSH user");
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
                bootstrap_nodes.push_back(std::move(bootstrap));
            }
            if (bootstrap_nodes.empty()) {
                throw std::runtime_error("mDNS discovery returned no eligible nodes");
            }
        }

        const bool remote = !bootstrap_nodes.empty();
        if (!remote && worker_local == 0) {
            throw std::runtime_error("need at least one ring worker");
        }
        const uint32_t n_workers = remote
            ? static_cast<uint32_t>(bootstrap_nodes.size()) : worker_local;
        if (n_workers == 0) {
            throw std::runtime_error("need at least one ring worker");
        }

        // These addresses are controller-private. Local ports are reserved before worker launch.
        std::vector<ring_worker> workers;
        workers.reserve(n_workers);
        std::vector<std::string> worker_models;
        worker_models.reserve(n_workers);
        for (uint32_t index = 0; index < n_workers; ++index) {
            const std::string address_host = remote ? bootstrap_nodes[index].ring_host : "127.0.0.1";
            const uint16_t port = remote ? bootstrap_nodes[index].ring_port : free_port();
            ring_worker worker;
            worker.address = { address_host, port };
            worker.connect_endpoint = tcp_endpoint(address_host, port);
            worker.bind_endpoint = tcp_endpoint(remote ? "0.0.0.0" : address_host, port);
            workers.push_back(std::move(worker));

            if (remote) {
                if (shard_dir.empty()) {
                    // SSH workers run from ~/potluck, so pass only the model name.
                    worker_models.push_back(model_name);
                } else {
                    worker_models.push_back(basename_of(
                        local_shard_path(shard_dir, model_path, index, n_workers)));
                }
            } else {
                worker_models.push_back(shard_dir.empty()
                    ? model_path : local_shard_path(shard_dir, model_path, index, n_workers));
            }
        }
        for (uint32_t index = 0; index < n_workers; ++index) {
            workers[index].next_endpoint = workers[(index + 1) % n_workers].connect_endpoint;
        }

        ServerRing ring;
        ring.workers = std::move(workers);
        ring.windows = build_ring_route(n_workers, n_layer);
        std::string result_error;
        ring.result = potluck::ring_receiver::bind("tcp://0.0.0.0:*", result_error);
        if (!ring.result.valid()) {
            throw std::runtime_error("cannot bind ring result receiver: " + result_error);
        }
        const std::string result_host = remote
            ? local_address_for_peer(bootstrap_nodes.front().ring_host, bootstrap_nodes.front().ring_port)
            : (host == "0.0.0.0" ? "127.0.0.1" : host);
        ring.result_endpoint = endpoint_host(ring.result.endpoint(), result_host);

        if (remote) {
            for (uint32_t index = 0; index < n_workers; ++index) {
                if (!launch_remote_worker(bootstrap_nodes[index], worker_models[index],
                                          ring.workers[index], ring.result_endpoint, index)) {
                    throw std::runtime_error("remote ring worker launch failed");
                }
            }
        } else {
            const std::string worker_path = exe_dir(argv[0]) + "/potluck-worker";
            if (worker_path == "/potluck-worker") {
                throw std::runtime_error("cannot locate potluck-worker beside potluck-server");
            }
            launch_local_workers(worker_path, worker_models, ring.workers, ring.result_endpoint);
        }
        uint64_t model_bytes = 0;
        if (shard_dir.empty()) {
            model_bytes = std::filesystem::file_size(model_path);
        } else {
            for (const std::string & path : worker_models) {
                model_bytes += std::filesystem::file_size(path);
            }
        }
        const std::vector<potluck::accel_profile> profiles =
            collect_accel_profiles(ring, n_workers);
        assign_gpu_layers(ring.windows, profiles, n_workers, model_bytes, n_layer,
                          llama_model_n_head_kv(meta), head_dim, n_ctx);
        std::fflush(stdout);
        configure_ring(ring, n_layer, n_ctx, n_seq_max, n_ubatch, seed, temp, top_p);

        httplib::Server server;
        std::mutex ring_mutex;
        const auto set_common_headers = [](httplib::Response & response) {
            response.set_header("Access-Control-Allow-Origin", "*");
            response.set_header("Cache-Control", "no-cache");
        };
        server.Options(R"(/.*)", [&](const httplib::Request &, httplib::Response & response) {
            set_common_headers(response);
            response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            response.set_header("Access-Control-Allow-Headers", "Content-Type");
            response.status = 204;
        });
        server.Get("/health", [&](const httplib::Request &, httplib::Response & response) {
            json health = { { "status", "ok" }, { "workers", ring.workers.size() }, { "windows", json::array() } };
            for (size_t i = 0; i < ring.windows.size(); ++i) {
                const potluck::ring_window & window = ring.windows[i];
                const ring_worker & worker = ring.workers[window.owner];
                health["windows"].push_back(json{
                    { "index", i }, { "owner", window.owner },
                    { "host", worker.address.host }, { "port", worker.address.port },
                    { "start", window.start }, { "end", window.end },
                    { "n_gpu_layers", window.n_gpu_layers }
                });
            }
            set_common_headers(response);
            response.set_content(health.dump(), "application/json");
        });
        server.Get("/v1/models", [&](const httplib::Request &, httplib::Response & response) {
            json result = { { "object", "list" }, { "data", json::array({ json{
                { "id", model_name }, { "object", "model" }, { "owned_by", "potluck" }
            } }) } };
            set_common_headers(response);
            response.set_content(result.dump(), "application/json");
        });

        auto handle = [&](const httplib::Request & request, httplib::Response & response, bool chat) {
            std::shared_ptr<std::unique_lock<std::mutex>> busy =
                std::make_shared<std::unique_lock<std::mutex>>(ring_mutex, std::defer_lock);
            if (!busy->try_lock()) {
                response.status = 429;
                set_common_headers(response);
                response.set_content(error_json("ring is busy; retry later").dump(), "application/json");
                return;
            }
            json req;
            try {
                req = json::parse(request.body);
            } catch (const std::exception & e) {
                response.status = 400;
                set_common_headers(response);
                response.set_content(error_json(std::string("invalid JSON: ") + e.what()).dump(), "application/json");
                return;
            }

            std::string prompt_text;
            uint32_t n_predict = n_predict_default;
            try {
                if (chat) {
                    if (!req.contains("messages") || !req["messages"].is_array() || req["messages"].empty()) {
                        throw std::runtime_error("missing messages");
                    }
                    common_chat_templates_inputs inputs;
                    try {
                        inputs.messages = common_chat_msgs_parse_oaicompat(req["messages"]);
                    } catch (const std::exception & e) {
                        throw std::runtime_error(std::string("invalid messages: ") + e.what());
                    }
                    if (req.contains("reasoning_effort") && req["reasoning_effort"].is_string() &&
                        req["reasoning_effort"].get<std::string>() == "none") {
                        inputs.enable_thinking = false;
                        inputs.chat_template_kwargs["enable_thinking"] = "false";
                    }
                    if (!chat_templates) {
                        throw std::runtime_error("model has no chat template");
                    }
                    prompt_text = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
                    n_predict = json_u32(req, "max_tokens", n_predict_default);
                } else {
                    if (!req.contains("prompt") || !req["prompt"].is_string() || req["prompt"].get<std::string>().empty()) {
                        throw std::runtime_error("missing prompt");
                    }
                    prompt_text = req["prompt"].get<std::string>();
                    n_predict = json_u32(req, "n_predict", n_predict_default);
                }
                const std::vector<llama_token> prompt = tokenize_prompt(vocab, prompt_text);
                const bool stream = json_bool(req, "stream", false);
                const std::string id = "chatcmpl-potluck-" + now_id();
                const std::string created = now_id();
                const auto common_chunk = [id, created, model_name](const json & choice) {
                    return json{
                        { "id", id }, { "object", "chat.completion.chunk" },
                        { "created", std::stoull(created) }, { "model", model_name },
                        { "choices", json::array({ choice }) }
                    }.dump();
                };

                if (stream) {
                    response.status = 200;
                    set_common_headers(response);
                    response.set_chunked_content_provider("text/event-stream; charset=utf-8",
                        [&, busy, prompt, chat, n_predict, id, created, common_chunk](size_t, httplib::DataSink & sink) mutable {
                            try {
                                if (chat) {
                                    const json role = { { "index", 0 }, { "delta", { { "role", "assistant" } } }, { "finish_reason", nullptr } };
                                    const std::string event = "data: " + common_chunk(role) + "\n\n";
                                    sink.write(event.data(), event.size());
                                }
                                (void) serve(ring, vocab, prompt, n_predict, [&](const std::string & piece) {
                                    json delta = chat ? json{ { "index", 0 }, { "delta", { { "content", piece } } }, { "finish_reason", nullptr } }
                                                       : json{ { "content", piece } };
                                    const std::string event = chat ? "data: " + common_chunk(delta) + "\n\n"
                                                                    : "data: " + delta.dump() + "\n\n";
                                    sink.write(event.data(), event.size());
                                });
                                if (chat) {
                                    const json final_choice = { { "index", 0 }, { "delta", json::object() }, { "finish_reason", "stop" } };
                                    const std::string event = "data: " + common_chunk(final_choice) + "\n\n";
                                    sink.write(event.data(), event.size());
                                }
                                const std::string done = "data: [DONE]\n\n";
                                sink.write(done.data(), done.size());
                                sink.done();
                                return true;
                            } catch (const std::exception & e) {
                                const json error_event = error_json(e.what());
                                const std::string event = "data: " + error_event.dump() + "\n\n";
                                sink.write(event.data(), event.size());
                                sink.done();
                                return false;
                            }
                        });
                    return;
                }

                const std::vector<llama_token> generated = serve(ring, vocab, prompt, n_predict, {});
                const std::string text = render_tokens(vocab, generated);
                json result;
                if (chat) {
                    result = {
                        { "id", id }, { "object", "chat.completion" }, { "created", std::stoull(created) },
                        { "model", model_name }, { "choices", json::array({ json{
                            { "index", 0 }, { "message", { { "role", "assistant" }, { "content", text } } },
                            { "finish_reason", "stop" }
                        } }) },
                        { "usage", { { "prompt_tokens", prompt.size() }, { "completion_tokens", generated.size() },
                                      { "total_tokens", prompt.size() + generated.size() } } }
                    };
                } else {
                    result = { { "content", text }, { "n_predict", generated.size() }, { "finish_reason", "stop" } };
                }
                set_common_headers(response);
                response.set_content(result.dump(), "application/json");
            } catch (const std::exception & e) {
                const std::string message = e.what();
                const bool client_error =
                    message == "missing prompt" || message == "prompt is empty" ||
                    message == "missing messages" || message.rfind("invalid ", 0) == 0;
                response.status = client_error ? 400 : 503;
                set_common_headers(response);
                response.set_content(error_json(message).dump(), "application/json");
            }
        };

        server.Post("/completion", [&](const httplib::Request & request, httplib::Response & response) {
            handle(request, response, false);
        });
        server.Post("/v1/chat/completions", [&](const httplib::Request & request, httplib::Response & response) {
            handle(request, response, true);
        });
        server.set_error_handler([&](const httplib::Request &, httplib::Response & response) {
            set_common_headers(response);
            if (response.status == 404) {
                response.set_content(error_json("not found").dump(), "application/json");
            }
        });

        std::printf("potluck-server: listening on http://%s:%u (%zu workers, %zu ring windows, model %s)\n",
                    host.c_str(), http_port, ring.workers.size(), ring.windows.size(), model_path.c_str());
        std::fflush(stdout);
        if (bench) {
            const std::vector<llama_token> bench_prompt = tokenize_prompt(vocab, "The capital of France is");
            serve_stats stats;
            const auto start = std::chrono::steady_clock::now();
            const std::vector<llama_token> bench_tokens = serve(ring, vocab, bench_prompt, 8, {}, &stats);
            const double wall = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            const double prefill = stats.prefill_seconds > 0.0
                ? bench_prompt.size() / stats.prefill_seconds : 0.0;
            const double decode = stats.decode_seconds > 0.0
                ? bench_tokens.size() / stats.decode_seconds : 0.0;
            const double aggregate = wall > 0.0 ? bench_tokens.size() / wall : 0.0;
            const double bytes_per_token = bench_tokens.empty()
                ? 0.0 : static_cast<double>(stats.head_payload_bytes) / bench_tokens.size();
            const uint64_t model_bytes = potluck::model_file_bytes(model_path);
            std::printf("bench ring route windows=%zu\n", ring.windows.size());
            for (size_t i = 0; i < ring.windows.size(); ++i) {
                const potluck::ring_window & window = ring.windows[i];
                const uint64_t bytes = model_bytes * (window.end - window.start) /
                    std::max<uint32_t>(1, n_layer);
                std::printf("bench ring window %zu owner=%u [%u,%u) weight-bytes=%llu n_gpu_layers=%d\n",
                            i, window.owner, window.start, window.end,
                            static_cast<unsigned long long>(bytes), window.n_gpu_layers);
            }
            std::printf("bench ring prefill-tok/s %.2f decode-tok/s %.2f aggregate-tok/s %.2f "
                        "ms/token %.2f wire-bytes/token %.1f head-peak-rss-mb %.1f\n",
                        prefill, decode, aggregate, aggregate > 0.0 ? 1000.0 / aggregate : 0.0,
                        bytes_per_token, peak_rss_mb());
            std::fflush(stdout);
        }
        if (!server.listen(host.c_str(), http_port)) {
            fail("cannot bind HTTP port " + std::to_string(http_port));
        }
        llama_model_free(meta);
        return 0;
    } catch (const std::exception & e) {
        fail(e.what());
    }
}
