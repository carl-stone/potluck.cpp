// potluck-server admission, discovery, profiling, and route planning.

#include "internal.h"

#include <cmath>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <unordered_map>
#include <map>
#include <memory>
#include <mutex>
#include <poll.h>
#include <set>
#include <system_error>
#include <sstream>
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
#if defined(__APPLE__)
#include <mach/mach.h>
#endif

static std::once_flag local_backend_once;

static void ensure_local_backend() {
    std::call_once(local_backend_once, [] {
        llama_backend_init();
        const enum ggml_backend_dev_type wanted[] = {
            GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_ACCEL
        };
        for (const enum ggml_backend_dev_type type : wanted) {
            for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(index);
                if (dev != nullptr && ggml_backend_dev_type(dev) == type) {
                    size_t free_bytes = 0;
                    size_t total_bytes = 0;
                    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
                }
            }
        }
    });
}

static void read_local_host_memory(uint64_t & free_bytes, uint64_t & total_bytes) {
    size_t free_size = 0;
    size_t total_size = 0;
    const ggml_backend_dev_t cpu =
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu != nullptr) {
        ggml_backend_dev_memory(cpu, &free_size, &total_size);
    }
#if !defined(_WIN32)
    const long page_size = sysconf(_SC_PAGESIZE);
    const long total_pages = sysconf(_SC_PHYS_PAGES);
    if (page_size > 0 && total_pages > 0) {
        const size_t system_total = static_cast<size_t>(page_size) *
                                    static_cast<size_t>(total_pages);
        if (total_size == 0) {
            total_size = system_total;
        }
#if defined(_SC_AVPHYS_PAGES)
        const long available_pages = sysconf(_SC_AVPHYS_PAGES);
        if (available_pages > 0) {
            free_size = static_cast<size_t>(page_size) *
                        static_cast<size_t>(available_pages);
        }
#endif
    }
#if defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string name;
    uint64_t value = 0;
    std::string unit;
    while (meminfo >> name >> value >> unit) {
        if (name == "MemAvailable:" &&
            value <= std::numeric_limits<uint64_t>::max() / 1024u) {
            free_size = static_cast<size_t>(value * 1024u);
            break;
        }
    }
#endif
#if defined(__APPLE__)
    mach_port_t host = mach_host_self();
    vm_size_t mach_page_size = 0;
    if (host_page_size(host, &mach_page_size) == KERN_SUCCESS && mach_page_size > 0) {
        vm_statistics64_data_t statistics = {};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        if (host_statistics64(host, HOST_VM_INFO64,
                              reinterpret_cast<host_info64_t>(&statistics),
                              &count) == KERN_SUCCESS) {
            const uint64_t available_pages =
                static_cast<uint64_t>(statistics.free_count) +
                static_cast<uint64_t>(statistics.inactive_count) +
                static_cast<uint64_t>(statistics.speculative_count);
            free_size = static_cast<size_t>(available_pages) *
                        static_cast<size_t>(mach_page_size);
        }
    }
    mach_port_deallocate(mach_task_self(), host);
#endif
#endif
    free_bytes = static_cast<uint64_t>(free_size);
    total_bytes = static_cast<uint64_t>(total_size);
}

static device_probe probe_local_pressure_fast() {
    device_probe result;
    result.host = "127.0.0.1";
    result.profile.rank = 0;
#if defined(__APPLE__)
    result.profile.os = potluck::os_kind::macos;
#elif defined(__linux__)
    result.profile.os = potluck::os_kind::linux_os;
#else
    result.error = "unsupported operating system for local pressure probe";
    return result;
#endif
    ensure_local_backend();
    read_local_host_memory(result.profile.host_free_bytes,
                           result.profile.host_total_bytes);
    const enum ggml_backend_dev_type wanted[] = {
        GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_ACCEL
    };
    for (const enum ggml_backend_dev_type type : wanted) {
        for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(index);
            if (dev == nullptr || ggml_backend_dev_type(dev) != type) {
                continue;
            }
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            if (total_bytes == 0) {
                continue;
            }
            result.profile.free_bytes = free_bytes;
            result.profile.total_bytes = total_bytes;
            const std::string name = ggml_backend_dev_name(dev);
            const std::string description = ggml_backend_dev_description(dev);
            if (name.rfind("MTL", 0) == 0 ||
                description.find("Metal") != std::string::npos) {
                result.profile.kind = potluck::accel_kind::metal;
            } else if (name.rfind("CUDA", 0) == 0 ||
                       name.rfind("ROCm", 0) == 0 ||
                       name.rfind("MUSA", 0) == 0 ||
                       description.find("CUDA") != std::string::npos) {
                result.profile.kind = potluck::accel_kind::cuda;
            } else {
                result.profile.kind = potluck::accel_kind::other;
            }
            result.ok = true;
            return result;
        }
    }
    result.ok = true;
    return result;
}



bool solve_ring_placement(const std::vector<device_probe> & candidates,
                          const halda_model_metadata & metadata,
                          uint32_t n_ubatch, bool head_participates,
                          const std::vector<uint32_t> & fixed_w,
                          int32_t k_override, double master_priority,
                          double gpu_mem_gib,
                          std::vector<device_probe> & active_devices,
                          std::vector<potluck::ring_window> & windows,
                          halda_solution * solution, std::string & error) {
    active_devices.clear();
    windows.clear();
    if (solution != nullptr) {
        *solution = {};
    }
    error.clear();
    if (candidates.empty()) {
        error = "HALDA has no reachable devices";
        return false;
    }
    if (metadata.n_layer < 2 || metadata.n_embd == 0 || metadata.n_ff == 0 ||
        metadata.n_head == 0 || metadata.n_head_kv == 0 || metadata.n_vocab == 0 ||
        metadata.n_ctx == 0 || n_ubatch == 0 || metadata.b == 0 ||
        metadata.ordered_types.empty() ||
        metadata.ordered_types.size() != metadata.flops_per_type.size()) {
        error = "HALDA model metadata is incomplete";
        return false;
    }
    if (!std::isfinite(gpu_mem_gib) || gpu_mem_gib < 0.0) {
        error = "HALDA GPU memory override is invalid";
        return false;
    }

    halda_options options;
    options.model.n_layer = metadata.n_layer;
    options.model.n_embd = metadata.n_embd;
    options.model.n_ff = metadata.n_ff;
    options.model.n_head = metadata.n_head;
    options.model.n_head_kv = metadata.n_head_kv;
    options.model.n_vocab = metadata.n_vocab;
    options.model.n_ctx = metadata.n_ctx;
    options.model.n_ubatch = n_ubatch;
    options.model.b = metadata.b;
    options.model.bi = metadata.bi;
    options.model.bo = metadata.bo;
    options.model.kv_per_layer = metadata.kv_per_layer;
    options.model.types.reserve(metadata.ordered_types.size());
    options.model.layer_flops.reserve(metadata.flops_per_type.size());
    for (size_t index = 0; index < metadata.ordered_types.size(); ++index) {
        options.model.types.push_back(
            static_cast<uint32_t>(metadata.ordered_types[index]));
        options.model.layer_flops.push_back(
            static_cast<double>(metadata.flops_per_type[index]));
    }
    options.k_override = k_override;
    options.master_priority = master_priority;
    options.fixed_w = fixed_w;
    options.head_participates = head_participates;

    constexpr long double gib_bytes = 1024.0L * 1024.0L * 1024.0L;
    if (static_cast<long double>(gpu_mem_gib) >
        static_cast<long double>(std::numeric_limits<uint64_t>::max()) / gib_bytes) {
        error = "HALDA GPU memory override is too large";
        return false;
    }
    const uint64_t gpu_cap = gpu_mem_gib == 0.0
        ? 0 : static_cast<uint64_t>(static_cast<long double>(gpu_mem_gib) * gib_bytes);
    options.devices.reserve(candidates.size());
    for (size_t index = 0; index < candidates.size(); ++index) {
        const device_probe & candidate = candidates[index];
        if (!candidate.ok) {
            error = "HALDA candidate " + candidate.host + " was not profiled";
            return false;
        }
        halda_device device;
        device.original_rank = static_cast<uint32_t>(index);
        device.name = candidate.host.empty()
            ? "device " + std::to_string(index) : candidate.host;
        device.head = head_participates && index == 0;
        switch (candidate.profile.os) {
        case potluck::os_kind::macos:
            device.profile.os = halda_os_kind::macos;
            break;
        case potluck::os_kind::linux_os:
            device.profile.os = halda_os_kind::linux_os;
            break;
        default:
            error = "HALDA candidate " + candidate.host + " has an unsupported operating system";
            return false;
        }
        switch (candidate.profile.kind) {
        case potluck::accel_kind::none:
            device.profile.accel = halda_accel_kind::none;
            break;
        case potluck::accel_kind::metal:
            device.profile.accel = halda_accel_kind::metal;
            break;
        case potluck::accel_kind::cuda:
            device.profile.accel = halda_accel_kind::cuda;
            break;
        case potluck::accel_kind::other:
            device.profile.accel = halda_accel_kind::none;
            std::fprintf(stderr,
                         "potluck-server: device %s has an unsupported accelerator; using CPU only\n",
                         candidate.host.c_str());
            break;
        default:
            error = "HALDA candidate " + candidate.host + " has an unsupported accelerator";
            return false;
        }
        device.profile.free_bytes = candidate.profile.free_bytes;
        device.profile.total_bytes = candidate.profile.total_bytes;
        device.profile.host_free_bytes = candidate.profile.host_free_bytes;
        device.profile.host_total_bytes = candidate.profile.host_total_bytes;
        device.profile.cpu_gflops.assign(candidate.profile.cpu_gflops.begin(),
                                         candidate.profile.cpu_gflops.end());
        if (device.profile.accel != halda_accel_kind::none) {
            device.profile.accel_gflops.assign(candidate.profile.accel_gflops.begin(),
                                               candidate.profile.accel_gflops.end());
        }
        device.profile.mem_copy_delay_ms = candidate.profile.mem_copy_delay_ms;
        device.profile.accel_copy_delay_ms = candidate.profile.accel_copy_delay_ms;
        device.profile.disk_read_seq_gbps = candidate.profile.disk_read_seq_gbps;
        device.profile.disk_read_rnd_gbps = candidate.profile.disk_read_rnd_gbps;
        device.profile.n_cpu_threads = candidate.profile.n_cpu_threads;

        if (candidate.placement_usable_limit != std::numeric_limits<uint64_t>::max()) {
            device.profile.free_bytes = std::min(
                device.profile.free_bytes, candidate.placement_usable_limit);
            device.profile.host_free_bytes = std::min(
                device.profile.host_free_bytes, candidate.placement_usable_limit);
        }
        if (gpu_cap != 0 && device.profile.accel != halda_accel_kind::none) {
            device.profile.free_bytes = std::min(device.profile.free_bytes, gpu_cap);
        }
        if (device.profile.accel == halda_accel_kind::none) {
            device.profile.free_bytes = 0;
            device.profile.total_bytes = 0;
        }
        options.devices.push_back(std::move(device));
    }

    halda_solution solved;
    std::string solve_error;
    if (!solve_halda(options, solved, solve_error)) {
        error = "HALDA solve failed: " +
            (solve_error.empty() ? std::string("no feasible placement") : solve_error);
        return false;
    }
    std::vector<halda_route_window> solved_route;
    if (!build_halda_route(solved, options.devices, solved_route, solve_error)) {
        error = "HALDA route failed: " +
            (solve_error.empty() ? std::string("invalid solution") : solve_error);
        return false;
    }

    std::unordered_map<uint32_t, uint32_t> active_index;
    active_index.reserve(solved.active_original_ranks.size());
    for (const uint32_t rank : solved.active_original_ranks) {
        if (rank >= candidates.size() ||
            active_index.emplace(rank, static_cast<uint32_t>(active_devices.size())).second == false) {
            error = "HALDA returned an invalid active device rank";
            active_devices.clear();
            return false;
        }
        active_devices.push_back(candidates[rank]);
    }
    windows.reserve(solved_route.size());
    for (const halda_route_window & source : solved_route) {
        const auto found = active_index.find(source.owner);
        if (found == active_index.end()) {
            error = "HALDA route references an excluded device";
            active_devices.clear();
            windows.clear();
            return false;
        }
        windows.push_back({ found->second, source.start, source.end,
                            source.n_gpu_layers });
    }
    if (windows.empty()) {
        error = "HALDA returned an empty ring route";
        active_devices.clear();
        return false;
    }
    if (solution != nullptr) {
        *solution = std::move(solved);
    }
    return true;
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
        std::fprintf(stderr, "potluck-server: mDNS discovery unavailable: %s\n",
                     discovery_error.c_str());
        return {};
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
    return result;
}

static std::vector<ggml_type> default_probe_types() {
    return { GGML_TYPE_F32 };
}

static bool parse_probe_u64(const std::map<std::string, std::string> & fields,
                            const char * key, uint64_t & value, std::string & error) {
    const auto it = fields.find(key);
    if (it == fields.end() || it->second.empty() ||
        !std::all_of(it->second.begin(), it->second.end(), [](unsigned char c) {
            return std::isdigit(c) != 0;
        })) {
        error = std::string("probe is missing or invalid ") + key;
        return false;
    }
    size_t used = 0;
    try {
        const unsigned long long parsed = std::stoull(it->second, &used, 10);
        if (used != it->second.size()) {
            throw std::invalid_argument("trailing characters");
        }
        value = static_cast<uint64_t>(parsed);
        return true;
    } catch (const std::exception &) {
        error = std::string("probe has invalid ") + key;
        return false;
    }
}

static bool parse_probe_f32(const std::map<std::string, std::string> & fields,
                            const char * key, float & value, std::string & error) {
    const auto it = fields.find(key);
    if (it == fields.end() || it->second.empty()) {
        error = std::string("probe is missing ") + key;
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const float parsed = std::strtof(it->second.c_str(), &end);
    if (end == nullptr || *end != '\0' || errno == ERANGE ||
        !std::isfinite(parsed) || parsed < 0.0f) {
        error = std::string("probe has invalid ") + key;
        return false;
    }
    value = parsed;
    return true;
}

static bool parse_probe_vector(const std::map<std::string, std::string> & fields,
                               const char * key, size_t expected,
                               std::vector<float> & values, std::string & error) {
    const auto it = fields.find(key);
    if (it == fields.end()) {
        error = std::string("probe is missing ") + key;
        return false;
    }
    if (it->second == "-" || it->second.empty()) {
        error = std::string("probe has an empty ") + key;
        return false;
    }
    values.clear();
    size_t begin = 0;
    while (begin <= it->second.size()) {
        const size_t end = it->second.find(',', begin);
        const std::string token = it->second.substr(begin, end == std::string::npos
            ? std::string::npos : end - begin);
        if (token.empty()) {
            error = std::string("probe has an invalid ") + key + " vector";
            return false;
        }
        char * end_ptr = nullptr;
        errno = 0;
        const float value = std::strtof(token.c_str(), &end_ptr);
        if (end_ptr == nullptr || *end_ptr != '\0' || errno == ERANGE ||
            !std::isfinite(value) || value < 0.0f) {
            error = std::string("probe has an invalid ") + key + " value";
            return false;
        }
        values.push_back(value);
        if (values.size() > expected) {
            error = std::string("probe has too many ") + key + " values";
            return false;
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    if (values.size() != expected) {
        error = std::string("probe has the wrong ") + key + " vector length";
        return false;
    }
    return true;
}

static bool parse_probe_types(const std::map<std::string, std::string> & fields,
                              const std::vector<ggml_type> & expected,
                              std::string & error) {
    const auto it = fields.find("types");
    if (it == fields.end()) {
        error = "probe is missing types";
        return false;
    }
    size_t begin = 0;
    size_t count = 0;
    while (begin <= it->second.size()) {
        const size_t end = it->second.find(',', begin);
        const std::string token = it->second.substr(begin, end == std::string::npos
            ? std::string::npos : end - begin);
        if (count >= expected.size() || token != ggml_type_name(expected[count])) {
            error = "probe GGML type order does not match the request";
            return false;
        }
        ++count;
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    if (count != expected.size()) {
        error = "probe type count does not match the request";
        return false;
    }
    return true;
}

static bool parse_probe_line(const std::string & line, potluck::device_profile & profile,
                             std::string & error,
                             const std::vector<ggml_type> & requested_types = {},
                             bool pressure_only = false) {
    std::istringstream input(line);
    std::string prefix;
    if (!(input >> prefix) || prefix != "potluck-probe") {
        error = "invalid probe output prefix";
        return false;
    }
    std::map<std::string, std::string> fields;
    std::string token;
    while (input >> token) {
        const size_t equal = token.find('=');
        if (equal == std::string::npos || equal == 0 || equal + 1 == token.size()) {
            error = "invalid probe field";
            return false;
        }
        const std::string key = token.substr(0, equal);
        if (!fields.emplace(key, token.substr(equal + 1)).second) {
            error = "duplicate probe field: " + key;
            return false;
        }
    }
    const auto required = [&](const char * key) {
        return fields.find(key) != fields.end();
    };
    const auto protocol = fields.find("protocol");
    if (protocol == fields.end() || protocol->second != "EDP3") {
        error = "probe protocol/build mismatch";
        return false;
    }
    const auto build = fields.find("build");
    if (build == fields.end() || build->second != potluck_probe_build_id) {
        error = "probe protocol/build mismatch";
        return false;
    }
    const auto mode = fields.find("mode");
    if (mode == fields.end() ||
        (pressure_only ? mode->second != "pressure" : mode->second != "startup")) {
        error = "probe mode does not match the request";
        return false;
    }
    const auto os = fields.find("os");
    if (os == fields.end()) {
        error = "probe is missing operating system";
        return false;
    }
    if (os->second == "macos") {
        profile.os = potluck::os_kind::macos;
    } else if (os->second == "linux") {
        profile.os = potluck::os_kind::linux_os;
    } else if (os->second == "windows") {
        error = "Windows worker probes are not supported by Potluck yet";
        return false;
    } else {
        error = "probe has an unknown operating system";
        return false;
    }
    const auto kind = fields.find("kind");
    if (kind == fields.end()) {
        error = "probe is missing accelerator kind";
        return false;
    }
    if (kind->second == "none") {
        profile.kind = potluck::accel_kind::none;
    } else if (kind->second == "metal") {
        profile.kind = potluck::accel_kind::metal;
    } else if (kind->second == "cuda") {
        profile.kind = potluck::accel_kind::cuda;
    } else if (kind->second == "other") {
        profile.kind = potluck::accel_kind::other;
    } else {
        error = "unknown probe accelerator kind";
        return false;
    }
    uint64_t rank = 0;
    if (!parse_probe_u64(fields, "rank", rank, error) ||
        rank > std::numeric_limits<uint32_t>::max()) {
        error = "probe rank is invalid";
        return false;
    }
    profile.rank = static_cast<uint32_t>(rank);
    if (profile.rank != 0) {
        error = "probe rank is not zero";
        return false;
    }
    if (!parse_probe_u64(fields, "accel_free", profile.free_bytes, error) ||
        !parse_probe_u64(fields, "accel_total", profile.total_bytes, error) ||
        !parse_probe_u64(fields, "host_free", profile.host_free_bytes, error) ||
        !parse_probe_u64(fields, "host_total", profile.host_total_bytes, error)) {
        return false;
    }
    if (profile.host_total_bytes == 0 || profile.host_free_bytes > profile.host_total_bytes ||
        profile.free_bytes > profile.total_bytes ||
        (profile.kind == potluck::accel_kind::none && profile.total_bytes != 0) ||
        (profile.kind != potluck::accel_kind::none && profile.total_bytes == 0)) {
        error = "probe memory fields are inconsistent";
        return false;
    }
    if (!pressure_only) {
        uint64_t n_threads = 0;
        if (!parse_probe_u64(fields, "n_cpu_threads", n_threads, error) ||
            n_threads == 0 || n_threads > std::numeric_limits<uint32_t>::max()) {
            error = "probe CPU thread count is invalid";
            return false;
        }
        profile.n_cpu_threads = static_cast<uint32_t>(n_threads);
    }
    if (pressure_only) {
        for (const char * key : { "n_cpu_threads", "types", "cpu_gflops",
                                  "accel_gflops", "mem_copy_delay_ms",
                                  "accel_copy_delay_ms",
                                  "disk_read_seq_gbps", "disk_read_rnd_gbps" }) {
            if (required(key)) {
                error = "pressure probe contains benchmark fields";
                return false;
            }
        }
        return true;
    }
    const std::vector<ggml_type> expected = requested_types.empty()
        ? default_probe_types() : requested_types;
    if (expected.empty() || expected.size() > GGML_TYPE_COUNT ||
        !parse_probe_types(fields, expected, error) ||
        !parse_probe_vector(fields, "cpu_gflops", expected.size(),
                            profile.cpu_gflops, error) ||
        !parse_probe_vector(fields, "accel_gflops", expected.size(),
                            profile.accel_gflops, error) ||
        !parse_probe_f32(fields, "mem_copy_delay_ms", profile.mem_copy_delay_ms, error) ||
        !parse_probe_f32(fields, "accel_copy_delay_ms",
                         profile.accel_copy_delay_ms, error) ||
        !parse_probe_f32(fields, "disk_read_seq_gbps",
                         profile.disk_read_seq_gbps, error) ||
        !parse_probe_f32(fields, "disk_read_rnd_gbps",
                         profile.disk_read_rnd_gbps, error)) {
        return false;
    }
    return true;
}

static std::string probe_type_argument(const std::vector<ggml_type> & requested_types) {
    const std::vector<ggml_type> types = requested_types.empty()
        ? default_probe_types() : requested_types;
    std::string value;
    for (size_t index = 0; index < types.size(); ++index) {
        if (index != 0) {
            value += ',';
        }
        value += ggml_type_name(types[index]);
    }
    return value;
}

static std::string probe_command_arguments(const std::vector<ggml_type> & requested_types,
                                           uint64_t probe_bytes, bool pressure_only) {
    std::string result = " --probe --probe-types " +
        shell_quote(probe_type_argument(requested_types));
    if (probe_bytes != 0) {
        result += " --probe-bytes " + std::to_string(probe_bytes);
    }
    if (pressure_only) {
        result += " --probe-pressure";
    }
    return result;
}

static device_probe run_probe_command(const std::string & command, const std::string & host,
                                      const std::vector<ggml_type> & requested_types = {},
                                      uint64_t probe_bytes = 0,
                                      bool pressure_only = false) {
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
    if (line.empty() || (newline != std::string::npos &&
                         output.find_first_not_of(" \t\r\n", newline + 1) != std::string::npos)) {
        result.error = "probe returned extra or empty output";
        return result;
    }
    result.ok = parse_probe_line(line, result.profile, result.error,
                                 requested_types, pressure_only);
    return result;
}

static std::mutex startup_probe_cache_mutex;
static std::unordered_map<std::string, potluck::device_profile> startup_probe_cache;

static std::string startup_probe_cache_key(const std::string & scope,
                                           const std::string & identity,
                                           const std::vector<ggml_type> & requested_types,
                                           uint64_t probe_bytes) {
    return scope + ":" + identity + ":" + probe_type_argument(requested_types) +
           ":" + std::to_string(probe_bytes);
}

static bool load_startup_probe(const std::string & key, potluck::device_profile & profile) {
    std::lock_guard<std::mutex> lock(startup_probe_cache_mutex);
    const auto it = startup_probe_cache.find(key);
    if (it == startup_probe_cache.end()) {
        return false;
    }
    profile = it->second;
    return true;
}

static void save_startup_probe(const std::string & key,
                               const potluck::device_profile & profile) {
    std::lock_guard<std::mutex> lock(startup_probe_cache_mutex);
    startup_probe_cache[key] = profile;
}

device_probe probe_local_worker(const std::string & worker_path,
                                const std::vector<ggml_type> & requested_types,
                                uint64_t probe_bytes) {
    const std::string key = startup_probe_cache_key(
        "local", worker_path, requested_types, probe_bytes);
    potluck::device_profile cached;
    if (load_startup_probe(key, cached)) {
        device_probe result = probe_local_pressure(worker_path);
        if (!result.ok) {
            return result;
        }
        std::string error;
        if (!merge_pressure_profile(cached, result.profile, error)) {
            result.ok = false;
            result.error = error;
            return result;
        }
        result.profile = std::move(cached);
        return result;
    }
    ensure_local_backend();
    device_probe result = run_probe_command(
        shell_quote(worker_path) +
            probe_command_arguments(requested_types, probe_bytes, false),
        "127.0.0.1", requested_types, probe_bytes, false);
    if (result.ok) {
        device_probe pressure = probe_local_pressure(worker_path);
        if (!pressure.ok) {
            return pressure;
        }
        std::string error;
        if (!merge_pressure_profile(result.profile, pressure.profile, error)) {
            result.ok = false;
            result.error = error;
            return result;
        }
        save_startup_probe(key, result.profile);
    }
    return result;
}

device_probe probe_remote_worker(const bootstrap_node & bootstrap,
                                 const std::vector<ggml_type> & requested_types,
                                 uint64_t probe_bytes) {
    const std::string identity = bootstrap.ssh_target + ":" +
        std::to_string(bootstrap.ssh_port) + ":" + bootstrap.known_hosts_file;
    const std::string key = startup_probe_cache_key(
        "remote", identity, requested_types, probe_bytes);
    potluck::device_profile cached;
    if (load_startup_probe(key, cached)) {
        device_probe result = probe_remote_pressure(bootstrap);
        if (!result.ok) {
            return result;
        }
        std::string error;
        if (!merge_pressure_profile(cached, result.profile, error)) {
            result.ok = false;
            result.error = error;
            return result;
        }
        result.profile = std::move(cached);
        return result;
    }
    const std::string inner = "cd ~/potluck || exit 1; "
        "LD_LIBRARY_PATH=.${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} ./potluck-worker" +
        probe_command_arguments(requested_types, probe_bytes, false);
    const std::string command = ssh_options(bootstrap) + " " +
        shell_quote(bootstrap.ssh_target) + " " + shell_quote(inner);
    device_probe result = run_probe_command(command, bootstrap.ring_host,
                                            requested_types, probe_bytes, false);
    if (result.ok) {
        save_startup_probe(key, result.profile);
    }
    return result;
}

device_probe probe_local_pressure(const std::string & worker_path) {
    (void) worker_path;
    return probe_local_pressure_fast();
}


device_probe probe_remote_pressure(const bootstrap_node & bootstrap) {
    const std::vector<ggml_type> types = default_probe_types();
    const std::string inner = "cd ~/potluck || exit 1; "
        "LD_LIBRARY_PATH=.${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} ./potluck-worker" +
        probe_command_arguments(types, 0, true);
    const std::string command = ssh_options(bootstrap) + " " + shell_quote(bootstrap.ssh_target) +
        " " + shell_quote(inner);
    device_probe result = run_probe_command(command, bootstrap.ring_host, types, 0, true);
    result.bootstrap = bootstrap;
    return result;
}

bool merge_pressure_profile(potluck::device_profile & target,
                            const potluck::device_profile & pressure,
                            std::string & error) {
    if (target.rank != pressure.rank || target.os != pressure.os ||
        pressure.os == potluck::os_kind::none ||
        pressure.host_total_bytes == 0 ||
        pressure.host_free_bytes > pressure.host_total_bytes ||
        pressure.free_bytes > pressure.total_bytes ||
        ((pressure.kind == potluck::accel_kind::none && pressure.total_bytes != 0) ||
         (pressure.kind != potluck::accel_kind::none && pressure.total_bytes == 0))) {
        error = "pressure profile is inconsistent with the static profile";
        return false;
    }
    target.kind = pressure.kind;
    target.free_bytes = pressure.free_bytes;
    target.total_bytes = pressure.total_bytes;
    target.host_free_bytes = pressure.host_free_bytes;
    target.host_total_bytes = pressure.host_total_bytes;
    error.clear();
    return true;
}

std::vector<device_probe> probe_remote_candidates(
    const std::vector<bootstrap_node> & candidates,
    const std::vector<ggml_type> & requested_types,
    uint64_t probe_bytes) {
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
        std::thread thread([bootstrap, requested_types, probe_bytes,
                            promise = std::move(promise)]() mutable {
            try {
                promise.set_value(probe_remote_worker(bootstrap, requested_types, probe_bytes));
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

std::vector<device_probe> probe_remote_pressure_candidates(
    const std::vector<bootstrap_node> & candidates) {
    std::vector<device_probe> results;
    results.reserve(candidates.size());
    for (const bootstrap_node & candidate : candidates) {
        results.push_back(probe_remote_pressure(candidate));
    }
    return results;
}

struct halda_tensor_info {
    ggml_type type = GGML_TYPE_COUNT;
    std::array<int64_t, GGML_MAX_DIMS> ne {};
    uint64_t bytes = 0;
};

static bool gguf_integer(const gguf_context * ctx, const std::string & key,
                         uint64_t & value, std::string & error) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        error = "GGUF is missing " + key;
        return false;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8: value = gguf_get_val_u8(ctx, id); return true;
        case GGUF_TYPE_UINT16: value = gguf_get_val_u16(ctx, id); return true;
        case GGUF_TYPE_UINT32: value = gguf_get_val_u32(ctx, id); return true;
        case GGUF_TYPE_UINT64: value = gguf_get_val_u64(ctx, id); return true;
        case GGUF_TYPE_INT8: {
            const int8_t v = gguf_get_val_i8(ctx, id);
            if (v >= 0) { value = static_cast<uint64_t>(v); return true; }
            break;
        }
        case GGUF_TYPE_INT16: {
            const int16_t v = gguf_get_val_i16(ctx, id);
            if (v >= 0) { value = static_cast<uint64_t>(v); return true; }
            break;
        }
        case GGUF_TYPE_INT32: {
            const int32_t v = gguf_get_val_i32(ctx, id);
            if (v >= 0) { value = static_cast<uint64_t>(v); return true; }
            break;
        }
        case GGUF_TYPE_INT64: {
            const int64_t v = gguf_get_val_i64(ctx, id);
            if (v >= 0) { value = static_cast<uint64_t>(v); return true; }
            break;
        }
        default:
            break;
    }
    error = "GGUF metadata key is not a non-negative integer: " + key;
    return false;
}

static bool checked_add_u64(uint64_t left, uint64_t right, uint64_t & result) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

static bool checked_product(const std::array<int64_t, GGML_MAX_DIMS> & values,
                            uint64_t & result) {
    result = 1;
    for (const int64_t value : values) {
        if (value <= 0 ||
            static_cast<uint64_t>(value) >
                std::numeric_limits<uint64_t>::max() / result) {
            return false;
        }
        result *= static_cast<uint64_t>(value);
    }
    return true;
}

static bool block_name(const std::string & name, uint32_t & layer, std::string & suffix) {
    if (name.rfind("blk.", 0) != 0) {
        return false;
    }
    size_t pos = 4;
    const size_t begin = pos;
    while (pos < name.size() && std::isdigit(static_cast<unsigned char>(name[pos]))) {
        ++pos;
    }
    if (pos == begin || pos >= name.size() || name[pos] != '.') {
        return false;
    }
    try {
        const unsigned long parsed = std::stoul(name.substr(begin, pos - begin));
        if (parsed > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        layer = static_cast<uint32_t>(parsed);
    } catch (const std::exception &) {
        return false;
    }
    suffix = name.substr(pos + 1);
    return !suffix.empty();
}

bool extract_halda_model_metadata(const std::filesystem::path & model_path,
                                  uint32_t n_ctx,
                                  halda_model_metadata & metadata,
                                  std::string & error) {
    metadata = halda_model_metadata{};
    error.clear();
    if (n_ctx == 0) {
        error = "HALDA metadata needs a non-zero context";
        return false;
    }
    gguf_init_params params = {};
    params.no_alloc = true;
    gguf_context * ctx = gguf_init_from_file(model_path.string().c_str(), params);
    if (ctx == nullptr) {
        error = "cannot read GGUF metadata from " + model_path.string();
        return false;
    }
    const auto cleanup = [&] { gguf_free(ctx); };
    const int64_t arch_id = gguf_find_key(ctx, "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(ctx, arch_id) != GGUF_TYPE_STRING) {
        cleanup();
        error = "GGUF is missing general.architecture";
        return false;
    }
    const char * arch_value = gguf_get_val_str(ctx, arch_id);
    if (arch_value == nullptr || arch_value[0] == '\0') {
        cleanup();
        error = "GGUF has an empty architecture";
        return false;
    }
    const std::string arch = arch_value;
    uint64_t n_embd = 0;
    uint64_t n_ff = 0;
    uint64_t n_head = 0;
    if (!gguf_integer(ctx, arch + ".embedding_length", n_embd, error) ||
        !gguf_integer(ctx, arch + ".feed_forward_length", n_ff, error) ||
        !gguf_integer(ctx, arch + ".attention.head_count", n_head, error) ||
        n_embd == 0 || n_ff == 0 || n_head == 0 ||
        n_embd > std::numeric_limits<uint32_t>::max() ||
        n_ff > std::numeric_limits<uint32_t>::max() ||
        n_head > std::numeric_limits<uint32_t>::max()) {
        cleanup();
        if (error.empty()) {
            error = "GGUF model dimensions are invalid";
        }
        return false;
    }
    const int64_t token_tensor = gguf_find_tensor(ctx, "token_embd.weight");
    const int64_t * token_shape = token_tensor >= 0
        ? gguf_get_tensor_ne(ctx, token_tensor) : nullptr;
    if (token_shape == nullptr || token_shape[0] <= 0 || token_shape[1] <= 0 ||
        static_cast<uint64_t>(token_shape[0]) != n_embd ||
        static_cast<uint64_t>(token_shape[1]) > std::numeric_limits<uint32_t>::max()) {
        cleanup();
        error = "GGUF token embedding shape is invalid";
        return false;
    }
    const uint32_t n_vocab = static_cast<uint32_t>(token_shape[1]);
    uint64_t metadata_layers = 0;
    if (!gguf_integer(ctx, arch + ".block_count", metadata_layers, error) ||
        metadata_layers == 0 || metadata_layers > std::numeric_limits<uint32_t>::max()) {
        cleanup();
        if (error.empty()) {
            error = "GGUF has an invalid block count";
        }
        return false;
    }
    std::map<uint32_t, std::map<std::string, halda_tensor_info>> blocks;
    const int64_t tensor_count = gguf_get_n_tensors(ctx);
    for (int64_t index = 0; index < tensor_count; ++index) {
        const char * raw_name = gguf_get_tensor_name(ctx, index);
        if (raw_name == nullptr) {
            cleanup();
            error = "GGUF contains an unnamed tensor";
            return false;
        }
        uint32_t layer = 0;
        std::string suffix;
        if (!block_name(raw_name, layer, suffix)) {
            continue;
        }
        if (layer >= metadata_layers) {
            cleanup();
            error = "GGUF block tensor is outside the declared layer range";
            return false;
        }
        if (blocks[layer].find(suffix) != blocks[layer].end()) {
            cleanup();
            error = "GGUF contains a duplicate block tensor: " + std::string(raw_name);
            return false;
        }
        halda_tensor_info info;
        info.type = gguf_get_tensor_type(ctx, index);
        const int64_t * ne = gguf_get_tensor_ne(ctx, index);
        if (ne == nullptr || info.type < GGML_TYPE_F32 || info.type >= GGML_TYPE_COUNT) {
            cleanup();
            error = "GGUF block tensor has invalid shape or type: " + std::string(raw_name);
            return false;
        }
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            info.ne[dim] = ne[dim];
        }
        info.bytes = gguf_get_tensor_size(ctx, index);
        if (info.bytes == 0) {
            cleanup();
            error = "GGUF block tensor has zero bytes: " + std::string(raw_name);
            return false;
        }
        blocks[layer].emplace(std::move(suffix), info);
    }
    if (blocks.size() != metadata_layers) {
        cleanup();
        error = "GGUF block tensors are incomplete";
        return false;
    }
    const auto first = blocks.find(0);
    if (first == blocks.end() || first->second.empty()) {
        cleanup();
        error = "GGUF has no tensors for block 0";
        return false;
    }
    uint64_t total_block_bytes = 0;
    std::set<ggml_type> ordered_types;
    std::map<ggml_type, uint64_t> flops;
    for (uint64_t layer = 0; layer < metadata_layers; ++layer) {
        const auto current = blocks.find(static_cast<uint32_t>(layer));
        if (current == blocks.end() || current->second.empty()) {
            cleanup();
            error = "GGUF block tensors are incomplete";
            return false;
        }
        uint64_t current_bytes = 0;
        for (const auto & entry : current->second) {
            if (!checked_add_u64(current_bytes, entry.second.bytes, current_bytes)) {
                cleanup();
                error = "GGUF block bytes overflow";
                return false;
            }
            uint64_t elements = 0;
            if (!checked_product(entry.second.ne, elements) ||
                elements > (std::numeric_limits<uint64_t>::max() / 2)) {
                cleanup();
                error = "GGUF block FLOPs overflow";
                return false;
            }
            const uint64_t tensor_flops = elements * 2;
            uint64_t & type_flops = flops[entry.second.type];
            if (!checked_add_u64(type_flops, tensor_flops, type_flops)) {
                cleanup();
                error = "GGUF per-type FLOPs overflow";
                return false;
            }
            ordered_types.insert(entry.second.type);
        }
        if (!checked_add_u64(total_block_bytes, current_bytes, total_block_bytes)) {
            cleanup();
            error = "GGUF block bytes overflow";
            return false;
        }
    }
    const uint64_t average_block_bytes =
        total_block_bytes / metadata_layers +
        (total_block_bytes % metadata_layers != 0 ? 1 : 0);
    if (average_block_bytes == 0) {
        cleanup();
        error = "GGUF block tensors have zero bytes";
        return false;
    }
    const auto tensor_bytes = [&](const char * name, uint64_t & bytes) {
        const int64_t id = gguf_find_tensor(ctx, name);
        if (id < 0) {
            error = std::string("GGUF is missing tensor ") + name;
            return false;
        }
        bytes = gguf_get_tensor_size(ctx, id);
        if (bytes == 0) {
            error = std::string("GGUF tensor has zero bytes: ") + name;
            return false;
        }
        return true;
    };
    uint64_t token_bytes = 0;
    uint64_t output_norm_bytes = 0;
    uint64_t output_bytes = 0;
    if (!tensor_bytes("token_embd.weight", token_bytes) ||
        !tensor_bytes("output_norm.weight", output_norm_bytes)) {
        cleanup();
        return false;
    }
    const int64_t output_id = gguf_find_tensor(ctx, "output.weight");
    if (output_id < 0) {
        output_bytes = token_bytes;
    } else {
        output_bytes = gguf_get_tensor_size(ctx, output_id);
        if (output_bytes == 0) {
            cleanup();
            error = "GGUF tensor has zero bytes: output.weight";
            return false;
        }
    }
    uint64_t n_head_kv = 0;
    uint64_t head_dim = 0;
    if (!gguf_integer(ctx, arch + ".attention.head_count_kv", n_head_kv, error) ||
        !gguf_integer(ctx, arch + ".attention.key_length", head_dim, error) ||
        n_head_kv == 0 || head_dim == 0 ||
        n_head_kv > std::numeric_limits<uint64_t>::max() / head_dim ||
        n_head_kv * head_dim > std::numeric_limits<uint64_t>::max() / 4 ||
        n_head_kv * head_dim * 4 > std::numeric_limits<uint64_t>::max() / n_ctx) {
        cleanup();
        if (error.empty()) {
            error = "GGUF KV metadata is invalid";
        }
        return false;
    }
    uint64_t kv_per_layer = n_head_kv * head_dim * 4 * n_ctx;
    uint64_t b_prime = 0;
    uint64_t output_total = 0;
    if (!checked_add_u64(output_norm_bytes, output_bytes, output_total) ||
        !checked_add_u64(average_block_bytes, kv_per_layer, b_prime)) {
        cleanup();
        error = "HALDA model byte constants overflow";
        return false;
    }
    const auto average_per_layer = [metadata_layers](uint64_t total) {
        return total / metadata_layers + (total % metadata_layers != 0 ? 1 : 0);
    };
    metadata.n_layer = static_cast<uint32_t>(metadata_layers);
    metadata.n_embd = static_cast<uint32_t>(n_embd);
    metadata.n_ff = static_cast<uint32_t>(n_ff);
    metadata.n_head = static_cast<uint32_t>(n_head);
    metadata.n_head_kv = static_cast<uint32_t>(n_head_kv);
    metadata.n_vocab = n_vocab;
    metadata.n_ctx = n_ctx;
    metadata.head_dim = static_cast<uint32_t>(head_dim);
    metadata.b = average_block_bytes;
    metadata.bi = token_bytes;
    metadata.bo = output_total;
    metadata.kv_per_layer = kv_per_layer;
    metadata.b_prime = b_prime;
    metadata.ordered_types.assign(ordered_types.begin(), ordered_types.end());
    metadata.flops_per_type.reserve(metadata.ordered_types.size());
    for (const ggml_type type : metadata.ordered_types) {
        const uint64_t value = average_per_layer(flops[type]);
        if (value == 0) {
            cleanup();
            error = "GGUF block FLOPs are too small to model";
            return false;
        }
        metadata.flops_per_type.push_back(value);
    }
    cleanup();
    return true;
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

    std::vector<std::pair<std::string, std::string>> staged_files;
    try {
        std::error_code iteration_error;
        for (const std::filesystem::directory_entry & entry :
             std::filesystem::directory_iterator(stage_dir, iteration_error)) {
            if (iteration_error) {
                break;
            }
            if (entry.is_symlink() || !entry.is_regular_file()) {
                throw std::runtime_error("staged payload contains a non-file entry");
            }
            staged_files.emplace_back(entry.path().filename().string(),
                                      local_sha256(entry.path().string()));
        }
        if (iteration_error) {
            throw std::system_error(iteration_error);
        }
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "potluck-server: cannot inspect staged payload on %s: %s\n",
                     bootstrap.ssh_target.c_str(), exception.what());
        return false;
    }
    std::sort(staged_files.begin(), staged_files.end());
    if (staged_files.empty()) {
        std::fprintf(stderr, "potluck-server: staged payload is empty on %s\n",
                     bootstrap.ssh_target.c_str());
        return false;
    }

    const auto cleanup_remote = [&] {
        const std::string cleanup = ssh + " " + target + " " +
            shell_quote("rm -rf ~/potluck/.incoming ~/potluck/.previous");
        (void) std::system(cleanup.c_str());
    };
    const std::string prepare = ssh + " " + target + " " +
        shell_quote("set -eu; mkdir -p ~/potluck; "
                    "rm -rf ~/potluck/.incoming ~/potluck/.previous; "
                    "mkdir -p ~/potluck/.incoming");
    if (std::system(prepare.c_str()) != 0) {
        std::fprintf(stderr, "potluck-server: cannot prepare payload directory on %s\n",
                     bootstrap.ssh_target.c_str());
        return false;
    }
    const std::string rsync = "rsync -a --whole-file -e " + shell_quote(ssh) + " " +
        shell_quote((stage_dir.string() + "/")) + " " +
        shell_quote(bootstrap.ssh_target + ":potluck/.incoming/");
    if (std::system(rsync.c_str()) != 0) {
        cleanup_remote();
        std::fprintf(stderr, "potluck-server: cannot refresh worker binaries on %s\n",
                     bootstrap.ssh_target.c_str());
        return false;
    }

    std::string verify_inner = "set -eu; cd ~/potluck/.incoming || exit 1; ";
    for (const auto & staged : staged_files) {
        const std::string name = shell_quote(staged.first);
        verify_inner += "expected=" + shell_quote(staged.second) + "; ";
        verify_inner += "actual=\"$( ("
            "sha256sum " + name + " 2>/dev/null || "
            "shasum -a 256 " + name + " 2>/dev/null) | cut -d' ' -f1)\"; ";
        verify_inner += "test \"$actual\" = \"$expected\"; ";
    }
    const std::string verify = ssh + " " + target + " " + shell_quote(verify_inner);
    if (std::system(verify.c_str()) != 0) {
        cleanup_remote();
        std::fprintf(stderr, "potluck-server: staged payload checksum failed on %s\n",
                     bootstrap.ssh_target.c_str());
        return false;
    }

    const std::string staged_probe_inner =
        "set -eu; cd ~/potluck || exit 1; "
        "LD_LIBRARY_PATH=.incoming${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} "
        "~/potluck/.incoming/potluck-worker --probe --probe-pressure";
    const std::string staged_probe = ssh + " " + target + " " +
        shell_quote(staged_probe_inner);
    if (std::system(staged_probe.c_str()) != 0) {
        cleanup_remote();
        std::fprintf(stderr, "potluck-server: staged worker start check failed on %s\n",
                     bootstrap.ssh_target.c_str());
        return false;
    }

    std::string install_inner =
        "set -eu; cd ~/potluck || exit 1; "
        "rollback() { status=$?; trap - 0; restore_status=0; "
        "if [ ! -e .previous/.complete ]; then "
        "rm -rf .incoming .previous || true; exit \"$status\"; fi; ";
    for (const auto & staged : staged_files) {
        const std::string name = shell_quote(staged.first);
        const std::string previous = ".previous/" + name;
        install_inner +=
            "if [ -e " + previous + " ] || [ -L " + previous + " ]; then "
            "if ! mv -f " + previous + " " + name +
            "; then restore_status=1; fi; "
            "else if ! rm -f " + name +
            "; then restore_status=1; fi; fi; ";
    }
    install_inner +=
        "if [ \"$restore_status\" -eq 0 ]; then "
        "rm -rf .incoming .previous || restore_status=1; "
        "else rm -rf .incoming || true; fi; "
        "if [ \"$restore_status\" -ne 0 ]; then "
        "echo 'potluck: worker binary rollback failed; .previous was kept' >&2; "
        "exit 1; fi; exit \"$status\"; }; "
        "trap rollback 0; rm -rf .previous; mkdir -p .previous; ";
    for (const auto & staged : staged_files) {
        const std::string name = shell_quote(staged.first);
        const std::string previous = ".previous/" + name;
        install_inner +=
            "if [ -e " + name + " ] || [ -L " + name + " ]; then "
            "cp -p " + name + " " + previous + "; fi; ";
    }
    install_inner += "touch .previous/.complete; ";
    for (const auto & staged : staged_files) {
        const std::string name = shell_quote(staged.first);
        install_inner += "mv -f .incoming/" + name + " " + name + "; ";
    }
    install_inner +=
        "if ! (LD_LIBRARY_PATH=.${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} "
        "~/potluck/potluck-worker --probe --probe-pressure); then exit 1; fi; "
        "trap - 0; rm -rf .incoming .previous";
    const std::string install = ssh + " " + target + " " +
        shell_quote(install_inner);
    if (std::system(install.c_str()) != 0) {
        std::fprintf(stderr, "potluck-server: cannot install worker binaries on %s\n",
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
bool ensure_remote_hf_artifact(const bootstrap_node & bootstrap,
                               const std::filesystem::path & remote_path,
                               const std::string & hf_repo,
                               const std::string & hf_file,
                               const std::string & hf_token,
                               const std::string & digest, bool offline,
                               std::string & error) {
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
    if (hf_repo.empty() || hf_file.empty()) {
        error = "Hugging Face model source is incomplete";
        return false;
    }
    const std::filesystem::path hf_file_path(hf_file);
    if (hf_file_path.is_absolute()) {
        error = "Hugging Face model file must be relative";
        return false;
    }
    for (const std::filesystem::path & part : hf_file_path) {
        if (part == "..") {
            error = "Hugging Face model file cannot contain '..'";
            return false;
        }
    }
    if (!valid_sha256_digest(digest)) {
        error = "Hugging Face model digest is invalid";
        return false;
    }

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

    if (remote_digest() == digest) {
        std::printf("potluck-server: %s already has %s\n",
                    bootstrap.ring_host.c_str(), remote.c_str());
        return true;
    }
    if (offline) {
        error = "remote model is missing and --offline is set";
        return false;
    }

    const std::filesystem::path parent = remote_path.parent_path();
    const std::string directory = "mkdir -p " +
        shell_quote(parent.empty() ? "." : parent.generic_string());
    const std::string url = "https://huggingface.co/" + hf_repo +
        "/resolve/main/" + hf_file;
    const std::string temporary = remote + ".part";
    const std::string curl =
        "curl -L --fail --retry 3 -o " + shell_quote(temporary) + " " +
        shell_quote(url);
    const std::string curl_authenticated =
        "curl -L --fail --retry 3 -H \"Authorization: Bearer $HF_TOKEN\" -o " +
        shell_quote(temporary) + " " + shell_quote(url);
    const std::string fetch =
        "set -eu; IFS= read -r HF_TOKEN; export HF_TOKEN; cd ~/potluck && " +
        directory + " && rm -f " + shell_quote(temporary) + " && if [ -n " +
        "\"$HF_TOKEN\" ]; then " + curl_authenticated + "; else " + curl +
        "; fi; actual=\"$(sha256sum " + shell_quote(temporary) +
        " 2>/dev/null || shasum -a 256 " + shell_quote(temporary) +
        " 2>/dev/null)\"; actual=\"${actual%% *}\"; if [ \"$actual\" != " +
        shell_quote(digest) + " ]; then rm -f " + shell_quote(temporary) +
        "; exit 1; fi; mv " + shell_quote(temporary) + " " +
        shell_quote(remote);
    std::printf("potluck-server: fetching %s directly on %s\n",
                remote.c_str(), bootstrap.ring_host.c_str());
    std::fflush(stdout);
    const std::string command = ssh + " " + shell_quote(bootstrap.ssh_target) +
        " " + shell_quote(fetch);
    FILE * fetch_pipe = popen(command.c_str(), "w");
    if (fetch_pipe == nullptr) {
        error = "cannot start remote Hugging Face download";
        return false;
    }
    const std::string token_line = hf_token + '\n';
    const bool token_sent =
        std::fwrite(token_line.data(), 1, token_line.size(), fetch_pipe) ==
        token_line.size() && std::fflush(fetch_pipe) == 0;
    const int fetch_status = pclose(fetch_pipe);
    if (!token_sent || fetch_status != 0) {
        error = "remote Hugging Face download failed";
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


// Workers report their device profile before asking for a schedule.
std::vector<potluck::device_profile> collect_device_profiles(ServerRing & ring, uint32_t n_workers) {
    constexpr int profile_timeout_ms = 120000;
    if (!ring.result.set_receive_timeout(profile_timeout_ms, ring.error)) {
        throw std::runtime_error("cannot set ring profile timeout: " + ring.error);
    }
    std::vector<potluck::device_profile> profiles(n_workers);
    std::vector<bool> seen(n_workers, false);
    for (uint32_t received = 0; received < n_workers; ++received) {
        potluck::message message;
        if (!ring.result.receive(message, ring.error)) {
            throw std::runtime_error("device profile collection stopped at " +
                                     std::to_string(received) + " of " +
                                     std::to_string(n_workers) + " workers: " + ring.error);
        }
        if (message.type != potluck::message_type::profile_result ||
            message.rank >= n_workers || seen[message.rank]) {
            throw std::runtime_error("ring worker sent an unexpected device profile");
        }
        if (!potluck::decode_device_profile(message.payload.data(), message.payload.size(),
                                            profiles[message.rank], ring.error)) {
            throw std::runtime_error("cannot decode device profile: " + ring.error);
        }
        if (profiles[message.rank].rank != message.rank) {
            throw std::runtime_error("device profile rank mismatch");
        }
        seen[message.rank] = true;
        const potluck::device_profile & profile = profiles[message.rank];
        std::printf("potluck-server: worker %u accelerator %s free %llu MiB total %llu MiB\n",
                    message.rank, accel_kind_name(profile.kind),
                    static_cast<unsigned long long>(profile.free_bytes / (1024ull * 1024ull)),
                    static_cast<unsigned long long>(profile.total_bytes / (1024ull * 1024ull)));
    }
    std::fflush(stdout);
    return profiles;
}
