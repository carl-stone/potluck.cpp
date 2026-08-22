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
#include <cmath>
#include <cctype>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <filesystem>
#include <fstream>
#include <memory>
#include <limits>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <poll.h>

#include <sys/resource.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <limits.h>
#endif

using json = nlohmann::ordered_json;

namespace {

[[noreturn]] void fail(const std::string & message) {
    std::fprintf(stderr, "potluck-server: %s\n", message.c_str());
    std::fflush(nullptr);
    std::_Exit(1); // startup failures skip the known ggml-metal teardown assert
}
volatile sig_atomic_t signal_wakeup_fd = -1;

void signal_wakeup_handler(int) {
    const int fd = static_cast<int>(signal_wakeup_fd);
    if (fd >= 0) {
        const char byte = 1;
        (void) ::write(fd, &byte, sizeof(byte));
    }
}

class server_signal_wakeup {
public:
    explicit server_signal_wakeup(std::function<void()> stop) : stop_(std::move(stop)) {
        if (::pipe(pipe_) != 0) {
            throw std::runtime_error(
                "cannot create signal wakeup pipe: " + std::string(std::strerror(errno)));
        }
        try {
            set_cloexec(pipe_[0]);
            set_cloexec(pipe_[1]);
            set_nonblocking(pipe_[1]);
            signal_wakeup_fd = pipe_[1];
            install(SIGINT, old_int_, int_installed_);
            install(SIGTERM, old_term_, term_installed_);
            waiter_ = std::thread([this] { wait_for_signal(); });
        } catch (...) {
            close();
            throw;
        }
    }

    ~server_signal_wakeup() {
        close();
    }

    server_signal_wakeup(const server_signal_wakeup &) = delete;
    server_signal_wakeup & operator=(const server_signal_wakeup &) = delete;

private:
    static void set_cloexec(int fd) {
        const int flags = fcntl(fd, F_GETFD);
        if (flags < 0 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) < 0) {
            throw std::runtime_error(
                "cannot configure signal wakeup pipe: " + std::string(std::strerror(errno)));
        }
    }

    static void set_nonblocking(int fd) {
        const int flags = fcntl(fd, F_GETFL, 0);
        if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
            throw std::runtime_error(
                "cannot configure signal wakeup pipe: " + std::string(std::strerror(errno)));
        }
    }

    static void install(int signo, struct sigaction & previous, bool & installed) {
        struct sigaction action = {};
        action.sa_handler = signal_wakeup_handler;
        if (sigemptyset(&action.sa_mask) != 0) {
            throw std::runtime_error(
                "cannot configure signal handler: " + std::string(std::strerror(errno)));
        }
        action.sa_flags = SA_RESTART;
        if (::sigaction(signo, &action, &previous) != 0) {
            throw std::runtime_error(
                "cannot install signal handler: " + std::string(std::strerror(errno)));
        }
        installed = true;
    }

    void wait_for_signal() noexcept {
        char byte = 0;
        for (;;) {
            const ssize_t count = ::read(pipe_[0], &byte, sizeof(byte));
            if (count > 0) {
                if (byte != 0) {
                    stop_();
                }
                return;
            }
            if (count < 0 && errno == EINTR) {
                continue;
            }
            return;
        }
    }

    void restore_signals() noexcept {
        if (term_installed_) {
            (void) ::sigaction(SIGTERM, &old_term_, nullptr);
            term_installed_ = false;
        }
        if (int_installed_) {
            (void) ::sigaction(SIGINT, &old_int_, nullptr);
            int_installed_ = false;
        }
        signal_wakeup_fd = -1;
    }

    void close() noexcept {
        restore_signals();
        if (waiter_.joinable()) {
            const char byte = 0;
            (void) ::write(pipe_[1], &byte, sizeof(byte));
            waiter_.join();
        }
        if (pipe_[0] >= 0) {
            (void) ::close(pipe_[0]);
            pipe_[0] = -1;
        }
        pipe_[1] = -1;
    }


    std::function<void()> stop_;
    int pipe_[2] = {-1, -1};
    std::thread waiter_;
    struct sigaction old_int_ = {};
    struct sigaction old_term_ = {};
    bool int_installed_ = false;
    bool term_installed_ = false;
};


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
bool valid_ssh_component(const std::string & value) {
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
std::string ipv4_address_for_host(const std::string & host) {
    in_addr literal = {};
    if (inet_pton(AF_INET, host.c_str(), &literal) == 1) {
        return host;
    }
    addrinfo hints = {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo * addresses = nullptr;
    const int resolve = getaddrinfo(host.c_str(), nullptr, &hints, &addresses);
    if (resolve != 0) {
        throw std::runtime_error("cannot resolve discovered node " + host + ": " +
                                 gai_strerror(resolve));
    }
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> guard(addresses, freeaddrinfo);
    for (const addrinfo * address = addresses; address != nullptr; address = address->ai_next) {
        if (address->ai_family != AF_INET) {
            continue;
        }
        char text[INET_ADDRSTRLEN] = {};
        const auto * sockaddr = reinterpret_cast<const sockaddr_in *>(address->ai_addr);
        if (inet_ntop(AF_INET, &sockaddr->sin_addr, text, sizeof(text)) != nullptr) {
            return text;
        }
    }
    throw std::runtime_error("cannot find an IPv4 address for discovered node " + host);
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

struct device_probe {
    std::string host;
    bootstrap_node bootstrap;
    potluck::accel_profile profile;
    bool ok = false;
    std::string error;
    uint64_t placement_usable_limit = std::numeric_limits<uint64_t>::max();

    uint64_t usable_bytes() const {
        constexpr uint64_t mib = 1024ull * 1024ull;
        const uint64_t accel_reserve = std::max<uint64_t>(512ull * mib, profile.total_bytes / 8);
        const uint64_t host_reserve = std::max<uint64_t>(2ull * 1024ull * mib,
                                                         profile.host_total_bytes / 8);
        const uint64_t accel = profile.free_bytes > accel_reserve
            ? profile.free_bytes - accel_reserve : 0;
        const uint64_t host = profile.host_free_bytes > host_reserve
            ? profile.host_free_bytes - host_reserve : 0;
        return std::min(std::max(accel, host), placement_usable_limit);
    }
};

enum class worker_kind {
    local,
    remote,
};
struct planned_worker {
    worker_kind kind = worker_kind::local;
    device_probe device;
    std::string model;
    ring_worker ring;
    pid_t local_pid = -1;
    pid_t remote_ssh_pid = -1;
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
struct ring_session {
    ServerRing ring;
    std::vector<planned_worker> workers;
    bool healthy = false;
    mutable std::mutex mutex;
};


struct serve_stats {
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    uint64_t head_payload_bytes = 0;
};

uint64_t saturating_add(uint64_t left, uint64_t right) {
    if (left > std::numeric_limits<uint64_t>::max() - right) {
        return std::numeric_limits<uint64_t>::max();
    }
    return left + right;
}

uint64_t saturating_mul(uint64_t left, uint64_t right) {
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

uint64_t proportional_layers(uint32_t n_layer, uint64_t usable, uint64_t capacity) {
    if (capacity == 0 || usable == 0 || n_layer == 0) {
        return 0;
    }
    const long double scaled = static_cast<long double>(n_layer) *
                               static_cast<long double>(usable) /
                               static_cast<long double>(capacity);
    return static_cast<uint64_t>(std::min<long double>(
        scaled, static_cast<long double>(std::numeric_limits<uint64_t>::max())));
}

std::vector<device_probe> admit_devices(std::vector<device_probe> candidates,
                                         uint32_t n_layer, uint64_t model_bytes,
                                         uint32_t n_head_kv, uint32_t head_dim,
                                         uint32_t n_ctx, bool allow_shortfall = false) {
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
    uint64_t capacity = 0;
    uint64_t max_layer_sum = 0;
    size_t prefix = 0;
    for (; prefix < positive.size(); ++prefix) {
        const uint64_t usable = positive[prefix].usable_bytes();
        capacity = saturating_add(capacity, usable);
        max_layer_sum = saturating_add(max_layer_sum, usable / layer_cost);
        if (max_layer_sum >= n_layer) {
            ++prefix;
            break;
        }
    }
    if ((prefix == 0 || max_layer_sum < n_layer) && !allow_shortfall) {
        uint64_t admitted = 0;
        for (const device_probe & candidate : positive) {
            admitted = saturating_add(admitted, candidate.usable_bytes());
        }
        throw std::runtime_error(
            "cluster memory is short: need " +
            std::to_string(needed / (1024ull * 1024ull)) + " MiB, admitted " +
            std::to_string(admitted / (1024ull * 1024ull)) + " MiB across " +
            std::to_string(positive.size()) + " devices");
    }
    const size_t admitted_count = max_layer_sum < n_layer ? positive.size() : prefix;
    for (size_t i = admitted_count; i < positive.size(); ++i) {
        std::fprintf(stderr, "potluck-server: excluding %s: no layer slot remains\n",
                     positive[i].host.c_str());
    }
    positive.resize(admitted_count);
    capacity = 0;
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
    uint64_t capacity = 0;
    for (const device_probe & device : devices) {
        const uint64_t bytes = device.usable_bytes();
        usable.push_back(bytes);
        max_layers.push_back(static_cast<uint32_t>(std::min<uint64_t>(
            n_layer, bytes / layer_cost)));
        capacity = saturating_add(capacity, bytes);
    }
    uint64_t max_layer_sum = 0;
    for (uint32_t count : max_layers) {
        max_layer_sum = saturating_add(max_layer_sum, count);
    }
    if (capacity == 0 || max_layer_sum < n_layer) {
        throw std::runtime_error("cluster memory is short: need " +
                                 std::to_string(route_needed_bytes(
                                     n_layer, model_bytes, n_head_kv, head_dim, n_ctx) /
                                     (1024ull * 1024ull)) +
                                 " MiB, admitted " + std::to_string(
                                     (capacity / (1024ull * 1024ull))) + " MiB across " +
                                 std::to_string(devices.size()) + " devices");
    }
    std::vector<uint32_t> share(devices.size(), 0);
    for (size_t i = 0; i < devices.size(); ++i) {
        const uint64_t raw = proportional_layers(n_layer, usable[i], capacity);
        share[i] = static_cast<uint32_t>(std::min<uint64_t>(
            max_layers[i], std::max<uint64_t>(max_layers[i] == 0 ? 0 : 1, raw)));
    }
    auto total_share = [&]() {
        uint64_t total = 0;
        for (uint32_t count : share) {
            total = saturating_add(total, count);
        }
        return total;
    };
    while (total_share() < n_layer) {
        size_t best = devices.size();
        for (size_t i = 0; i < devices.size(); ++i) {
            if (share[i] >= max_layers[i]) {
                continue;
            }
            if (best == devices.size() ||
                max_layers[i] - share[i] > max_layers[best] - share[best] ||
                (max_layers[i] - share[i] == max_layers[best] - share[best] && i < best)) {
                best = i;
            }
        }
        if (best == devices.size()) {
            throw std::runtime_error("cannot distribute model layers across admitted devices");
        }
        ++share[best];
    }
    while (total_share() > n_layer) {
        size_t best = devices.size();
        for (size_t i = 0; i < devices.size(); ++i) {
            if (share[i] == 0) {
                continue;
            }
            if (best == devices.size() || share[i] > share[best] ||
                (share[i] == share[best] && i > best)) {
                best = i;
            }
        }
        --share[best];
    }
    std::vector<potluck::ring_window> windows;
    uint32_t next_layer = 0;
    for (uint32_t cycle = 0; cycle < 2; ++cycle) {
        for (size_t i = 0; i < devices.size(); ++i) {
            const uint32_t span = cycle == 0
                ? share[i] / 2 + share[i] % 2 : share[i] / 2;
            if (span == 0) {
                continue;
            }
            windows.push_back({ static_cast<uint32_t>(i), next_layer, next_layer + span, 0 });
            next_layer += span;
        }
    }
    if (next_layer != n_layer) {
        throw std::runtime_error("capability route does not cover the model");
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

bool parse_probe_line(const std::string & line, potluck::accel_profile & profile,
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

device_probe run_probe_command(const std::string & command, const std::string & host) {
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
        " " + shell_quote("cd ~/potluck || exit 1; ./potluck-worker --probe");
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

std::string local_sha256(const std::string & path) {
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

std::string model_path_key(const std::filesystem::path & canonical_path) {
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

bool valid_sha256_digest(const std::string & digest) {
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

struct model_digest_cache {
    std::mutex mutex;
    std::filesystem::path path;
    std::filesystem::file_time_type mtime;
    uintmax_t size = 0;
    std::string digest;
    bool valid = false;

    static std::filesystem::path persistent_path(
        const std::filesystem::path & canonical_path) {
        const char * home = std::getenv("HOME");
        if (home == nullptr || home[0] == '\0') {
            return {};
        }
        return std::filesystem::path(home) / ".cache" / "potluck" /
               (model_path_key(canonical_path) + ".sha256");
    }

    std::string get(const std::filesystem::path & model_path) {
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
};

std::string command_output(const std::string & command) {
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

std::string file_contents(const std::filesystem::path & path) {
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

std::string first_file_line(const std::filesystem::path & path) {
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


bool ensure_remote_model(const bootstrap_node & bootstrap, const std::string & model_path,
                         const std::string & digest, std::string & error) {
    error.clear();
    const std::string name = basename_of(model_path);
    const std::string ssh = ssh_options(bootstrap);
    const std::string remote_check =
        "(cd ~/potluck && (sha256sum " + shell_quote(name) +
        " 2>/dev/null || shasum -a 256 " + shell_quote(name) +
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
                    bootstrap.ring_host.c_str(), name.c_str());
        return true;
    }
    std::printf("potluck-server: sending %s to %s (%llu MiB)\n", name.c_str(),
                bootstrap.ring_host.c_str(),
                static_cast<unsigned long long>(
                    std::filesystem::file_size(model_path) / (1024ull * 1024ull)));
    std::fflush(stdout);
    const std::string mkdir = ssh + " " + shell_quote(bootstrap.ssh_target) +
        " " + shell_quote("mkdir -p ~/potluck");
    if (std::system(mkdir.c_str()) != 0) {
        error = "cannot prepare remote model directory";
        return false;
    }
    const std::string rsync = "rsync -a --whole-file --partial --inplace -e " +
        shell_quote(ssh) + " " + shell_quote(model_path) + " " +
        shell_quote(bootstrap.ssh_target + ":potluck/" + name);
    if (std::system(rsync.c_str()) != 0) {
        error = "model transfer failed";
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

void terminate_child_process(pid_t pid) {
    if (pid <= 0) {
        return;
    }
    (void) kill(-pid, SIGTERM);
    (void) kill(pid, SIGTERM);
    int status = 0;
    for (int attempt = 0; attempt < 50; ++attempt) {
        const pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid || (result < 0 && errno == ECHILD)) {
            return;
        }
        if (result < 0 && errno != EINTR) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    (void) kill(-pid, SIGKILL);
    (void) kill(pid, SIGKILL);
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
}

pid_t launch_remote_worker(const bootstrap_node & bootstrap, const std::string & model,
                           const ring_worker & worker, const std::string & result_endpoint,
                           uint32_t index) {
    const std::string log = "worker-" + std::to_string(index) + ".log";
    const std::string pid_file = ".potluck-worker-" + std::to_string(index) + ".pid";
    const std::string remote = "cd ~/potluck || exit 1; nohup ./potluck-worker " +
                               shell_quote(model) +
                               " --bind " + shell_quote(worker.bind_endpoint) +
                               " --next " + shell_quote(worker.next_endpoint) +
                               " --result " + shell_quote(result_endpoint) +
                               " --rank " + std::to_string(index) +
                               " >" + shell_quote(log) + " 2>&1 < /dev/null & " +
                               "pid=$!; echo \"$pid\" > " + shell_quote(pid_file) + "; " +
                               "attempt=0; while [ \"$attempt\" -lt 120 ]; do " +
                               "case \"$(cat " + shell_quote(log) +
                               " 2>/dev/null)\" in *\"WORKER rank \"*) " +
                               "echo POTLUCK_WORKER_READY; wait \"$pid\"; exit $?;; esac; " +
                               "kill -0 \"$pid\" 2>/dev/null || exit 1; " +
                               "attempt=$((attempt + 1)); sleep 1; done; exit 1";
    const std::string ssh = ssh_options(bootstrap);
    const std::string command = ssh + " " + shell_quote(bootstrap.ssh_target) + " " + shell_quote(remote);
    std::printf("potluck-server: launch[%u] %s\n", index, command.c_str());
    std::fflush(stdout);

    int output_pipe[2] = {-1, -1};
    if (pipe(output_pipe) != 0) {
        throw std::runtime_error("cannot create SSH launch pipe: " + std::string(std::strerror(errno)));
    }
    const pid_t ssh_pid = fork();
    if (ssh_pid < 0) {
        close(output_pipe[0]);
        close(output_pipe[1]);
        throw std::runtime_error("cannot fork SSH launcher: " + std::string(std::strerror(errno)));
    }
    if (ssh_pid == 0) {
        close(output_pipe[0]);
        (void) setpgid(0, 0);
        const int input = open("/dev/null", O_RDONLY);
        if (input >= 0) {
            (void) dup2(input, STDIN_FILENO);
            close(input);
        }
        if (dup2(output_pipe[1], STDOUT_FILENO) < 0 ||
            dup2(output_pipe[1], STDERR_FILENO) < 0) {
            _exit(127);
        }
        close(output_pipe[1]);
        execl("/bin/sh", "sh", "-c", command.c_str(), static_cast<char *>(nullptr));
        _exit(127);
    }
    close(output_pipe[1]);
    (void) setpgid(ssh_pid, ssh_pid);

    std::string output;
    bool ready = false;
    int status = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (!ready) {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) {
            break;
        }
        struct pollfd descriptor = { output_pipe[0], POLLIN | POLLHUP, 0 };
        const int poll_timeout = static_cast<int>(std::min<int64_t>(remaining, 1000));
        const int poll_result = poll(&descriptor, 1, poll_timeout);
        if (poll_result < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        if (poll_result > 0 && (descriptor.revents & (POLLIN | POLLHUP)) != 0) {
            char buffer[1024];
            const ssize_t count = read(output_pipe[0], buffer, sizeof(buffer));
            if (count > 0) {
                output.append(buffer, static_cast<size_t>(count));
                if (output.size() > 4096) {
                    output.erase(0, output.size() - 4096);
                }
                ready = output.find("POTLUCK_WORKER_READY") != std::string::npos;
            } else if (count == 0) {
                break;
            }
        }
        if (!ready && waitpid(ssh_pid, &status, WNOHANG) == ssh_pid) {
            break;
        }
    }
    close(output_pipe[0]);
    if (ready) {
        return ssh_pid;
    }

    terminate_child_process(ssh_pid);
    const std::string tail_command = ssh + " " + shell_quote(bootstrap.ssh_target) + " " +
                                     shell_quote("tail -n 40 ~/potluck/" + log);
    std::fprintf(stderr, "potluck-server: SSH launch failed: %s\n", command.c_str());
    if (!output.empty()) {
        std::fprintf(stderr, "potluck-server: SSH launcher output: %s\n", output.c_str());
    }
    std::fprintf(stderr, "potluck-server: remote log tail command: %s\n", tail_command.c_str());
    std::system(tail_command.c_str());
    throw std::runtime_error("remote worker did not become ready");
}

pid_t launch_local_worker(const std::string & worker_path, const std::string & model,
                          const ring_worker & worker, const std::string & result_endpoint,
                          uint32_t index) {
    std::vector<std::string> args = {
        worker_path, model,
        "--bind", worker.bind_endpoint,
        "--next", worker.next_endpoint,
        "--result", result_endpoint,
        "--rank", std::to_string(index)
    };
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (std::string & arg : args) {
        argv.push_back(arg.data());
    }
    argv.push_back(nullptr);
    const pid_t pid = fork();
    if (pid < 0) {
        throw std::runtime_error("cannot fork potluck-worker");
    }
    if (pid != 0) {
        return pid;
    }
    execv(worker_path.c_str(), argv.data());
    std::_Exit(127);
}
bool stop_remote_workers(const bootstrap_node & bootstrap) {
    const std::string command =
        "cd ~/potluck || exit 1; setopt nonomatch 2>/dev/null || true; for pid_file in .potluck-worker-*.pid; do " +
        std::string("test -s \"$pid_file\" || continue; ") +
        "pid=$(cat \"$pid_file\"); rank=${pid_file#.potluck-worker-}; rank=${rank%.pid}; " +
        "command_line=$(ps -p \"$pid\" -o command= 2>/dev/null || true); " +
        "case \"$command_line\" in *potluck-worker*) ;; *) rm -f \"$pid_file\"; continue ;; esac; " +
        "case \"$command_line\" in *\"--rank ${rank}\"*) ;; *) rm -f \"$pid_file\"; continue ;; esac; " +
        "kill \"$pid\" 2>/dev/null || true; "
        "for attempt in 1 2 3 4 5; do "
        "kill -0 \"$pid\" 2>/dev/null || break; sleep 1; done; "
        "if kill -0 \"$pid\" 2>/dev/null; then kill -9 \"$pid\" 2>/dev/null || true; fi; "
        "for attempt in 1 2 3 4 5; do "
        "kill -0 \"$pid\" 2>/dev/null || break; sleep 1; done; "
        "if kill -0 \"$pid\" 2>/dev/null; then exit 1; fi; "
        "rm -f \"$pid_file\"; done";
    const std::string ssh = ssh_options(bootstrap) + " " +
        shell_quote(bootstrap.ssh_target) + " " + shell_quote(command);
    return std::system(ssh.c_str()) == 0;
}

void stop_planned_workers(const std::vector<planned_worker> & planned) {
    for (size_t index = 0; index < planned.size(); ++index) {
        const planned_worker & plan = planned[index];
        if (plan.kind == worker_kind::local) {
            if (plan.local_pid <= 0) {
                continue;
            }
            int status = 0;
            if (waitpid(plan.local_pid, &status, WNOHANG) == 0) {
                (void) kill(plan.local_pid, SIGTERM);
                for (int attempt = 0; attempt < 50; ++attempt) {
                    if (waitpid(plan.local_pid, &status, WNOHANG) == plan.local_pid) {
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                if (waitpid(plan.local_pid, &status, WNOHANG) == 0) {
                    (void) kill(plan.local_pid, SIGKILL);
                }
            }
            (void) waitpid(plan.local_pid, &status, 0);
            continue;
        }
        (void) stop_remote_workers(plan.device.bootstrap);
        terminate_child_process(plan.remote_ssh_pid);
    }
}

void launch_local_workers(const std::string & worker_path,
                          const std::vector<std::string> & models,
                          const std::vector<ring_worker> & workers,
                          const std::string & result_endpoint) {
    if (models.size() != workers.size()) {
        throw std::runtime_error("local worker model and endpoint counts differ");
    }
    for (uint32_t index = 0; index < models.size(); ++index) {
        launch_local_worker(worker_path, models[index], workers[index],
                            result_endpoint, index);
    }
}

void prepare_ring_controls(ServerRing & ring, uint32_t n_workers, int timeout_ms) {
    if (!ring.controls.empty()) {
        if (ring.controls.size() != n_workers) {
            throw std::runtime_error("ring control sender count mismatch");
        }
        return;
    }
    ring.controls.reserve(n_workers);
    for (uint32_t index = 0; index < n_workers; ++index) {
        potluck::ring_sender sender = potluck::ring_sender::connect(
            ring.workers[index].connect_endpoint, ring.error);
        if (!sender.valid()) {
            throw std::runtime_error("cannot connect ring control sender for worker " +
                                     std::to_string(index) + ": " + ring.error);
        }
        if (!sender.set_send_timeout(timeout_ms, ring.error)) {
            throw std::runtime_error("cannot set ring control timeout: " + ring.error);
        }
        ring.controls.push_back(std::move(sender));
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
    constexpr int decode_timeout_ms = 600000;

    prepare_ring_controls(ring, n_workers, handshake_timeout_ms);

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
    std::printf("potluck-server: waiting for ring ready on %s\n",
                ring.result.endpoint().c_str());
    std::fflush(stdout);

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
    ring.ingress = potluck::ring_sender::connect(ring.workers.front().connect_endpoint, ring.error);
    if (!ring.ingress.valid()) {
        throw std::runtime_error("cannot connect ring ingress sender: " + ring.error);
    }
    if (!ring.ingress.set_send_timeout(decode_timeout_ms, ring.error)) {
        throw std::runtime_error("cannot set ring ingress timeout: " + ring.error);
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
bool reset_ring_workers(ServerRing & ring, std::string & error) {
    error.clear();
    potluck::message reset;
    reset.type = potluck::message_type::reset;
    for (potluck::ring_sender & control : ring.controls) {
        if (!control.valid()) {
            continue;
        }
        std::string send_error;
        if (!control.set_send_timeout(1000, send_error) ||
            !control.send(reset, send_error)) {
            if (error.empty()) {
                error = send_error;
            }
        }
    }
    if (!ring.controls.empty()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return error.empty();
}


class request_cancelled final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::vector<int32_t> drive_batch(ServerRing & ring,
                                 const std::vector<int32_t> & positions,
                                 const std::vector<int32_t> & sequences,
                                 const std::vector<int32_t> & tokens,
                                 int32_t clear_seq, int32_t trim_seq, int32_t trim_to,
                                 uint32_t n_logits,
                                 serve_stats * stats = nullptr,
                                 std::function<bool()> should_cancel = {},
                                 bool * batch_started = nullptr) {
    if (batch_started != nullptr) {
        *batch_started = false;
    }
    const bool clear_only = positions.empty() && sequences.empty() && tokens.empty() &&
        clear_seq != -1;
    if ((!clear_only && (positions.empty() || positions.size() != sequences.size() ||
                         positions.size() != tokens.size())) ||
        (clear_only && (trim_seq >= 0 || trim_to >= 0 || n_logits != 0))) {
        throw std::runtime_error("invalid batch dimensions");
    }
    potluck::message input;
    input.type = potluck::message_type::batch_decode;
    input.flags = 0;
    input.sequence = positions.empty() ? 0 : static_cast<uint64_t>(positions.back());
    if (!potluck::encode_batch_payload(positions, sequences, tokens, nullptr, 0,
                                       clear_seq, trim_seq, trim_to, n_logits, input.payload)) {
        throw std::runtime_error("cannot encode ring batch");
    }
    if (stats != nullptr) {
        stats->head_payload_bytes += input.payload.size();
    }
    const auto cancelled = [&]() {
        return should_cancel && should_cancel();
    };
    if (cancelled()) {
        throw request_cancelled("request cancelled");
    }
    constexpr int poll_timeout_ms = 250;
    if (!ring.ingress.set_send_timeout(poll_timeout_ms, ring.error)) {
        throw std::runtime_error("cannot set ring ingress timeout: " + ring.error);
    }
    if (!ring.ingress.send(input, ring.error)) {
        if (cancelled()) {
            throw request_cancelled("request cancelled");
        }
        throw std::runtime_error("cannot inject ring batch: " + ring.error);
    }
    if (batch_started != nullptr) {
        *batch_started = true;
    }
    if (!ring.result.set_receive_timeout(poll_timeout_ms, ring.error)) {
        throw std::runtime_error("cannot set ring result timeout: " + ring.error);
    }
    potluck::message output;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(600);
    bool cancel_requested = false;
    std::chrono::steady_clock::time_point cancel_deadline;
    for (;;) {
        if (!cancel_requested && cancelled()) {
            cancel_requested = true;
            cancel_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        }
        if (ring.result.receive(output, ring.error)) {
            if (cancel_requested) {
                throw request_cancelled("request cancelled");
            }
            break;
        }
        if (ring.error.find("timeout") == std::string::npos) {
            throw std::runtime_error("ring result receiver closed: " + ring.error);
        }
        const auto now = std::chrono::steady_clock::now();
        if (cancel_requested && now >= cancel_deadline) {
            throw std::runtime_error("ring result receiver timed out after cancellation");
        }
        if (now >= deadline) {
            throw std::runtime_error("ring result receiver timed out");
        }
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

void configure_slot(ServerRing & ring, const potluck::slot_config & config) {
    potluck::message message;
    message.type = potluck::message_type::slot_config;
    message.flags = 0;
    message.sequence = static_cast<uint64_t>(std::max<int32_t>(0, config.seq));
    if (!potluck::encode_slot_config(config, message.payload)) {
        throw std::runtime_error("cannot encode slot configuration");
    }
    constexpr int poll_timeout_ms = 250;
    if (!ring.ingress.set_send_timeout(poll_timeout_ms, ring.error)) {
        throw std::runtime_error("cannot set ring ingress timeout: " + ring.error);
    }
    if (!ring.ingress.send(message, ring.error)) {
        throw std::runtime_error("cannot send slot configuration: " + ring.error);
    }
}

std::vector<llama_token> serve(ServerRing & ring, const llama_vocab * vocab,
                               const std::vector<llama_token> & prompt,
                               uint32_t n_predict,
                               const std::function<void(const std::string &)> & emit,
                               serve_stats * stats = nullptr,
                               int32_t sequence_id = 0,
                               potluck::slot_config sampling = {},
                               uint32_t prefill_batch = 512) {
    if (prompt.empty()) {
        throw std::runtime_error("prompt is empty");
    }
    sampling.seq = sequence_id;
    configure_slot(ring, sampling);
    const uint32_t batch_limit = std::max<uint32_t>(1, prefill_batch);
    const auto prefill_start = std::chrono::steady_clock::now();
    size_t offset = 0;
    while (offset < prompt.size()) {
        const size_t count = std::min<size_t>(batch_limit, prompt.size() - offset);
        std::vector<int32_t> positions(count);
        std::vector<int32_t> sequences(count, sequence_id);
        std::vector<int32_t> tokens(count);
        for (size_t i = 0; i < count; ++i) {
            positions[i] = static_cast<int32_t>(offset + i);
            tokens[i] = static_cast<int32_t>(prompt[offset + i]);
        }
        const bool final_chunk = offset + count == prompt.size();
        (void) drive_batch(ring, positions, sequences, tokens,
                           offset == 0 ? sequence_id : -1, -1, -1,
                           final_chunk ? 1 : 0, stats);
        offset += count;
    }
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
        const std::vector<int32_t> result = drive_batch(
            ring, { static_cast<int32_t>(position) }, { sequence_id },
            { static_cast<int32_t>(previous) }, -1, -1, -1, 1, stats);
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

enum class slot_state {
    free,
    queued,
    prefill,
    decode,
    done,
    cancelled,
};

const char * slot_state_name(slot_state state) {
    switch (state) {
        case slot_state::free: return "free";
        case slot_state::queued: return "queued";
        case slot_state::prefill: return "prefill";
        case slot_state::decode: return "decode";
        case slot_state::done: return "done";
        case slot_state::cancelled: return "cancelled";
    }
    return "free";
}

struct scheduled_slot {
    uint32_t index = 0;
    int32_t seq = 0;
    slot_state state = slot_state::free;
    std::vector<llama_token> prompt;
    size_t prefill_offset = 0;
    uint32_t n_predict = 0;
    uint32_t n_decoded = 0;
    uint32_t next_position = 0;
    llama_token last = 0;
    potluck::slot_config sampling;
    bool configured = false;
    bool ever_used = false;
    bool needs_clear = false;
    bool stream = false;
    bool chat = false;
    std::string id;
    uint64_t created = 0;
    std::deque<std::string> pieces;
    std::vector<llama_token> generated;
    std::string error;
    bool finished = false;
    bool cancelled = false;
    bool release_when_finished = false;
    bool callback_done = false;
    std::mutex mutex;
    std::condition_variable cv;
};

struct batch_item {
    std::shared_ptr<scheduled_slot> slot;
    int32_t position = 0;
    int32_t token = 0;
};

class slot_scheduler {
public:
    slot_scheduler(ServerRing & ring, const llama_vocab * vocab, uint32_t n_slots,
                   uint32_t prefill_batch, std::function<bool(std::string &)> rebuild = {})
        : ring_(ring), vocab_(vocab),
          prefill_batch_(std::max<uint32_t>(1, prefill_batch)),
          rebuild_(std::move(rebuild)) {
        slots_.reserve(std::max<uint32_t>(1, n_slots));
        for (uint32_t i = 0; i < std::max<uint32_t>(1, n_slots); ++i) {
            auto slot = std::make_shared<scheduled_slot>();
            slot->index = i;
            slot->seq = static_cast<int32_t>(i);
            slots_.push_back(std::move(slot));
        }
    }

    void start() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!thread_.joinable()) {
            thread_ = std::thread([this] { run(); });
        }
    }

    void request_stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            for (const auto & slot : slots_) {
                std::lock_guard<std::mutex> slot_lock(slot->mutex);
                slot->cancelled = true;
                slot->state = slot_state::cancelled;
                slot->finished = true;
                if (slot->error.empty()) {
                    slot->error = "server stopped";
                }
                slot->cv.notify_all();
            }
        }
        work_cv_.notify_all();
    }

    void stop() {
        request_stop();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    ~slot_scheduler() {
        stop();
    }

    std::shared_ptr<scheduled_slot> acquire(const std::vector<llama_token> & prompt,
                                            uint32_t n_predict,
                                            const potluck::slot_config & sampling,
                                            bool stream, bool chat,
                                            const std::string & id, uint64_t created) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (stopping_ || rebuilding_) {
            return {};
        }
        if (!work_cv_.wait_for(lock, std::chrono::seconds(30), [&] {
                if (stopping_ || rebuilding_) {
                    return true;
                }
                for (const auto & slot : slots_) {
                    std::lock_guard<std::mutex> slot_lock(slot->mutex);
                    if (slot->state == slot_state::free) {
                        return true;
                    }
                }
                return false;
            })) {
            return {};
        }
        if (stopping_ || rebuilding_) {
            return {};
        }
        for (const auto & slot : slots_) {
            std::lock_guard<std::mutex> slot_lock(slot->mutex);
            if (slot->state != slot_state::free) {
                continue;
            }
            slot->prompt = prompt;
            slot->prefill_offset = 0;
            slot->n_predict = n_predict;
            slot->n_decoded = 0;

            slot->next_position = 0;
            slot->last = 0;
            slot->sampling = sampling;
            slot->sampling.seq = slot->seq;
            slot->configured = false;
            slot->needs_clear = slot->ever_used;
            slot->stream = stream;
            slot->chat = chat;
            slot->id = id;
            slot->created = created;
            slot->pieces.clear();
            slot->error.clear();
            slot->finished = false;
            slot->cancelled = false;
            slot->release_when_finished = false;
            slot->callback_done = false;
            slot->state = slot_state::queued;
            work_cv_.notify_all();
            return slot;
        }
        return {};
    }
    bool rebuilding() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return rebuilding_;
    }


    void wait_done(const std::shared_ptr<scheduled_slot> & slot) {
        std::unique_lock<std::mutex> lock(slot->mutex);
        slot->cv.wait(lock, [&] { return slot->finished; });
    }
    void wait_first(const std::shared_ptr<scheduled_slot> & slot) {
        std::unique_lock<std::mutex> lock(slot->mutex);
        slot->cv.wait(lock, [&] {
            return slot->cancelled || !slot->pieces.empty() || slot->finished;
        });
    }

    bool take_piece(const std::shared_ptr<scheduled_slot> & slot, std::string & piece) {
        std::unique_lock<std::mutex> lock(slot->mutex);
        slot->cv.wait(lock, [&] {
            return slot->cancelled || !slot->pieces.empty() || slot->finished;
        });
        if (!slot->pieces.empty()) {
            piece = std::move(slot->pieces.front());
            slot->pieces.pop_front();
            return true;
        }
        return false;
    }
    void cancel(const std::shared_ptr<scheduled_slot> & slot) {
        std::lock_guard<std::mutex> lock(slot->mutex);
        slot->cancelled = true;
        slot->release_when_finished = true;
        slot->state = slot_state::cancelled;
        slot->cv.notify_all();
        work_cv_.notify_all();
    }
    void acknowledge_cancel(const std::shared_ptr<scheduled_slot> & slot) {
        std::lock_guard<std::mutex> lock(slot->mutex);
        slot->callback_done = true;
        slot->cv.notify_all();
        work_cv_.notify_all();
    }
    void release(const std::shared_ptr<scheduled_slot> & slot) {
        {
            std::lock_guard<std::mutex> lock(slot->mutex);
            slot->prompt.clear();
            slot->pieces.clear();
            slot->generated.clear();
            slot->n_decoded = 0;
            slot->cancelled = false;
            slot->release_when_finished = false;
            slot->callback_done = false;
            slot->state = slot_state::free;
        }
        work_cv_.notify_all();
    }

    json health() const {
        json output = json::array();
        for (const auto & slot : slots_) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            output.push_back(json{
                { "index", slot->index },
                { "state", slot_state_name(slot->state) },
                { "n_decoded", slot->n_decoded },
            });
        }
        return output;
    }

    bool is_stopping() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopping_;
    }

private:
    void reap_cancelled() {
        std::vector<std::shared_ptr<scheduled_slot>> ready;
        for (const auto & slot : slots_) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (slot->release_when_finished && slot->callback_done && slot->finished) {
                slot->release_when_finished = false;
                ready.push_back(slot);
            }
        }
        for (const auto & slot : ready) {
            release(slot);
        }
    }
    void finish(const std::shared_ptr<scheduled_slot> & slot, const std::string & error = {}) {
        std::lock_guard<std::mutex> lock(slot->mutex);
        if (!error.empty() && !slot->cancelled) {
            slot->error = error;
        }
        slot->finished = true;
        if (!slot->cancelled) {
            slot->state = slot_state::done;
        }
        slot->cv.notify_all();
    }

    void emit(const std::shared_ptr<scheduled_slot> & slot, llama_token token,
              uint32_t position) {
        std::lock_guard<std::mutex> lock(slot->mutex);
        if (slot->cancelled) {
            return;
        }
        if (!llama_vocab_is_eog(vocab_, token)) {
            slot->generated.push_back(token);
            slot->pieces.push_back(token_piece(vocab_, token));
            ++slot->n_decoded;
        }
        slot->last = token;
        slot->next_position = position + 1;
        if (llama_vocab_is_eog(vocab_, token) || slot->n_decoded >= slot->n_predict) {
            slot->finished = true;
            slot->state = slot_state::done;
        } else {
            slot->state = slot_state::decode;
        }
        slot->cv.notify_all();
    }


    void run_round(const std::vector<std::shared_ptr<scheduled_slot>> & active) {
        std::shared_ptr<scheduled_slot> clear_slot;
        std::vector<std::shared_ptr<scheduled_slot>> cancelled;
        for (const auto & slot : active) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (slot->state == slot_state::cancelled) {
                cancelled.push_back(slot);
                if ((slot->ever_used || slot->needs_clear) && !clear_slot) {
                    clear_slot = slot;
                }
            }
        }
        if (!clear_slot) {
            for (const auto & slot : active) {
                std::lock_guard<std::mutex> lock(slot->mutex);
                if (slot->state != slot_state::cancelled && slot->needs_clear) {
                    clear_slot = slot;
                    break;
                }
            }
        }

        std::vector<std::shared_ptr<scheduled_slot>> selected;
        for (const auto & slot : active) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (slot->state == slot_state::free || slot->state == slot_state::done ||
                slot->state == slot_state::cancelled) {
                continue;
            }
            
            selected.push_back(slot);
        }
        if (selected.empty()) {
            if (clear_slot) {
                const std::vector<int32_t> empty;
                try {
                    (void) drive_batch(ring_, empty, empty, empty, clear_slot->seq,
                                       -1, -1, 0, nullptr,
                                       [this] { return is_stopping(); });
                } catch (const request_cancelled &) {
                    return;
                }
                std::lock_guard<std::mutex> lock(clear_slot->mutex);
                clear_slot->needs_clear = false;
                clear_slot->ever_used = false;
            }
            for (const auto & slot : cancelled) {
                std::lock_guard<std::mutex> lock(slot->mutex);
                if (!slot->ever_used && !slot->needs_clear) {
                    slot->finished = true;
                    slot->cv.notify_all();
                }
            }
            return;
        }

        std::vector<std::pair<std::shared_ptr<scheduled_slot>, potluck::slot_config>> configurations;
        for (const auto & slot : selected) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (!slot->configured) {
                configurations.emplace_back(slot, slot->sampling);
            }
        }
        for (const auto & configuration : configurations) {
            configure_slot(ring_, configuration.second);
            std::lock_guard<std::mutex> lock(configuration.first->mutex);
            configuration.first->configured = true;
        }

        std::vector<batch_item> prefill_body;
        std::vector<batch_item> prefill_logits;
        std::vector<batch_item> decode;
        std::vector<std::pair<std::shared_ptr<scheduled_slot>, size_t>> prefill_offsets;
        size_t prefill_count = 0;
        for (const auto & slot : selected) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (slot->state != slot_state::prefill) {
                continue;
            }
            const size_t remaining = slot->prompt.size() - slot->prefill_offset;
            const size_t count = std::min(remaining, prefill_batch_ - prefill_count);
            if (count != 0) {
                prefill_offsets.emplace_back(slot, slot->prefill_offset);
            }
            for (size_t i = 0; i < count; ++i) {
                batch_item item;
                item.slot = slot;
                item.position = static_cast<int32_t>(slot->prefill_offset + i);
                item.token = static_cast<int32_t>(slot->prompt[slot->prefill_offset + i]);
                if (i + 1 == count && slot->prefill_offset + count == slot->prompt.size()) {
                    prefill_logits.push_back(item);
                } else {
                    prefill_body.push_back(item);
                }
            }
            slot->prefill_offset += count;
            prefill_count += count;
            if (prefill_count == prefill_batch_) {
                break;
            }
        }
        const size_t available = prefill_batch_ - prefill_count;
        for (const auto & slot : selected) {
            if (decode.size() == available) {
                break;
            }
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (slot->state != slot_state::decode) {
                continue;
            }
            decode.push_back({
                slot, static_cast<int32_t>(slot->next_position),
                static_cast<int32_t>(slot->last)
            });
        }

        std::vector<batch_item> items;
        items.reserve(prefill_body.size() + decode.size() + prefill_logits.size());
        items.insert(items.end(), prefill_body.begin(), prefill_body.end());
        items.insert(items.end(), decode.begin(), decode.end());
        items.insert(items.end(), prefill_logits.begin(), prefill_logits.end());
        if (items.empty()) {
            return;
        }
        std::vector<int32_t> positions;
        std::vector<int32_t> sequences;
        std::vector<int32_t> tokens;
        positions.reserve(items.size());
        sequences.reserve(items.size());
        tokens.reserve(items.size());
        for (const batch_item & item : items) {
            positions.push_back(item.position);
            sequences.push_back(item.slot->seq);
            tokens.push_back(item.token);
        }
        const uint32_t n_logits = static_cast<uint32_t>(decode.size() + prefill_logits.size());
        const int32_t clear_seq = clear_slot ? clear_slot->seq : -1;
        const auto batch_cancelled = [&]() {
            for (const auto & slot : selected) {
                std::lock_guard<std::mutex> lock(slot->mutex);
                if (slot->cancelled) {
                    return true;
                }
            }
            return false;
        };
        bool batch_started = false;
        const auto fail_selected = [&]() {
            for (const auto & slot : selected) {
                std::lock_guard<std::mutex> lock(slot->mutex);
                slot->cancelled = true;
                slot->state = slot_state::cancelled;
                slot->finished = true;
                slot->error = "request cancelled";
                slot->ever_used = true;
                slot->needs_clear = true;
                slot->cv.notify_all();
            }
            work_cv_.notify_all();
        };
        std::vector<int32_t> result;
        try {
            result = drive_batch(
                ring_, positions, sequences, tokens, clear_seq, -1, -1, n_logits,
                nullptr, batch_cancelled, &batch_started);
        } catch (const request_cancelled &) {
            if (batch_started) {
                fail_selected();
            } else {
                for (const auto & offset : prefill_offsets) {
                    std::lock_guard<std::mutex> lock(offset.first->mutex);
                    offset.first->prefill_offset = offset.second;
                }
                for (const auto & slot : selected) {
                    std::lock_guard<std::mutex> lock(slot->mutex);
                    if (slot->cancelled) {
                        slot->finished = true;
                        slot->error = "request cancelled";
                        slot->cv.notify_all();
                    }
                }
            }
            return;
        } catch (...) {
            for (const auto & slot : selected) {
                std::lock_guard<std::mutex> lock(slot->mutex);
                if (batch_started) {
                    slot->ever_used = true;
                }
                slot->needs_clear = true;
            }
            throw;
        }
        if (batch_cancelled()) {
            fail_selected();
            return;
        }
        if (clear_slot) {
            std::lock_guard<std::mutex> lock(clear_slot->mutex);
            clear_slot->needs_clear = false;
            if (clear_slot->cancelled) {
                clear_slot->ever_used = false;
            }
        }
        for (const auto & slot : selected) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            slot->ever_used = true;
            if (slot->cancelled) {
                slot->needs_clear = true;
            }
        }
        const size_t result_start = items.size() - n_logits;
        for (size_t i = 0; i < decode.size(); ++i) {
            emit(decode[i].slot, static_cast<llama_token>(result[result_start + i]),
                 static_cast<uint32_t>(decode[i].position));
        }
        for (size_t i = 0; i < prefill_logits.size(); ++i) {
            emit(prefill_logits[i].slot,
                 static_cast<llama_token>(result[result_start + decode.size() + i]),
                 static_cast<uint32_t>(prefill_logits[i].position));
        }
        for (const auto & slot : cancelled) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (!slot->ever_used && !slot->needs_clear) {
                slot->finished = true;
                slot->cv.notify_all();
            }
        }
        work_cv_.notify_all();
    }

    void attempt_rebuild() {
        std::string detail;
        bool rebuilt = false;
        try {
            rebuilt = rebuild_ && rebuild_(detail);
        } catch (const std::exception & exception) {
            detail = exception.what();
        }
        std::printf("potluck-server: ring rebuild %s: %s\n",
                    rebuilt ? "succeeded" : "failed",
                    detail.empty() ? (rebuilt ? "ring is ready" : "no detail") : detail.c_str());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            rebuilding_ = !rebuilt;
        }
        work_cv_.notify_all();
    }

    void run() {
        for (;;) {
            std::vector<std::shared_ptr<scheduled_slot>> active;
            bool retry_rebuild = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_cv_.wait_for(lock, std::chrono::seconds(1), [&] {
                    if (stopping_) {
                        return true;
                    }
                    if (rebuilding_ && rebuild_ &&
                        std::chrono::steady_clock::now() - last_rebuild_ >=
                            std::chrono::seconds(30)) {
                        return true;
                    }
                    for (const auto & slot : slots_) {
                        std::lock_guard<std::mutex> slot_lock(slot->mutex);
                        if (slot->state == slot_state::queued ||
                            slot->state == slot_state::prefill ||
                            slot->state == slot_state::decode ||
                            slot->state == slot_state::cancelled) {
                            return true;
                        }
                    }
                    return false;
                });
                if (stopping_) {
                    return;
                }
                if (rebuilding_ && rebuild_ &&
                    std::chrono::steady_clock::now() - last_rebuild_ >=
                        std::chrono::seconds(30)) {
                    last_rebuild_ = std::chrono::steady_clock::now();
                    retry_rebuild = true;
                } else {
                    for (const auto & slot : slots_) {
                        std::lock_guard<std::mutex> slot_lock(slot->mutex);
                        if (slot->state == slot_state::queued) {
                            if (slot->n_predict == 0) {
                                slot->finished = true;
                                slot->state = slot_state::done;
                                slot->cv.notify_all();
                            } else {
                                slot->state = slot_state::prefill;
                                active.push_back(slot);
                            }
                        } else if (slot->state == slot_state::prefill ||
                                   slot->state == slot_state::decode ||
                                   slot->state == slot_state::cancelled) {
                            active.push_back(slot);
                        }
                    }
                }
            }
            if (retry_rebuild) {
                attempt_rebuild();
                continue;
            }
            if (active.empty()) {
                work_cv_.notify_all();
                continue;
            }
            try {
                run_round(active);
            } catch (const std::exception &) {
                if (is_stopping()) {
                    return;
                }
                bool start_rebuild = false;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    const auto now = std::chrono::steady_clock::now();
                    if (!rebuilding_) {
                        rebuilding_ = true;
                        if (last_rebuild_.time_since_epoch().count() == 0 ||
                            now - last_rebuild_ >= std::chrono::seconds(30)) {
                            last_rebuild_ = now;
                            start_rebuild = true;
                        }
                    }
                }
                for (const auto & slot : active) {
                    finish(slot, "cluster is rebuilding; retry");
                }
                reap_cancelled();
                if (start_rebuild) {
                    work_cv_.notify_all();
                    attempt_rebuild();
                }
            }
            reap_cancelled();
        }
    }

    ServerRing & ring_;
    const llama_vocab * vocab_;
    uint32_t prefill_batch_;
    std::function<bool(std::string &)> rebuild_;
    std::chrono::steady_clock::time_point last_rebuild_{};
    bool rebuilding_ = false;
    mutable std::mutex mutex_;
    std::condition_variable work_cv_;
    std::vector<std::shared_ptr<scheduled_slot>> slots_;
    std::thread thread_;
    bool stopping_ = false;
};


std::atomic<uint64_t> request_counter{0};

std::string request_id(uint64_t & created) {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    created = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(now).count());
    const uint64_t sequence = request_counter.fetch_add(1, std::memory_order_relaxed);
    return "chatcmpl-potluck-" + std::to_string(created) + "-" + std::to_string(sequence);
}

json error_json(const std::string & message) {
    return json{ { "error", message } };
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string model_path;
        std::string host = "127.0.0.1";
        std::string hosts_spec;
        std::string launch;
        std::string head_share = "auto";
        uint16_t http_port = 8080;
        uint32_t worker_local = 0;
        uint32_t n_predict_default = 24;
        uint32_t n_ctx = 4096;
        uint32_t n_seq_max = 4;
        uint32_t n_ubatch = 512;
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
            else if (arg == "--host") host = take("--host");
            else if (arg == "--port") http_port = static_cast<uint16_t>(std::stoul(take("--port")));
            else if (arg == "--workers") {
                worker_local = static_cast<uint32_t>(std::stoul(take("--workers")));
                workers_option = true;
            }
            else if (arg == "--hosts") hosts_spec = take("--hosts");
            else if (arg == "--launch") launch = take("--launch");
            else if (arg == "--head-share") head_share = take("--head-share");
            else if (arg == "--ctx") n_ctx = static_cast<uint32_t>(std::stoul(take("--ctx")));
            else if (arg == "--slots" || arg == "--batch") {
                n_seq_max = std::max<uint32_t>(1, static_cast<uint32_t>(std::stoul(take(arg.c_str()))));
            }
            else if (arg == "--ubatch") {
                n_ubatch = std::max<uint32_t>(1, static_cast<uint32_t>(std::stoul(take("--ubatch"))));
            }
            else if (arg == "--temp") temp = std::stof(take("--temp"));
            else if (arg == "--top-p") top_p = std::stof(take("--top-p"));
            else if (arg == "--seed") seed = static_cast<uint32_t>(std::stoul(take("--seed")));
            else if (arg == "--n-predict") n_predict_default = static_cast<uint32_t>(std::stoul(take("--n-predict")));
            else if (arg == "--bench") bench = true;
            else throw std::runtime_error(
                "usage: potluck-server -m model.gguf [--workers N] [--slots N] [--ubatch N] "
                "[--head-share auto|off] [--hosts a,b,c --launch ssh]");
        }
        if (model_path.empty()) {
            throw std::runtime_error("need -m model.gguf");
        }
        if (head_share != "auto" && head_share != "off") {
            throw std::runtime_error("--head-share must be auto or off");
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
        if (n_ctx == 0) {
            n_ctx = 4096;
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
                validate_ssh_target(value);
                bootstrap.ssh_target = value;
                bootstrap.ring_host = ring_host(value);
                bootstrap_nodes.push_back(std::move(bootstrap));
            }
        }

        ring_session session;
        model_digest_cache digest_cache;
        auto bring_up_ring = [&](ring_session & target, std::string & error) -> bool {
            ServerRing ring;
            std::vector<planned_worker> planned;
            try {
                if (!target.ring.controls.empty()) {
                    std::string reset_error;
                    if (!reset_ring_workers(target.ring, reset_error) && !reset_error.empty()) {
                        std::fprintf(stderr, "potluck-server: old ring reset: %s\n",
                                     reset_error.c_str());
                    }
                }
                if (!target.workers.empty()) {
                    stop_planned_workers(target.workers);
                    target.workers.clear();
                }
                if (hosts_spec.empty() && !workers_option) {
                    bootstrap_nodes = discover_bootstrap_nodes();
                }
                const bool has_remote = !bootstrap_nodes.empty();
        if (!has_remote && worker_local == 0) {
            throw std::runtime_error("need at least one ring worker");
        }
        const std::string worker_path = exe_dir(argv[0]) + "/potluck-worker";
        if (worker_path == "/potluck-worker") {
            throw std::runtime_error("cannot locate potluck-worker beside potluck-server");
        }
        const std::filesystem::path adjacent_root =
            std::filesystem::path(worker_path).parent_path().parent_path().parent_path();
        const std::filesystem::path cwd_stage_dir =
            std::filesystem::current_path() / "dist/mac-arm64";
        const std::filesystem::path stage_dir =
            std::filesystem::is_regular_file(cwd_stage_dir / "potluck-build-id")
                ? cwd_stage_dir : adjacent_root / "dist/mac-arm64";
        const bool has_staged_payload =
            std::filesystem::is_regular_file(stage_dir / "potluck-build-id");
        const std::string local_platform = first_command_line("uname -sm 2>/dev/null");
        if (has_remote && has_staged_payload) {
            for (const bootstrap_node & bootstrap : bootstrap_nodes) {
                (void) refresh_remote_binaries(bootstrap, stage_dir, local_platform);
            }
        }

        const uint64_t model_bytes = std::filesystem::file_size(model_path);
        const uint32_t n_head_kv = llama_model_n_head_kv(meta);
        const uint64_t layer_cost = route_layer_cost(
            n_layer, model_bytes, n_head_kv, head_dim, n_ctx);
        std::vector<device_probe> candidates;
        device_probe head;
        bool head_participates = false;
        uint64_t head_budget = 0;
        uint64_t head_reserve = 4ull * 1024ull * 1024ull * 1024ull;
        if (has_remote) {
            candidates = probe_remote_candidates(bootstrap_nodes);
            if (has_staged_payload) {
                for (device_probe & candidate : candidates) {
                    if (!candidate.ok &&
                        refresh_remote_binaries(candidate.bootstrap, stage_dir, local_platform)) {
                        device_probe retry = probe_remote_worker(candidate.bootstrap);
                        if (retry.ok) {
                            candidate = std::move(retry);
                        }
                    }
                }
            }
            for (const device_probe & probe : candidates) {
                if (!probe.ok) {
                    std::fprintf(stderr, "potluck-server: excluding %s: %s\n",
                                 probe.bootstrap.ssh_target.c_str(),
                                 probe.error.empty() ? "probe failed" : probe.error.c_str());
                }
            }
            candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                            [](const device_probe & probe) {
                                                return !probe.ok;
                                            }),
                             candidates.end());
            if (candidates.empty()) {
                throw std::runtime_error("no admissible Potluck devices");
            }

            if (head_share == "auto") {
                head = probe_local_worker(worker_path);
                if (!head.ok) {
                    std::fprintf(stderr, "potluck-server: head probe failed: %s\n",
                                 head.error.c_str());
                } else {
                    head.host = "head";
                    head_reserve = std::max(head_reserve, head.profile.host_total_bytes / 4);
                    head_budget = head.profile.host_free_bytes > head_reserve
                        ? head.profile.host_free_bytes - head_reserve : 0;
                }
            }
        }
        else {
            device_probe physical = probe_local_worker(worker_path);
            physical.host = "127.0.0.1";
            if (!physical.ok) {
                throw std::runtime_error("local worker probe failed: " + physical.error);
            }
            const uint64_t physical_usable = physical.usable_bytes();
            const uint64_t max_local_peers = std::min<uint64_t>(
                { static_cast<uint64_t>(worker_local), static_cast<uint64_t>(n_layer),
                  physical_usable / layer_cost });
            const uint32_t local_count = static_cast<uint32_t>(
                std::max<uint64_t>(1, max_local_peers));
            const uint64_t shared_bytes = physical_usable / local_count;
            const uint64_t remainder = physical_usable % local_count;
            for (uint32_t index = 0; index < local_count; ++index) {
                device_probe probe = physical;
                probe.placement_usable_limit = shared_bytes +
                    (index < remainder ? 1 : 0);
                candidates.push_back(std::move(probe));
            }
        }
        const bool allow_remote_shortfall =
            has_remote && head_share == "auto" && head.ok;
        std::vector<device_probe> devices = admit_devices(
            std::move(candidates), n_layer, model_bytes, n_head_kv, head_dim, n_ctx,
            allow_remote_shortfall);
        if (has_remote && head_share == "auto" && head.ok) {
            head.placement_usable_limit = head_budget;
            head_participates =
                head_budget >= 2ull * 1024ull * 1024ull * 1024ull &&
                head.usable_bytes() >= layer_cost;
            if (head_participates) {
                head.bootstrap = {};
                head.ok = true;
                devices.push_back(std::move(head));
            }
        }
        std::printf("potluck-server: head participation %s (budget %llu MiB, reserve %llu MiB)\n",
                    head_participates ? "on" : "off",
                    static_cast<unsigned long long>(head_budget / (1024ull * 1024ull)),
                    static_cast<unsigned long long>(head_reserve / (1024ull * 1024ull)));
        if (devices.empty()) {
            throw std::runtime_error("no admissible Potluck devices");
        }

        if (has_remote && std::none_of(devices.begin(), devices.end(),
                                       [](const device_probe & device) {
                                           return !device.bootstrap.ssh_target.empty();
                                       })) {
            throw std::runtime_error("no remote device remains after admission");
        }
        const uint32_t n_workers = static_cast<uint32_t>(devices.size());
        if (n_workers == 0) {
            throw std::runtime_error("need at least one ring worker");
        }
        std::string head_address;
        if (has_remote) {
            head_address = local_address_for_peer(devices.front().bootstrap.ring_host,
                                                  devices.front().bootstrap.ring_port);
        }
        planned.reserve(n_workers);
        for (uint32_t index = 0; index < n_workers; ++index) {
            const device_probe & device = devices[index];
            planned_worker plan;
            plan.device = device;
            const bool is_remote = has_remote && !device.bootstrap.ssh_target.empty();
            plan.kind = is_remote ? worker_kind::remote : worker_kind::local;
            plan.model = is_remote ? model_name : model_path;
            const std::string address_host = is_remote
                ? ipv4_address_for_host(device.bootstrap.ring_host)
                : (has_remote ? head_address : "127.0.0.1");
            const uint16_t port = is_remote ? device.bootstrap.ring_port : free_port();
            plan.ring.address = { address_host, port };
            plan.ring.connect_endpoint = tcp_endpoint(address_host, port);
            plan.ring.bind_endpoint = tcp_endpoint(
                is_remote || has_remote ? "0.0.0.0" : address_host, port);
            planned.push_back(std::move(plan));
        }
        for (uint32_t index = 0; index < n_workers; ++index) {
            const device_probe & device = devices[index];
            const bool is_remote = has_remote && !device.bootstrap.ssh_target.empty();
            planned[index].ring.next_endpoint =
                n_workers == 1 && is_remote
                    ? tcp_endpoint("127.0.0.1", planned[index].ring.address.port)
                    : planned[(index + 1) % n_workers].ring.connect_endpoint;
        }

        ring.workers.reserve(planned.size());
        for (const planned_worker & plan : planned) {
            ring.workers.push_back(plan.ring);
        }
        ring.windows = build_ring_route(devices, n_layer, model_bytes, n_head_kv,
                                        head_dim, n_ctx);
        std::vector<uint32_t> device_share(devices.size(), 0);
        for (const potluck::ring_window & window : ring.windows) {
            device_share[window.owner] += window.end - window.start;
        }
        for (size_t index = 0; index < ring.windows.size(); ++index) {
            const potluck::ring_window & window = ring.windows[index];
            std::printf("potluck-server: window %zu owner=%u host=%s layers=[%u,%u) share=%u usable=%llu MiB\n",
                        index, window.owner, ring.workers[window.owner].address.host.c_str(),
                        window.start, window.end, device_share[window.owner],
                        static_cast<unsigned long long>(
                            devices[window.owner].usable_bytes() / (1024ull * 1024ull)));
        }
        std::fflush(stdout);
        std::string result_error;
        ring.result = potluck::ring_receiver::bind("tcp://0.0.0.0:*", result_error);
        if (!ring.result.valid()) {
            throw std::runtime_error("cannot bind ring result receiver: " + result_error);
        }
        const std::string result_host = has_remote ? head_address
            : (host == "0.0.0.0" ? "127.0.0.1" : host);
        ring.result_endpoint = endpoint_host(ring.result.endpoint(), result_host);

        std::string model_digest;
        if (has_remote) {
            model_digest = pinned_model_digest(model_path, adjacent_root);
            if (model_digest.empty()) {
                model_digest = digest_cache.get(model_path);
            }
        }
        for (const planned_worker & plan : planned) {
            if (plan.kind != worker_kind::remote) {
                continue;
            }
            if (!stop_remote_workers(plan.device.bootstrap)) {
                throw std::runtime_error("remote worker cleanup failed for " +
                                         plan.device.bootstrap.ssh_target);
            }
            if (has_staged_payload &&
                !refresh_remote_binaries(plan.device.bootstrap, stage_dir, local_platform)) {
                throw std::runtime_error("worker binary refresh failed for " +
                                         plan.device.bootstrap.ssh_target);
            }
            std::string distribution_error;
            if (!ensure_remote_model(plan.device.bootstrap, model_path, model_digest,
                                     distribution_error)) {
                throw std::runtime_error("model distribution failed for " +
                                         plan.device.bootstrap.ssh_target + ": " +
                                         distribution_error);
            }
        }
        for (uint32_t index = 0; index < n_workers; ++index) {
            planned_worker & plan = planned[index];
            if (plan.kind == worker_kind::remote) {
                plan.remote_ssh_pid = launch_remote_worker(plan.device.bootstrap, plan.model,
                                                           ring.workers[index],
                                                           ring.result_endpoint, index);
            } else {
                plan.local_pid = launch_local_worker(worker_path, plan.model,
                                                     ring.workers[index],
                                                     ring.result_endpoint, index);
            }
        }
        const std::vector<potluck::accel_profile> profiles =
            collect_accel_profiles(ring, n_workers);
        assign_gpu_layers(ring.windows, profiles, n_workers, model_bytes, n_layer,
                          n_head_kv, head_dim, n_ctx);
        std::fflush(stdout);
        configure_ring(ring, n_layer, n_ctx, n_seq_max, n_ubatch, seed, temp, top_p);
        target.ring = std::move(ring);
        target.workers = std::move(planned);
        target.healthy = true;
        error.clear();
        return true;
            } catch (const std::exception & exception) {
                if (!ring.controls.empty()) {
                    std::string reset_error;
                    if (!reset_ring_workers(ring, reset_error) && !reset_error.empty()) {
                        std::fprintf(stderr, "potluck-server: failed-ring reset: %s\n",
                                     reset_error.c_str());
                    }
                }
                stop_planned_workers(planned);
                target.healthy = false;
                error = exception.what();
                return false;
            }
        };
        std::string startup_error;
        if (!bring_up_ring(session, startup_error)) {
            throw std::runtime_error(startup_error);
        }
        ServerRing & ring = session.ring;

        auto rebuild_ring = [&](std::string & error) {
            std::vector<planned_worker> old_workers;
            {
                std::lock_guard<std::mutex> lock(session.mutex);
                session.healthy = false;
            }
            if (!session.ring.controls.empty()) {
                std::string reset_error;
                if (!reset_ring_workers(session.ring, reset_error) &&
                    !reset_error.empty()) {
                    std::fprintf(stderr, "potluck-server: old ring reset: %s\n",
                                 reset_error.c_str());
                }
            }
            {
                std::lock_guard<std::mutex> lock(session.mutex);
                old_workers = std::move(session.workers);
            }
            stop_planned_workers(old_workers);
            ring_session replacement;
            if (!bring_up_ring(replacement, error)) {
                return false;
            }
            {
                std::lock_guard<std::mutex> lock(session.mutex);
                session.ring = std::move(replacement.ring);
                session.workers = std::move(replacement.workers);
                session.healthy = true;
            }
            return true;
        };
        httplib::Server server;
        slot_scheduler scheduler(ring, vocab, n_seq_max, n_ubatch, rebuild_ring);
        scheduler.start();
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
            std::lock_guard<std::mutex> ring_lock(session.mutex);
            json health = { { "status", (scheduler.rebuilding() || !session.healthy)
                                        ? "rebuilding" : "ok" },
                            { "workers", ring.workers.size() },
                            { "windows", json::array() }, { "slots", scheduler.health() } };
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
            std::shared_ptr<scheduled_slot> slot;
            try {
                json req;
                try {
                    req = json::parse(request.body);
                } catch (const std::exception & e) {
                    throw std::runtime_error(std::string("invalid JSON: ") + e.what());
                }
                if (!req.is_object()) {
                    throw std::runtime_error("invalid JSON: request must be an object");
                }
                static const std::unordered_set<std::string> allowed = {
                    "messages", "prompt", "max_tokens", "n_predict", "stream",
                    "reasoning_effort", "temperature", "top_p", "top_k", "seed"
                };
                for (auto it = req.begin(); it != req.end(); ++it) {
                    if (allowed.find(it.key()) == allowed.end()) {
                        throw std::runtime_error("unsupported field: " + it.key());
                    }
                }
                if (req.contains("reasoning_effort") &&
                    !req["reasoning_effort"].is_string()) {
                    throw std::runtime_error(
                        "invalid reasoning_effort: expected a string");
                }

                std::string prompt_text;
                uint32_t n_predict = n_predict_default;
                if (chat) {
                    if (!req.contains("messages") || !req["messages"].is_array() ||
                        req["messages"].empty()) {
                        throw std::runtime_error("missing messages");
                    }
                    common_chat_templates_inputs inputs;
                    try {
                        inputs.messages = common_chat_msgs_parse_oaicompat(req["messages"]);
                    } catch (const std::exception & e) {
                        throw std::runtime_error(std::string("invalid messages: ") + e.what());
                    }
                    if (req.contains("reasoning_effort")) {
                        const std::string effort =
                            req["reasoning_effort"].get<std::string>();
                        if (effort == "none") {
                            inputs.enable_thinking = false;
                            inputs.chat_template_kwargs["enable_thinking"] = "false";
                        } else if (!effort.empty()) {
                            inputs.chat_template_kwargs["reasoning_effort"] =
                                json(effort).dump();
                        }
                    }
                    if (!chat_templates) {
                        throw std::runtime_error("model has no chat template");
                    }
                    prompt_text = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
                    n_predict = json_u32(req, "max_tokens", n_predict_default);
                } else {
                    if (!req.contains("prompt") || !req["prompt"].is_string() ||
                        req["prompt"].get<std::string>().empty()) {
                        throw std::runtime_error("missing prompt");
                    }
                    prompt_text = req["prompt"].get<std::string>();
                    n_predict = json_u32(req, "n_predict", n_predict_default);
                }
                const std::vector<llama_token> prompt = tokenize_prompt(vocab, prompt_text);
                const bool stream = json_bool(req, "stream", false);
                potluck::slot_config sampling;
                sampling.temp = temp;
                sampling.top_p = top_p;
                sampling.seed = seed;
                if (req.contains("temperature")) {
                    if (!req["temperature"].is_number()) {
                        throw std::runtime_error("invalid temperature: expected a number");
                    }
                    sampling.temp = req["temperature"].get<float>();
                }
                if (!std::isfinite(sampling.temp) || sampling.temp < 0.0f) {
                    throw std::runtime_error("invalid temperature: expected a finite non-negative number");
                }
                if (req.contains("top_p")) {
                    if (!req["top_p"].is_number()) {
                        throw std::runtime_error("invalid top_p: expected a number");
                    }
                    sampling.top_p = req["top_p"].get<float>();
                }
                if (!std::isfinite(sampling.top_p) ||
                    sampling.top_p < 0.0f || sampling.top_p > 1.0f) {
                    throw std::runtime_error("invalid top_p: expected a number from 0 to 1");
                }
                if (req.contains("top_k")) {
                    sampling.top_k = json_u32(req, "top_k", 0);
                    if (sampling.top_k > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
                        throw std::runtime_error("invalid top_k: out of range");
                    }
                }
                if (req.contains("seed")) {
                    sampling.seed = json_u32(req, "seed", 0);
                }
                uint64_t created = 0;
                const std::string id = request_id(created);
                slot = scheduler.acquire(prompt, n_predict, sampling, stream, chat,
                                          id, created);
                if (!slot) {
                    response.status = 503;
                    set_common_headers(response);
                    const char * message = scheduler.rebuilding()
                        ? "cluster is rebuilding; retry" : "all conversation slots are busy";
                    response.set_content(error_json(message).dump(), "application/json");
                    return;
                }
                const auto common_chunk = [id, created, model_name](const json & choice) {
                    return json{
                        { "id", id }, { "object", "chat.completion.chunk" },
                        { "created", created }, { "model", model_name },
                        { "choices", json::array({ choice }) }
                    }.dump();
                };

                if (stream) {
                    scheduler.wait_first(slot);
                    bool initial_cancelled = false;
                    std::string initial_error;
                    {
                        std::lock_guard<std::mutex> lock(slot->mutex);
                        initial_cancelled = slot->cancelled;
                        initial_error = slot->error;
                    }
                    if (initial_cancelled) {
                        scheduler.acknowledge_cancel(slot);
                        scheduler.release(slot);
                        response.status = 503;
                        set_common_headers(response);
                        response.set_content(error_json("request cancelled").dump(),
                                             "application/json");
                        return;
                    }
                    if (!initial_error.empty()) {
                        throw std::runtime_error(initial_error);
                    }
                    response.status = 200;
                    set_common_headers(response);
                    response.set_chunked_content_provider(
                        "text/event-stream; charset=utf-8",
                        [&, slot, chat, common_chunk](size_t, httplib::DataSink & sink) mutable {
                            auto write = [&](const std::string & event) {
                                return sink.write(event.data(), event.size());
                            };
                            auto abort = [&]() {
                                scheduler.cancel(slot);
                                scheduler.acknowledge_cancel(slot);
                                sink.done();
                                return false;
                            };
                            try {
                                if (chat) {
                                    const json role = {
                                        { "index", 0 },
                                        { "delta", { { "role", "assistant" } } },
                                        { "finish_reason", nullptr }
                                    };
                                    if (!write("data: " + common_chunk(role) + "\n\n")) {
                                        return abort();
                                    }
                                }
                                std::string piece;
                                for (;;) {
                                    bool cancelled = false;
                                    {
                                        std::lock_guard<std::mutex> lock(slot->mutex);
                                        cancelled = slot->cancelled;
                                    }
                                    if (cancelled) {
                                        return abort();
                                    }
                                    if (!scheduler.take_piece(slot, piece)) {
                                        {
                                            std::lock_guard<std::mutex> lock(slot->mutex);
                                            cancelled = slot->cancelled;
                                        }
                                        if (cancelled) {
                                            return abort();
                                        }
                                        break;
                                    }
                                    const json delta = chat
                                        ? json{ { "index", 0 },
                                                { "delta", { { "content", piece } } },
                                                { "finish_reason", nullptr } }
                                        : json{ { "content", piece } };
                                    const std::string event = chat
                                        ? "data: " + common_chunk(delta) + "\n\n"
                                        : "data: " + delta.dump() + "\n\n";
                                    if (!write(event)) {
                                        return abort();
                                    }
                                }
                                scheduler.wait_done(slot);
                                std::string error;
                                bool cancelled = false;
                                {
                                    std::lock_guard<std::mutex> lock(slot->mutex);
                                    cancelled = slot->cancelled;
                                    error = slot->error;
                                }
                                if (cancelled) {
                                    return abort();
                                }
                                if (!error.empty()) {
                                    write("data: " + error_json(error).dump() + "\n\n");
                                    sink.done();
                                    scheduler.release(slot);
                                    return false;
                                }
                                if (chat) {
                                    const json final_choice = {
                                        { "index", 0 }, { "delta", json::object() },
                                        { "finish_reason", "stop" }
                                    };
                                    if (!write("data: " + common_chunk(final_choice) + "\n\n")) {
                                        return abort();
                                    }
                                }
                                if (!write("data: [DONE]\n\n")) {
                                    return abort();
                                }
                                sink.done();
                                scheduler.release(slot);
                                return true;
                            } catch (const std::exception &) {
                                return abort();
                            }
                        });
                    return;
                }

                scheduler.wait_done(slot);
                std::vector<llama_token> generated;
                std::string error;
                {
                    std::lock_guard<std::mutex> lock(slot->mutex);
                    generated = slot->generated;
                    error = slot->error;
                }
                scheduler.release(slot);
                slot.reset();
                if (!error.empty()) {
                    throw std::runtime_error(error);
                }
                const std::string text = render_tokens(vocab, generated);
                json result;
                if (chat) {
                    result = {
                        { "id", id }, { "object", "chat.completion" }, { "created", created },
                        { "model", model_name }, { "choices", json::array({ json{
                            { "index", 0 },
                            { "message", { { "role", "assistant" }, { "content", text } } },
                            { "finish_reason", "stop" }
                        } }) },
                        { "usage", { { "prompt_tokens", prompt.size() },
                                     { "completion_tokens", generated.size() },
                                     { "total_tokens", prompt.size() + generated.size() } } }
                    };
                } else {
                    result = { { "content", text }, { "n_predict", generated.size() },
                               { "finish_reason", "stop" } };
                }
                set_common_headers(response);
                response.set_content(result.dump(), "application/json");
            } catch (const std::exception & e) {
                if (slot) {
                    scheduler.release(slot);
                }
                const std::string message = e.what();
                const bool client_error =
                    message == "missing prompt" || message == "prompt is empty" ||
                    message == "missing messages" || message.rfind("invalid ", 0) == 0 ||
                    message.rfind("unsupported field:", 0) == 0;
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
            const std::vector<llama_token> bench_tokens =
                serve(ring, vocab, bench_prompt, 8, {}, &stats, 0, {}, n_ubatch);
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
        const auto shutdown = [&]() {
            scheduler.stop();
            if (!session.ring.controls.empty()) {
                std::string reset_error;
                if (!reset_ring_workers(session.ring, reset_error) &&
                    !reset_error.empty()) {
                    std::fprintf(stderr, "potluck-server: shutdown reset: %s\n",
                                 reset_error.c_str());
                }
            }
            stop_planned_workers(session.workers);
            session.workers.clear();
            llama_model_free(meta);
        };
        bool listen_ok = false;
        try {
            if (server.bind_to_port(host.c_str(), http_port)) {
                server_signal_wakeup signal_wakeup([&] {
                    scheduler.request_stop();
                    server.stop();
                });
                listen_ok = server.listen_after_bind();
            }
        } catch (...) {
            shutdown();
            throw;
        }
        shutdown();
        if (!listen_ok) {
            fail("cannot bind HTTP port " + std::to_string(http_port));
        }
        return 0;
    } catch (const std::exception & e) {
        fail(e.what());
    }
}
