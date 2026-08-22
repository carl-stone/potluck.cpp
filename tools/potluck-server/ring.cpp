// potluck-server ring launch, configuration, and batch transport.

#include "internal.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

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
bool bring_up_ring(ring_session & target, ring_startup_options & options,
                   std::string & error) {
    if (options.bootstrap_nodes == nullptr || options.digest_cache == nullptr) {
        error = "ring startup options are incomplete";
        return false;
    }
    std::vector<bootstrap_node> & bootstrap_nodes = *options.bootstrap_nodes;
    model_digest_cache & digest_cache = *options.digest_cache;
    const std::string & hosts_spec = options.hosts_spec;
    const bool workers_option = options.workers_option;
    const uint32_t worker_local = options.worker_local;
    const std::string & model_path = options.model_path;
    const std::string & model_name = options.model_name;
    const std::string & host = options.host;
    const std::string & head_share = options.head_share;
    const uint32_t n_layer = options.n_layer;
    const uint32_t head_dim = options.head_dim;
    const uint32_t n_head_kv = options.n_head_kv;
    const uint32_t n_ctx = options.n_ctx;
    const uint32_t n_seq_max = options.n_seq_max;
    const uint32_t n_ubatch = options.n_ubatch;
    const uint32_t seed = options.seed;
    const float temp = options.temp;
    const float top_p = options.top_p;
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
        const std::string & worker_path = options.worker_path;
        if (worker_path == "/potluck-worker") {
            throw std::runtime_error("cannot locate potluck-worker beside potluck-server");
        }
        const std::filesystem::path & adjacent_root = options.adjacent_root;
        const std::filesystem::path & stage_dir = options.stage_dir;
        const bool has_staged_payload = options.has_staged_payload;
        const std::string & local_platform = options.local_platform;
        if (has_remote && has_staged_payload) {
            for (const bootstrap_node & bootstrap : bootstrap_nodes) {
                (void) refresh_remote_binaries(bootstrap, stage_dir, local_platform);
            }
        }

 
        const uint64_t model_bytes = std::filesystem::file_size(model_path);
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
}
bool rebuild_ring(ring_session & session, ring_startup_options & options,
                  std::string & error) {
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
    if (!bring_up_ring(replacement, options, error)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(session.mutex);
        session.ring = std::move(replacement.ring);
        session.workers = std::move(replacement.workers);
        session.healthy = true;
    }
    return true;
}
