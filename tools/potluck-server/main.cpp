// potluck-server: an OpenAI-compatible HTTP server around a direct
// ZeroMQ piped ring. Requests traverse the configured ring route.

#include "llama.h"
#include "arg.h"
#include "chat.h"
#include "nlohmann/json.hpp"
#include "potluck-discovery.h"
#include "potluck-transport.h"
#include "potluck_runtime.h"
#include <cpp-httplib/httplib.h>
#include "internal.h"

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
std::atomic<int> signal_wakeup_fd{-1};
std::atomic<unsigned> signal_wakeup_active{0};
static_assert(std::atomic<int>::is_always_lock_free,
              "signal wakeup fd atomic must be lock-free");
static_assert(std::atomic<unsigned>::is_always_lock_free,
              "signal wakeup state atomic must be lock-free");

void signal_wakeup_handler(int) {
    signal_wakeup_active.fetch_add(1, std::memory_order_relaxed);
    const int fd = signal_wakeup_fd.load(std::memory_order_relaxed);
    if (fd >= 0) {
        const char byte = 1;
        (void) ::write(fd, &byte, sizeof(byte));
    }
    signal_wakeup_active.fetch_sub(1, std::memory_order_relaxed);
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
            signal_wakeup_fd.store(pipe_[1], std::memory_order_relaxed);
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
    }

    void close() noexcept {
        signal_wakeup_fd.store(-1, std::memory_order_relaxed);
        restore_signals();
        if (waiter_.joinable() && pipe_[1] >= 0) {
            const char byte = 0;
            (void) ::write(pipe_[1], &byte, sizeof(byte));
            waiter_.join();
        }
        while (signal_wakeup_active.load(std::memory_order_relaxed) != 0) {
            std::this_thread::yield();
        }
        if (pipe_[0] >= 0) {
            (void) ::close(pipe_[0]);
            pipe_[0] = -1;
        }
        if (pipe_[1] >= 0) {
            (void) ::close(pipe_[1]);
            pipe_[1] = -1;
        }
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
std::vector<std::string> parse_spec_types(const std::string & text) {
    const std::vector<std::string> names = split_csv(text);
    const std::vector<common_speculative_type> types =
        common_speculative_types_from_names(names);
    if (std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_NONE) != types.end()) {
        throw std::runtime_error("--spec-type must select an implementation");
    }
    return names;
}

uint32_t parse_nonnegative_u32(const std::string & value, const char * option) {
    try {
        size_t used = 0;
        const unsigned long parsed = std::stoul(value, &used);
        if (used != value.size() ||
            parsed > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("invalid value");
        }
        return static_cast<uint32_t>(parsed);
    } catch (...) {
        throw std::runtime_error(std::string(option) + " expects a non-negative integer");
    }
}


uint32_t parse_positive_u32(const std::string & value, const char * option) {
    try {
        size_t used = 0;
        const unsigned long parsed = std::stoul(value, &used);
        if (used != value.size() || parsed == 0 ||
            parsed > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("invalid value");
        }
        return static_cast<uint32_t>(parsed);
    } catch (...) {
        throw std::runtime_error(std::string(option) + " expects a positive integer");
    }
}
int32_t parse_positive_i32(const std::string & value, const char * option) {
    const uint32_t parsed = parse_positive_u32(value, option);
    if (parsed > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(std::string(option) + " exceeds the supported range");
    }
    return static_cast<int32_t>(parsed);
}

double parse_positive_double(const std::string & value, const char * option) {
    try {
        size_t used = 0;
        const double parsed = std::stod(value, &used);
        if (used != value.size() || !std::isfinite(parsed) || parsed <= 0.0) {
            throw std::runtime_error("invalid value");
        }
        return parsed;
    } catch (...) {
        throw std::runtime_error(std::string(option) + " expects a positive finite number");
    }
}

potluck::prefetch_mode parse_prefetch(const std::string & value) {
    if (value == "off") {
        return potluck::prefetch_mode::off;
    }
    if (value == "advise") {
        return potluck::prefetch_mode::advise;
    }
    if (value == "force") {
        return potluck::prefetch_mode::force;
    }
    throw std::runtime_error("--prefetch expects off, advise, or force");
}

std::vector<uint32_t> parse_layer_window(const std::string & text) {
    const std::vector<std::string> values = split_csv(text);
    if (values.size() > 32) {
        throw std::runtime_error("--layer-window accepts at most 32 entries");
    }
    std::vector<uint32_t> windows;
    windows.reserve(values.size());
    for (const std::string & value : values) {
        windows.push_back(parse_positive_u32(value, "--layer-window"));
    }
    return windows;
}

void validate_manual_workload(const std::vector<uint32_t> & layer_window,
                              int32_t k_override, uint32_t n_layer) {
    if (!layer_window.empty()) {
        uint64_t total = 0;
        for (const uint32_t layers : layer_window) {
            total += layers;
        }
        if (total == 0 || n_layer % total != 0) {
            throw std::runtime_error(
                "--layer-window sum must divide the model layer count exactly");
        }
    }
    if (k_override > 0 && n_layer % static_cast<uint32_t>(k_override) != 0) {
        throw std::runtime_error("--n-cycles must divide the model layer count exactly");
    }
}
const char * prefetch_name(potluck::prefetch_mode mode) {
    switch (mode) {
        case potluck::prefetch_mode::off: return "off";
        case potluck::prefetch_mode::advise: return "advise";
        case potluck::prefetch_mode::force: return "force";
    }
    return "advise";
}

void log_manual_overrides(const char * binary,
                          const std::vector<uint32_t> & layer_window,
                          double gpu_mem_gib, int32_t k_override,
                          double master_priority, potluck::prefetch_mode prefetch) {
    if (layer_window.empty() && gpu_mem_gib == 0.0 && k_override < 0 &&
        master_priority == 1.01 && prefetch == potluck::prefetch_mode::advise) {
        return;
    }
    std::fprintf(stderr, "%s: expert override", binary);
    if (!layer_window.empty()) {
        std::fprintf(stderr, " layer-window=");
        for (size_t index = 0; index < layer_window.size(); ++index) {
            std::fprintf(stderr, "%s%u", index == 0 ? "" : ",", layer_window[index]);
        }
    }
    if (gpu_mem_gib > 0.0) {
        std::fprintf(stderr, " gpu-mem=%.3fGiB", gpu_mem_gib);
    }
    if (k_override > 0) {
        std::fprintf(stderr, " n-cycles=%d", k_override);
    }
    if (master_priority != 1.01) {
        std::fprintf(stderr, " master-priority=%.6g", master_priority);
    }
    if (prefetch != potluck::prefetch_mode::advise) {
        std::fprintf(stderr, " prefetch=%s", prefetch_name(prefetch));
    }
    std::fputc('\n', stderr);
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
void validate_api_key(const std::string & api_key) {
    if (api_key.empty()) {
        throw std::runtime_error("--api-key cannot be empty");
    }
    if (api_key.size() > 8185) {
        throw std::runtime_error("--api-key is too long");
    }
    for (const unsigned char value : api_key) {
        if (value == '\r' || value == '\n' || value == 0) {
            throw std::runtime_error("--api-key contains an invalid control character");
        }
    }
}

void validate_cors_origin(const std::string & origin) {
    if (origin.empty()) {
        throw std::runtime_error("--cors-origin cannot be empty");
    }
    if (origin == "*") {
        throw std::runtime_error("--cors-origin does not allow wildcard origins");
    }
    if (origin.find(',') != std::string::npos) {
        throw std::runtime_error("--cors-origin accepts one exact origin");
    }
    for (const unsigned char value : origin) {
        if (value <= 0x20 || value == 0x7f) {
            throw std::runtime_error("--cors-origin contains whitespace or a control character");
        }
    }
    const size_t scheme_separator = origin.find("://");
    if (scheme_separator == std::string::npos || scheme_separator == 0 ||
        scheme_separator + 3 >= origin.size()) {
        throw std::runtime_error("--cors-origin must be an origin such as http://localhost:3000");
    }
    const std::string authority = origin.substr(scheme_separator + 3);
    if (authority.find('/') != std::string::npos ||
        authority.find('?') != std::string::npos ||
        authority.find('#') != std::string::npos) {
        throw std::runtime_error("--cors-origin must not include a path, query, or fragment");
    }
}

std::string resolve_hf_model(const std::string & repo,
                             const std::string & file,
                             const std::string & token,
                             bool offline,
                             std::string & resolved_repo,
                             std::string & resolved_file) {
    resolved_repo.clear();
    resolved_file.clear();
    common_params params;
    params.hf_token = token;
    params.offline = offline;
    params.model.hf_repo = repo;
    params.model.hf_file = file;

    try {
        common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
        if (common_models_handler_is_preset_repo(handler)) {
            throw std::runtime_error("repository contains a preset, not a GGUF model");
        }
        if (handler.plan.primary.path.empty() || handler.plan.model_files.empty()) {
            throw std::runtime_error("repository has no compatible GGUF model");
        }
        if (handler.plan.model_files.size() != 1) {
            throw std::runtime_error("split GGUF repositories are not supported");
        }
        resolved_repo = common_download_split_repo_tag(repo).first;
        resolved_file = handler.plan.primary.path;
        common_models_handler_apply(handler, params);
    } catch (const std::exception & e) {
        throw std::runtime_error("failed to resolve Hugging Face model '" + repo + "': " + e.what());
    }

    if (params.model.path.empty()) {
        throw std::runtime_error("failed to resolve Hugging Face model '" + repo +
                                 "': download did not produce a model path");
    }
    std::error_code path_error;
    if (!std::filesystem::is_regular_file(params.model.path, path_error)) {
        std::string detail = path_error ? ": " + path_error.message() : std::string();
        throw std::runtime_error("failed to resolve Hugging Face model '" + repo +
                                 "': resolved path is not a file: " + params.model.path + detail);
    }
    return params.model.path;
}











} // namespace

int main(int argc, char ** argv) {
    try {
        std::string model_path;
        std::string hf_repo;
        std::string hf_file;
        std::string hf_remote_repo;
        std::string hf_remote_file;
        const char * hf_token_env = std::getenv("HF_TOKEN");
        std::string hf_token = hf_token_env == nullptr ? std::string() : hf_token_env;
        bool hf_offline = false;
        std::string host = "0.0.0.0";
        std::string hosts_spec;
        std::string launch;
        std::string head_share = "auto";
        uint16_t http_port = 8080;
        uint32_t worker_local = 1;
        uint32_t n_predict_default = 24;
        uint32_t n_ctx = 4096;
        uint32_t n_seq_max = 4;
        uint32_t n_ubatch = 512;
        uint32_t seed = 0;
        float temp = 0.0f;
        float top_p = 0.0f;
        potluck::prefetch_mode prefetch = potluck::prefetch_mode::advise;
        bool force_prefetch = false;
        std::vector<uint32_t> layer_window;
        double gpu_mem_gib = 0.0;
        int32_t k_override = -1;
        double master_priority = 1.01;
        std::string spec_draft_model;
        std::vector<std::string> spec_types;
        uint32_t spec_draft_n_max = 0;
        bool bench = false;
        bool workers_option = false;
        std::string api_key;
        std::string cors_origin;
        bool api_key_option = false;
        bool cors_origin_option = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto take = [&](const char * option) -> std::string {
                if (++i >= argc) {
                    throw std::runtime_error(std::string("missing value for ") + option);
                }
                return argv[i];
            };
            if (arg == "-m" || arg == "--model") model_path = take("--model");
            else if (arg == "-hf" || arg == "-hfr" || arg == "--hf" || arg == "--hf-repo") {
                hf_repo = take("--hf-repo");
            }
            else if (arg == "-hff" || arg == "--hf-file") hf_file = take("--hf-file");
            else if (arg == "-hft" || arg == "--hf-token") hf_token = take("--hf-token");
            else if (arg == "--offline") hf_offline = true;
            else if (arg == "--host") host = take("--host");
            else if (arg == "--port") http_port = static_cast<uint16_t>(std::stoul(take("--port")));
            else if (arg == "--api-key") {
                api_key = take("--api-key");
                api_key_option = true;
            }
            else if (arg == "--cors-origin") {
                cors_origin = take("--cors-origin");
                cors_origin_option = true;
            }
            else if (arg == "--workers") {
                worker_local = static_cast<uint32_t>(std::stoul(take("--workers")));
                workers_option = true;
            }
            else if (arg == "--hosts") hosts_spec = take("--hosts");
            else if (arg == "--launch") launch = take("--launch");
            else if (arg == "--head-share") head_share = take("--head-share");
            else if (arg == "-c" || arg == "--ctx" || arg == "--ctx-size") {
                n_ctx = static_cast<uint32_t>(std::stoul(take(arg.c_str())));
            }
            else if (arg == "--slots" || arg == "--batch") {
                n_seq_max = std::max<uint32_t>(1, static_cast<uint32_t>(std::stoul(take(arg.c_str()))));
            }
            else if (arg == "--ubatch") {
                n_ubatch = std::max<uint32_t>(1, static_cast<uint32_t>(std::stoul(take("--ubatch"))));
            }
            else if (arg == "--temp") temp = std::stof(take("--temp"));
            else if (arg == "--top-p") top_p = std::stof(take("--top-p"));
            else if (arg == "--seed") seed = static_cast<uint32_t>(std::stoul(take("--seed")));
            else if (arg == "-n" || arg == "--n-predict") {
                n_predict_default = static_cast<uint32_t>(std::stoul(take(arg.c_str())));
            }
            else if (arg == "-lw" || arg == "--layer-window" || arg == "--n-layer-window") {
                layer_window = parse_layer_window(take(arg.c_str()));
            }
            else if (arg == "-gm" || arg == "--gpu-mem") {
                gpu_mem_gib = parse_positive_double(take(arg.c_str()), arg.c_str());
            }
            else if (arg == "-k" || arg == "--n-cycles") {
                k_override = parse_positive_i32(take(arg.c_str()), arg.c_str());
            }
            else if (arg == "--master-priority") {
                master_priority = parse_positive_double(take("--master-priority"),
                                                        "--master-priority");
            }
            else if (arg == "--prefetch") {
                prefetch = parse_prefetch(take("--prefetch"));
            }
            else if (arg == "--force") {
                force_prefetch = true;
            }
            else if (arg == "--spec-draft-model" || arg == "-md" ||
                     arg == "--model-draft") {
                spec_draft_model = take(arg.c_str());
            }
            else if (arg == "--spec-type") {
                spec_types = parse_spec_types(take("--spec-type"));
            }
            else if (arg == "--spec-draft-n-max") {
                spec_draft_n_max = parse_nonnegative_u32(
                    take("--spec-draft-n-max"), "--spec-draft-n-max");
            }
            else if (arg == "--bench") bench = true;
            else throw std::runtime_error(
                "usage: potluck-server -m model.gguf | -hf owner/repo[:quant] "
                "[--hf-file FILE] [--hf-token TOKEN] [--offline] "
                "[-c N] [-n N] [--slots N] [--ubatch N] "
                "[--temp F] [--top-p F] [--seed N] "
                "[-lw A,B | -gm N | -k N] [--master-priority F] "
                "[--prefetch off|advise|force] [--force] "
                "[--spec-type TYPE[,TYPE]] [--spec-draft-model FILE] "
                "[--spec-draft-n-max N] "
                "[--api-key VALUE] [--cors-origin ORIGIN] "
                "[--head-share auto|off] [--hosts a,b,c --launch ssh]");
        }
        if (force_prefetch) {
            prefetch = potluck::prefetch_mode::force;
        }
        if (api_key_option) {
            validate_api_key(api_key);
        }
        if (cors_origin_option) {
            validate_cors_origin(cors_origin);
        }
        if (model_path.empty() && hf_repo.empty()) {
            throw std::runtime_error("need -m model.gguf or -hf owner/repo[:quant]");
        }
        if (!model_path.empty() && !hf_repo.empty()) {
            throw std::runtime_error("cannot combine -m/--model with -hf/--hf-repo");
        }
        if (!hf_file.empty() && hf_repo.empty()) {
            throw std::runtime_error("-hff/--hf-file requires -hf/--hf-repo");
        }
        if (!hf_repo.empty()) {
            model_path = resolve_hf_model(hf_repo, hf_file, hf_token, hf_offline,
                                          hf_remote_repo, hf_remote_file);
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
            throw std::runtime_error("cannot load model metadata from " + model_path);
        }
        std::unique_ptr<llama_model, decltype(&llama_model_free)> metadata(meta, llama_model_free);

        try {
        const llama_vocab * vocab = llama_model_get_vocab(meta);
        const uint32_t n_layer = static_cast<uint32_t>(llama_model_n_layer(meta));
        if (n_layer == 0) {
            throw std::runtime_error("model metadata reports zero layers");
        }
        validate_manual_workload(layer_window, k_override, n_layer);
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
        const std::string model_identity = hf_repo.empty() ? basename_of(model_path) : hf_repo;
        const std::string worker_model_name = basename_of(model_path);
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
        ring_startup_options options;
        options.hosts_spec = hosts_spec;
        options.model_path = model_path;
        options.hf_repo = hf_remote_repo;
        options.hf_file = hf_remote_file;
        options.hf_token = hf_token;
        options.hf_offline = hf_offline;
        options.model_name = worker_model_name;
        options.worker_path = exe_dir(argv[0]) + "/potluck-worker";
        options.host = host;
        options.head_share = head_share;
        options.local_platform = first_command_line("uname -sm 2>/dev/null");
        options.adjacent_root =
            std::filesystem::path(options.worker_path).parent_path().parent_path().parent_path();
        const std::filesystem::path cwd_stage_dir =
            std::filesystem::current_path() / "dist/mac-arm64";
        options.stage_dir = std::filesystem::is_regular_file(cwd_stage_dir / "potluck-build-id")
            ? cwd_stage_dir : options.adjacent_root / "dist/mac-arm64";
        options.has_staged_payload =
            std::filesystem::is_regular_file(options.stage_dir / "potluck-build-id");
        options.bootstrap_nodes = &bootstrap_nodes;
        options.digest_cache = &digest_cache;
        options.workers_option = workers_option;
        options.worker_local = worker_local;
        options.n_layer = n_layer;
        options.head_dim = head_dim;
        options.n_head_kv = llama_model_n_head_kv(meta);
        options.n_ctx = n_ctx;
        options.n_seq_max = n_seq_max;
        options.n_ubatch = n_ubatch;
        options.speculative_n_rs_seq = potluck_speculative_n_rs_seq(
            spec_types, spec_draft_model, spec_draft_n_max);
        options.speculative_head_reserve = potluck_speculative_head_reserve(
            options.n_head_kv, options.head_dim, options.n_ctx, options.n_layer,
            options.n_seq_max, spec_types, spec_draft_model);
        options.seed = seed;
        options.temp = temp;
        options.top_p = top_p;
        options.prefetch = prefetch;
        options.layer_window = layer_window;
        options.gpu_mem_gib = gpu_mem_gib;
        options.k_override = k_override;
        options.master_priority = master_priority;
        log_manual_overrides("potluck-server", layer_window, gpu_mem_gib,
                             k_override, master_priority, prefetch);

        httplib::Server server;
        slot_speculative_config speculative;
        speculative.draft_model = spec_draft_model;
        speculative.types = spec_types;
        speculative.n_draft = spec_draft_n_max;
        speculative.n_ctx = n_ctx;
        speculative.n_batch = n_ubatch;
        speculative.n_ubatch = n_ubatch;

        slot_scheduler scheduler(session.ring, vocab, n_seq_max, n_ubatch,
                                 [&](std::string & error) {
                                     return rebuild_ring(session, options, false, error);
                                 },
                                 [&](std::string & error) {
                                     return heartbeat_ring(session, error);
                                 },
                                 [&](std::string & error) {
                                     return refresh_ring_if_needed(session, options, error);
                                 },
                                 meta, std::move(speculative));
        session.health_reason = "ring startup in progress";
        setup_http_routes(server, session, scheduler, vocab, model_identity,
                          chat_templates, n_predict_default, temp, top_p, seed,
                          api_key, cors_origin);

        std::thread listener_thread;
        std::mutex listener_mutex;
        std::string listener_error;
        std::atomic<bool> listener_failed_state{false};
        std::atomic<bool> listen_ok{false};
        std::atomic<bool> signal_requested{false};
        std::unique_ptr<server_signal_wakeup> signal_wakeup;
        bool shutdown_done = false;

        const auto listener_failure = [&](const char * message) {
            try {
                std::lock_guard<std::mutex> lock(listener_mutex);
                listener_error = message == nullptr ? "HTTP listener failed" : message;
            } catch (...) {
            }
            listener_failed_state.store(true, std::memory_order_release);
            session.stopping.store(true, std::memory_order_release);
            scheduler.request_stop();
            server.stop();
            server.decommission();
        };
        const auto listener_failure_message = [&]() {
            std::lock_guard<std::mutex> lock(listener_mutex);
            return listener_error.empty() ? std::string("HTTP listener stopped unexpectedly")
                                          : listener_error;
        };

        const auto shutdown = [&]() {
            session.stopping.store(true, std::memory_order_release);
            server.stop();
            scheduler.request_stop();
            if (listener_thread.joinable()) {
                listener_thread.join();
            }
            scheduler.stop();
            signal_wakeup.reset();
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
            metadata.reset();
        };
        const auto shutdown_once = [&]() {
            if (!shutdown_done) {
                shutdown_done = true;
                shutdown();
            }
        };

        try {
            signal_wakeup = std::make_unique<server_signal_wakeup>([&] {
                signal_requested.store(true, std::memory_order_release);
                session.stopping.store(true, std::memory_order_release);
                scheduler.request_stop();
                server.stop();
            });
            if (!server.bind_to_port(host.c_str(), http_port)) {
                if (signal_requested.load(std::memory_order_acquire)) {
                    shutdown_once();
                    return 0;
                }
                throw std::runtime_error("cannot bind HTTP port " + std::to_string(http_port));
            }
            listener_thread = std::thread([&] {
                try {
                    const bool ok = server.listen_after_bind();
                    listen_ok.store(ok, std::memory_order_release);
                    if (!ok) {
                        listener_failure("HTTP listener stopped unexpectedly");
                    }
                } catch (const std::exception & exception) {
                    listener_failure(exception.what());
                } catch (...) {
                    listener_failure("HTTP listener failed with an unknown exception");
                }
            });
            server.wait_until_ready();
            if (!server.is_running()) {
                if (signal_requested.load(std::memory_order_acquire)) {
                    shutdown_once();
                    return 0;
                }
                throw std::runtime_error(listener_failure_message());
            }
            std::printf("potluck-server: HTTP bound while loading ring at http://%s:%u (model %s)\n",
                        host.c_str(), http_port, model_path.c_str());
            std::fflush(stdout);

            std::string startup_error;
            if (!bring_up_ring(session, options, startup_error)) {
                if (signal_requested.load(std::memory_order_acquire)) {
                    shutdown_once();
                    return 0;
                }
                throw std::runtime_error(startup_error);
            }
            if (session.stopping.load(std::memory_order_acquire)) {
                if (signal_requested.load(std::memory_order_acquire)) {
                    shutdown_once();
                    return 0;
                }
                throw std::runtime_error(listener_failure_message());
            }

            scheduler.start();
            std::printf("potluck-server: listening on http://%s:%u (%zu workers, %zu ring windows, model %s)\n",
                        host.c_str(), http_port, session.ring.workers.size(), session.ring.windows.size(), model_path.c_str());
            std::fflush(stdout);
            if (bench) {
                const std::vector<llama_token> bench_prompt = tokenize_prompt(vocab, "The capital of France is");
                const auto start = std::chrono::steady_clock::now();
                const auto bench_slots = scheduler.acquire_many(
                    bench_prompt, 8, {}, {}, false, false, "bench", 0, 1);
                if (bench_slots.size() != 1) {
                    throw std::runtime_error("cannot acquire a benchmark conversation slot");
                }
                const std::shared_ptr<scheduled_slot> & bench_slot = bench_slots.front();
                scheduler.wait_done(bench_slot);
                std::vector<llama_token> bench_tokens;
                std::string bench_error;
                {
                    std::lock_guard<std::mutex> lock(bench_slot->mutex);
                    bench_tokens = bench_slot->generated;
                    bench_error = bench_slot->error;
                }
                scheduler.release(bench_slot);
                if (!bench_error.empty()) {
                    throw std::runtime_error("benchmark failed: " + bench_error);
                }
                const double wall = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
                const double aggregate = wall > 0.0 ? bench_tokens.size() / wall : 0.0;
                const uint64_t model_bytes = potluck::model_file_bytes(model_path);
                std::printf("bench ring route windows=%zu\n", session.ring.windows.size());
                for (size_t i = 0; i < session.ring.windows.size(); ++i) {
                    const potluck::ring_window & window = session.ring.windows[i];
                    const uint64_t bytes = model_bytes * (window.end - window.start) /
                        std::max<uint32_t>(1, n_layer);
                    std::printf("bench ring window %zu owner=%u [%u,%u) weight-bytes=%llu n_gpu_layers=%d\n",
                                i, window.owner, window.start, window.end,
                                static_cast<unsigned long long>(bytes), window.n_gpu_layers);
                }
                std::printf("bench ring aggregate-tok/s %.2f ms/token %.2f head-peak-rss-mb %.1f\n",
                            aggregate, aggregate > 0.0 ? 1000.0 / aggregate : 0.0,
                            peak_rss_mb());
                std::fflush(stdout);
            }
            if (listener_thread.joinable()) {
                listener_thread.join();
            }
            if (listener_failed_state.load(std::memory_order_acquire) &&
                !signal_requested.load(std::memory_order_acquire)) {
                throw std::runtime_error(listener_failure_message());
            }
            shutdown_once();
        } catch (...) {
            shutdown_once();
            throw;
        }
        } catch (...) {
            metadata.reset();
            throw;
        }
        return 0;
    } catch (const std::exception & e) {
        fail(e.what());
    }
}
