// potluck-server: an OpenAI-compatible HTTP server around a distributed
// potluck.cpp layer chain. Each request owns the chain for its full decode;
// concurrent requests receive 429 instead of corrupting recurrent state.

#include "llama.h"
#include "common.h"
#include "chat.h"
#include "nlohmann/json.hpp"
#include "potluck-protocol.h"
#include "potluck-transport.h"
#include "potluck_runtime.h"
#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
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

std::vector<uint32_t> parse_bounds(const std::string & text) {
    const std::vector<std::string> parts = split_csv(text);
    std::vector<uint32_t> values;
    values.reserve(parts.size());
    for (const std::string & part : parts) {
        size_t used = 0;
        const unsigned long value = std::stoul(part, &used);
        if (used != part.size() || value > UINT32_MAX) {
            throw std::runtime_error("invalid --bounds value: " + part);
        }
        values.push_back(static_cast<uint32_t>(value));
    }
    return values;
}

std::vector<std::string> parse_hosts(const std::string & text) {
    return split_csv(text);
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

std::string local_shard_path(const std::string & directory, const std::string & source,
                             uint32_t index, uint32_t count) {
    return directory + "/" + model_stem(source) + ".potluck-" + std::to_string(index) +
           "of" + std::to_string(count) + ".gguf";
}

bool launch_remote_worker(const std::string & host, const std::string & model,
                          uint16_t port, uint32_t index) {
    const std::string log = "worker-" + std::to_string(index) + ".log";
    const std::string remote = "cd ~/potluck && nohup ./potluck-worker " + shell_quote(model) +
                               " 0.0.0.0 " + std::to_string(port) + " >" + shell_quote(log) +
                               " 2>&1 < /dev/null &";
    const std::string command = "ssh -o BatchMode=yes " + shell_quote(host) + " " + shell_quote(remote);
    std::printf("potluck-server: launch[%u] %s\n", index, command.c_str());
    std::fflush(stdout);
    const int rc = std::system(command.c_str());
    if (rc == 0) {
        return true;
    }
    const std::string tail_command = "ssh -o BatchMode=yes " + shell_quote(host) +
                                     " " + shell_quote("tail -n 40 ~/potluck/" + log);
    std::fprintf(stderr, "potluck-server: SSH launch failed (exit %d): %s\n", rc, command.c_str());
    std::fprintf(stderr, "potluck-server: remote log tail command: %s\n", tail_command.c_str());
    std::system(tail_command.c_str());
    return false;
}

struct ServerChain {
    std::vector<potluck::node_addr> workers;
    std::vector<uint32_t> bounds;
    potluck::tcp_channel stage0;
    potluck::tcp_channel result;
    std::string error;
};
struct serve_stats {
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    uint64_t coordinator_payload_bytes = 0;
};


std::vector<potluck::node_addr> spawn_local(const std::string & worker_path,
                                             const std::vector<std::string> & models,
                                             const std::string & host) {
    std::vector<potluck::node_addr> addresses;
    addresses.reserve(models.size());
    for (uint32_t index = 0; index < models.size(); ++index) {
        const uint16_t port = free_port();
        const std::string port_text = std::to_string(port);
        const pid_t pid = fork();
        if (pid < 0) {
            throw std::runtime_error("cannot fork potluck-worker");
        }
        if (pid == 0) {
            std::vector<std::string> args = { worker_path, models[index], host, port_text };
            std::vector<char *> argv;
            argv.reserve(args.size() + 1);
            for (std::string & arg : args) {
                argv.push_back(arg.data());
            }
            argv.push_back(nullptr);
            execv(worker_path.c_str(), argv.data());
            std::_Exit(127);
        }
        (void) pid;
        addresses.push_back({ host, port });
    }
    return addresses;
}

std::vector<potluck::node_addr> read_workers_file(const std::string & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open workers file: " + path);
    }
    std::vector<potluck::node_addr> addresses;
    std::string line;
    while (std::getline(input, line)) {
        const size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line.resize(comment);
        }
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r')) {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        const size_t colon = line.rfind(':');
        if (colon == std::string::npos || colon == 0 || colon + 1 == line.size()) {
            throw std::runtime_error("invalid worker address: " + line);
        }
        addresses.push_back({ line.substr(0, colon), static_cast<uint16_t>(std::stoul(line.substr(colon + 1))) });
    }
    return addresses;
}

ServerChain wire_chain(const std::vector<potluck::node_addr> & addresses,
                       const std::string & head_host, uint32_t n_layer,
                       uint32_t n_ctx, uint32_t n_seq_max, uint32_t n_ubatch,
                       uint32_t seed, float temp, float top_p,
                       const std::vector<int32_t> & ngl,
                       std::vector<uint32_t> bounds) {
    if (addresses.empty()) {
        throw std::runtime_error("no workers configured");
    }
    const uint32_t n_workers = static_cast<uint32_t>(addresses.size());
    if (bounds.empty()) {
        bounds.resize(n_workers + 1);
        for (uint32_t i = 0; i <= n_workers; ++i) {
            bounds[i] = static_cast<uint32_t>((static_cast<uint64_t>(n_layer) * i) / n_workers);
        }
    }
    if (bounds.size() != n_workers + 1 || bounds.front() != 0 || bounds.back() != n_layer) {
        throw std::runtime_error("--bounds must contain n_workers+1 values from 0 to n_layer");
    }
    for (size_t i = 1; i < bounds.size(); ++i) {
        if (bounds[i] <= bounds[i - 1]) {
            throw std::runtime_error("worker bounds must be strictly increasing");
        }
    }

    ServerChain chain;
    chain.workers = addresses;
    chain.bounds = bounds;
    potluck::tcp_listener result_listener = potluck::tcp_listener::bind_host(head_host, 0);
    if (!result_listener.valid()) {
        throw std::runtime_error("cannot bind chain result listener");
    }

    potluck::node_config config;
    config.n_workers = n_workers;
    config.n_ctx = n_ctx;
    config.n_seq_max = n_seq_max;
    config.n_ubatch = n_ubatch;
    config.seed = seed;
    config.temp = temp;
    config.top_p = top_p;
    config.bounds = bounds;
    config.workers = addresses;
    config.ngl = ngl;
    config.head_link = { head_host, result_listener.port() };

    chain.stage0 = potluck::connect_retry(addresses[0].host, addresses[0].port, 9000, 200, chain.error);
    if (!chain.stage0.valid()) {
        throw std::runtime_error("cannot connect to worker 0: " + chain.error);
    }
    chain.stage0.set_timeouts(potluck::handshake_timeout_s(), potluck::handshake_timeout_s());
    std::vector<uint8_t> payload;
    if (!potluck::encode_config(config, payload)) {
        throw std::runtime_error("cannot encode worker configuration");
    }
    potluck::message config_message;
    config_message.type = potluck::message_type::node_config;
    config_message.payload = std::move(payload);
    if (!chain.stage0.send(config_message, chain.error)) {
        throw std::runtime_error("cannot send worker configuration: " + chain.error);
    }
    // Stage 0 reports ready only after every downstream worker has wired its
    // link. Receive first so a rejected shard closes the timed handshake
    // instead of leaving this result-listener accept blocked forever.
    potluck::message ready;
    if (!chain.stage0.receive(ready, chain.error) || ready.type != potluck::message_type::ready) {
        throw std::runtime_error("chain never became ready: " + chain.error);
    }
    chain.result = result_listener.accept(chain.error);
    if (!chain.result.valid()) {
        throw std::runtime_error("tail never connected back: " + chain.error);
    }
    chain.stage0.set_timeouts(potluck::decode_timeout_s(), potluck::decode_timeout_s());
    chain.result.set_timeouts(potluck::decode_timeout_s(), potluck::decode_timeout_s());
    std::printf("potluck-server: chain ready (%u workers)\n", n_workers);
    std::fflush(stdout);
    return chain;
}

std::vector<int32_t> drive_batch(ServerChain & chain,
                                 const std::vector<int32_t> & positions,
                                 const std::vector<int32_t> & sequences,
                                 const std::vector<int32_t> & tokens,
                                 int32_t clear, int32_t trim_to, uint32_t n_logits,
                                 serve_stats * stats = nullptr) {
    if (positions.empty() || positions.size() != sequences.size() || positions.size() != tokens.size()) {
        throw std::runtime_error("invalid batch dimensions");
    }
    potluck::message input;
    input.type = potluck::message_type::batch_decode;
    input.sequence = static_cast<uint64_t>(positions.back());
    if (!potluck::encode_batch_payload(positions, sequences, tokens, nullptr, 0,
                                       clear, trim_to, n_logits, input.payload)) {
        throw std::runtime_error("cannot encode chain batch");
    }
    if (stats != nullptr) {
        stats->coordinator_payload_bytes += input.payload.size();
    }
    if (!chain.stage0.send(input, chain.error)) {
        throw std::runtime_error("cannot send chain batch: " + chain.error);
    }
    potluck::message output;
    if (!chain.result.receive(output, chain.error)) {
        throw std::runtime_error("chain result channel closed: " + chain.error);
    }
    if (stats != nullptr) {
        stats->coordinator_payload_bytes += output.payload.size();
    }
    if (output.type != potluck::message_type::batch_result) {
        throw std::runtime_error("unexpected chain result message");
    }
    std::vector<int32_t> result_positions, result_sequences, result_tokens;
    std::vector<float> result_hidden;
    int32_t ignored_clear = 0, ignored_trim = -1;
    uint32_t ignored_logits = 0;
    if (!potluck::decode_batch_payload(output.payload.data(), output.payload.size(), 0,
                                       ignored_clear, ignored_trim, ignored_logits,
                                       result_positions, result_sequences, result_tokens,
                                       result_hidden, chain.error)) {
        throw std::runtime_error("cannot decode chain result: " + chain.error);
    }
    if (result_positions != positions || result_sequences != sequences || result_tokens.size() != positions.size()) {
        throw std::runtime_error("chain result entries do not match the request");
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

std::vector<potluck::worker_bench_metrics> request_worker_metrics(
        ServerChain & chain) {
    potluck::message request;
    request.type = potluck::message_type::profile_result;
    if (!chain.stage0.send(request, chain.error)) {
        throw std::runtime_error("cannot request worker benchmark metrics: " + chain.error);
    }
    potluck::message response;
    if (!chain.stage0.receive(response, chain.error) ||
        response.type != potluck::message_type::profile_result) {
        throw std::runtime_error("worker benchmark metrics response missing: " + chain.error);
    }
    std::vector<potluck::worker_bench_metrics> metrics;
    if (!potluck::decode_worker_bench_metrics(response.payload.data(), response.payload.size(),
                                               metrics, chain.error)) {
        throw std::runtime_error("cannot decode worker benchmark metrics: " + chain.error);
    }
    return metrics;
}


std::vector<llama_token> serve(ServerChain & chain, const llama_vocab * vocab,
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
    (void) drive_batch(chain, positions, sequences, tokens, 1, -1, 1, stats);
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
        const std::vector<int32_t> result = drive_batch(chain,
            { static_cast<int32_t>(position) }, { 0 }, { static_cast<int32_t>(previous) }, 0, -1, 1, stats);
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
        std::string workers_file;
        std::string hosts_spec;
        std::string launch;
        std::string bounds_spec;
        uint16_t http_port = 8080;
        uint16_t worker_port_base = 39271;
        uint32_t worker_local = 2;
        uint32_t gpu_layers = 0;
        uint32_t gpu_mem_mb = 0;
        uint32_t n_predict_default = 24;
        uint32_t n_ctx = 4096;
        uint32_t n_seq_max = 1;
        uint32_t n_ubatch = 1; // one-token internal ubatches preserve llama-cli numerics
        uint32_t seed = 0;
        float temp = 0.0f;
        float top_p = 0.0f;
        bool bench = false;

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
            else if (arg == "--workers") worker_local = static_cast<uint32_t>(std::stoul(take("--workers")));
            else if (arg == "--workers-file") workers_file = take("--workers-file");
            else if (arg == "--hosts") hosts_spec = take("--hosts");
            else if (arg == "--launch") launch = take("--launch");
            else if (arg == "--worker-port-base") worker_port_base = static_cast<uint16_t>(std::stoul(take("--worker-port-base")));
            else if (arg == "--bounds") bounds_spec = take("--bounds");
            else if (arg == "--gpu-layers" || arg == "--ngl") gpu_layers = static_cast<uint32_t>(std::stoul(take("--gpu-layers")));
            else if (arg == "--gpu-mem") gpu_mem_mb = static_cast<uint32_t>(std::stoul(take("--gpu-mem")));
            else if (arg == "--ctx") n_ctx = static_cast<uint32_t>(std::stoul(take("--ctx")));
            else if (arg == "--batch") n_seq_max = std::max<uint32_t>(1, static_cast<uint32_t>(std::stoul(take("--batch"))));
            else if (arg == "--temp") temp = std::stof(take("--temp"));
            else if (arg == "--top-p") top_p = std::stof(take("--top-p"));
            else if (arg == "--seed") seed = static_cast<uint32_t>(std::stoul(take("--seed")));
            else if (arg == "--n-predict") n_predict_default = static_cast<uint32_t>(std::stoul(take("--n-predict")));
            else if (arg == "--bench") bench = true;
            else throw std::runtime_error("usage: potluck-server -m model.gguf [--workers N | --hosts a,b,c] [--launch ssh] [--shard-dir DIR] [--bounds A,B,...]");
        }
        if (model_path.empty()) {
            throw std::runtime_error("need -m model.gguf");
        }
        if (!launch.empty() && launch != "ssh") {
            throw std::runtime_error("--launch supports only ssh");
        }
        if (!hosts_spec.empty() && (!workers_file.empty() || worker_local != 2)) {
            throw std::runtime_error("--hosts cannot be combined with --workers or --workers-file");
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
        const std::string model_name = basename_of(model_path);
        common_chat_templates_ptr chat_templates = common_chat_templates_init(meta, "");

        std::vector<potluck::node_addr> addresses;
        std::vector<std::string> worker_models;
        if (!hosts_spec.empty()) {
            const std::vector<std::string> hosts = parse_hosts(hosts_spec);
            worker_local = 0;
            addresses.reserve(hosts.size());
            for (uint32_t i = 0; i < hosts.size(); ++i) {
                addresses.push_back({ hosts[i], static_cast<uint16_t>(worker_port_base + i) });
            }
            if (shard_dir.empty()) {
                for (size_t i = 0; i < hosts.size(); ++i) {
                    // launch_remote_worker() changes to ~/potluck first, so a
                    // basename avoids quoting a literal "~" as a path.
                    worker_models.push_back(model_name);
                }
            } else {
                for (size_t i = 0; i < hosts.size(); ++i) {
                    worker_models.push_back(basename_of(local_shard_path(shard_dir, model_path, i, hosts.size())));
                }
            }
            if (launch == "ssh") {
                for (uint32_t i = 0; i < hosts.size(); ++i) {
                    if (!launch_remote_worker(hosts[i], worker_models[i], addresses[i].port, i)) {
                        throw std::runtime_error("remote worker launch failed");
                    }
                }
            }
        } else if (!workers_file.empty()) {
            addresses = read_workers_file(workers_file);
        } else {
            const std::string worker_path = exe_dir(argv[0]) + "/potluck-worker";
            if (worker_path == "/potluck-worker") {
                throw std::runtime_error("cannot locate potluck-worker beside potluck-server");
            }
            worker_models.assign(worker_local, model_path);
            if (!shard_dir.empty()) {
                for (uint32_t i = 0; i < worker_local; ++i) {
                    worker_models[i] = local_shard_path(shard_dir, model_path, i, worker_local);
                }
            }
            addresses = spawn_local(worker_path, worker_models, host == "0.0.0.0" ? "127.0.0.1" : host);
        }
        if (addresses.empty()) {
            throw std::runtime_error("need at least one worker");
        }
        if (worker_models.empty()) {
            worker_models.assign(addresses.size(), model_path);
        }

        std::vector<uint32_t> bounds;
        if (!bounds_spec.empty()) {
            bounds = parse_bounds(bounds_spec);
        }
        if (!bounds.empty() && bounds.size() != addresses.size() + 1) {
            throw std::runtime_error("--bounds must have one more value than workers");
        }

        std::vector<int32_t> ngl(addresses.size(), 0);
        if (gpu_layers > 0 || gpu_mem_mb > 0) {
            uint32_t total = std::min(gpu_layers, n_layer);
            if (gpu_mem_mb > 0) {
                const uint64_t layer_bytes = std::max<uint64_t>(1, potluck::model_file_bytes(model_path) / n_layer);
                total = std::min<uint32_t>(n_layer, static_cast<uint32_t>((static_cast<uint64_t>(gpu_mem_mb) * 1024 * 1024) / layer_bytes));
            }
            const std::vector<uint32_t> effective = bounds.empty() ? [&] {
                std::vector<uint32_t> v(addresses.size() + 1);
                for (uint32_t i = 0; i <= addresses.size(); ++i) v[i] = static_cast<uint32_t>((static_cast<uint64_t>(n_layer) * i) / addresses.size());
                return v;
            }() : bounds;
            for (uint32_t i = 0; i < addresses.size(); ++i) {
                const int64_t remaining = static_cast<int64_t>(total) - effective[i];
                const int64_t width = effective[i + 1] - effective[i];
                ngl[i] = static_cast<int32_t>(std::max<int64_t>(0, std::min(remaining, width)));
            }
        }

        ServerChain chain = wire_chain(addresses, host == "0.0.0.0" ? "127.0.0.1" : host,
                                        n_layer, n_ctx, n_seq_max, n_ubatch,
                                        seed, temp, top_p, ngl, bounds);

        httplib::Server server;
        std::mutex chain_mutex;
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
            json health = { { "status", "ok" }, { "workers", chain.workers.size() }, { "windows", json::array() } };
            for (size_t i = 0; i < chain.workers.size(); ++i) {
                health["windows"].push_back(json{
                    { "index", i }, { "host", chain.workers[i].host }, { "port", chain.workers[i].port },
                    { "start", chain.bounds[i] }, { "end", chain.bounds[i + 1] }
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
                std::make_shared<std::unique_lock<std::mutex>>(chain_mutex, std::defer_lock);
            if (!busy->try_lock()) {
                response.status = 429;
                set_common_headers(response);
                response.set_content(error_json("chain is busy; retry later").dump(), "application/json");
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
                                (void) serve(chain, vocab, prompt, n_predict, [&](const std::string & piece) {
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

                const std::vector<llama_token> generated = serve(chain, vocab, prompt, n_predict, {});
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

        std::printf("potluck-server: listening on http://%s:%u (%zu workers, model %s)\n",
                    host.c_str(), http_port, addresses.size(), model_path.c_str());
        std::fflush(stdout);
        if (bench) {
            const std::vector<llama_token> bench_prompt = tokenize_prompt(vocab, "The capital of France is");
            serve_stats stats;
            const auto start = std::chrono::steady_clock::now();
            const std::vector<llama_token> bench_tokens = serve(chain, vocab, bench_prompt, 8, {}, &stats);
            const double wall = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            const double prefill = stats.prefill_seconds > 0.0
                ? bench_prompt.size() / stats.prefill_seconds : 0.0;
            const double decode = stats.decode_seconds > 0.0
                ? bench_tokens.size() / stats.decode_seconds : 0.0;
            const double aggregate = wall > 0.0 ? bench_tokens.size() / wall : 0.0;
            const double bytes_per_token = bench_tokens.empty()
                ? 0.0 : static_cast<double>(stats.coordinator_payload_bytes) / bench_tokens.size();
            const std::vector<potluck::worker_bench_metrics> metrics =
                request_worker_metrics(chain);
            if (metrics.size() != chain.workers.size()) {
                throw std::runtime_error("worker benchmark metrics count mismatch");
            }
            std::vector<float> worker_speed(chain.workers.size(), 0.0f);
            std::vector<float> worker_rss(chain.workers.size(), 0.0f);
            for (const auto & metric : metrics) {
                if (metric.index >= chain.workers.size()) {
                    throw std::runtime_error("worker benchmark metrics index out of range");
                }
                worker_speed[metric.index] = metric.decode_tok_s;
                worker_rss[metric.index] = metric.peak_rss_mb;
            }
            const float max_worker_rss = worker_rss.empty()
                ? 0.0f : *std::max_element(worker_rss.begin(), worker_rss.end());
            std::printf("bench worker host window       weight-bytes gpu-layers decode-tok/s peak-rss-mb\n");
            for (size_t i = 0; i < chain.workers.size(); ++i) {
                const uint64_t bytes = potluck::model_file_bytes(model_path) *
                    (chain.bounds[i + 1] - chain.bounds[i]) / std::max<uint32_t>(1, n_layer);
                std::printf("bench %6zu %-15s [%u,%u) %12llu %10d %12.2f %11.1f\n", i,
                            chain.workers[i].host.c_str(), chain.bounds[i], chain.bounds[i + 1],
                            static_cast<unsigned long long>(bytes), ngl[i],
                            worker_speed[i], worker_rss[i]);
            }
            std::printf("bench cluster prefill-tok/s %.2f decode-tok/s %.2f aggregate-tok/s %.2f "
                        "ms/token %.2f wire-bytes/token %.1f coordinator-peak-rss-mb %.1f "
                        "worker-peak-rss-mb-max %.1f\n",
                        prefill, decode, aggregate, aggregate > 0.0 ? 1000.0 / aggregate : 0.0,
                        bytes_per_token, peak_rss_mb(), max_worker_rss);
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
