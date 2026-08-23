// potluck-worker: controller-private direct ZeroMQ ring peer.
#include "potluck-transport.h"
#include "potluck_runtime.h"

#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

[[noreturn]] void fail(const std::string & what) {
    throw std::runtime_error(what);
}

const char * accel_kind_name(potluck::accel_kind kind) {
    switch (kind) {
        case potluck::accel_kind::metal:
            return "metal";
        case potluck::accel_kind::cuda:
            return "cuda";
        case potluck::accel_kind::other:
            return "other";
        default:
            return "none";
    }
}

void trace_prp_event(bool enabled, uint32_t rank, uint64_t sequence,
                     uint32_t window, uint32_t round, const char * event, size_t bytes) {
    if (!enabled) {
        return;
    }
    std::printf("PRP seq=%llu window=%u round=%u rank=%u event=%s bytes=%zu\n",
                static_cast<unsigned long long>(sequence), window, round, rank, event, bytes);
    std::fflush(stdout);
}


bool parse_u32(const std::string & text, uint32_t & value) {
    if (text.empty()) {
        return false;
    }
    size_t used = 0;
    try {
        const unsigned long parsed = std::stoul(text, &used, 10);
        if (used != text.size() || parsed > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        value = static_cast<uint32_t>(parsed);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool probe_local_accel(potluck::accel_profile & out, std::string & error) {
    const uint32_t rank = out.rank;
    out = potluck::accel_profile{};
    out.rank = rank;
    error.clear();

    const ggml_backend_dev_t cpu =
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu == nullptr) {
        error = "CPU backend is unavailable";
        return false;
    }
    size_t host_free = 0;
    size_t host_total = 0;
    ggml_backend_dev_memory(cpu, &host_free, &host_total);
    out.host_free_bytes = host_free;
    out.host_total_bytes = host_total;

    const enum ggml_backend_dev_type wanted[] = {
        GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_ACCEL
    };
    for (size_t ti = 0; ti < 2 && out.kind == potluck::accel_kind::none; ++ti) {
        for (size_t di = 0; di < ggml_backend_dev_count(); ++di) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(di);
            if (dev == nullptr || ggml_backend_dev_type(dev) != wanted[ti]) {
                continue;
            }
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            if (total_bytes == 0) {
                continue;
            }
            out.free_bytes = free_bytes;
            out.total_bytes = total_bytes;
            const std::string name = ggml_backend_dev_name(dev);
            const std::string description = ggml_backend_dev_description(dev);
            if (name.rfind("MTL", 0) == 0 || description.find("Metal") != std::string::npos) {
                out.kind = potluck::accel_kind::metal;
            } else if (name.rfind("CUDA", 0) == 0 || name.rfind("ROCm", 0) == 0 ||
                       name.rfind("MUSA", 0) == 0 ||
                       description.find("CUDA") != std::string::npos) {
                out.kind = potluck::accel_kind::cuda;
            } else {
                out.kind = potluck::accel_kind::other;
            }
            break;
        }
    }
    return true;
}
llama_sampler * make_sampler(const potluck::slot_config & config, uint32_t n_vocab,
                             std::string & error) {
    if (!std::isfinite(config.temp) || config.temp < 0.0f ||
        !std::isfinite(config.top_p) || config.top_p < 0.0f ||
        config.top_p > 1.0f || !std::isfinite(config.min_p) ||
        config.min_p < 0.0f || config.min_p > 1.0f ||
        !std::isfinite(config.presence_penalty) ||
        !std::isfinite(config.frequency_penalty) ||
        !std::isfinite(config.repeat_penalty) || config.repeat_penalty <= 0.0f ||
        config.penalty_last_n < -1 ||
        (config.top_logprobs != 0 && !config.logprobs)) {
        error = "slot sampler has invalid sampling parameters";
        return nullptr;
    }
    if (config.top_k > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
        n_vocab > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        error = "slot sampler top_k or vocabulary is too large";
        return nullptr;
    }
    const bool has_penalties =
        config.presence_penalty != 0.0f || config.frequency_penalty != 0.0f ||
        config.repeat_penalty != 1.0f;
    const bool has_filters =
        config.top_k > 0 || (config.top_p > 0.0f && config.top_p < 1.0f) ||
        config.min_p > 0.0f || has_penalties;
    if (config.temp == 0.0f && !has_filters) {
        return nullptr;
    }
    llama_sampler_chain_params params = llama_sampler_chain_default_params();
    llama_sampler * chain = llama_sampler_chain_init(params);
    if (chain == nullptr) {
        error = "cannot create slot sampler";
        return nullptr;
    }
    const auto add = [&](llama_sampler * sampler) {
        if (sampler == nullptr) {
            error = "cannot create slot sampler component";
            llama_sampler_free(chain);
            return false;
        }
        llama_sampler_chain_add(chain, sampler);
        return true;
    };
    if (has_penalties &&
        !add(llama_sampler_init_penalties(static_cast<int32_t>(n_vocab),
                                          config.penalty_last_n,
                                          config.repeat_penalty,
                                          config.frequency_penalty,
                                          config.presence_penalty))) {
        return nullptr;
    }
    if (config.top_k > 0 && !add(llama_sampler_init_top_k(static_cast<int32_t>(config.top_k)))) {
        return nullptr;
    }
    if (config.top_p > 0.0f && config.top_p < 1.0f &&
        !add(llama_sampler_init_top_p(config.top_p, 1))) {
        return nullptr;
    }
    if (config.min_p > 0.0f && !add(llama_sampler_init_min_p(config.min_p, 1))) {
        return nullptr;
    }
    if (config.temp > 0.0f) {
        if (!add(llama_sampler_init_temp(config.temp)) ||
            !add(llama_sampler_init_dist(config.seed))) {
            return nullptr;
        }
    } else if (!add(llama_sampler_init_greedy())) {
        return nullptr;
    }
    return chain;
}

std::vector<potluck::token_logprob> make_logprobs(
        llama_context * context, int32_t index, const float * logits,
        uint32_t n_vocab, const potluck::slot_config & config, llama_token sampled) {
    if (!config.logprobs || logits == nullptr || n_vocab == 0) {
        return {};
    }
    const float * sampled_probs = llama_get_sampled_probs_ith(context, index);
    llama_token * sampled_tokens = llama_get_sampled_candidates_ith(context, index);
    const uint32_t sampled_count = std::min(
        llama_get_sampled_probs_count_ith(context, index),
        llama_get_sampled_candidates_count_ith(context, index));
    if (sampled_probs != nullptr && sampled_tokens != nullptr && sampled_count != 0) {
        std::vector<potluck::token_logprob> result;
        const size_t keep = std::max<size_t>(1, config.top_logprobs);
        result.reserve(std::min<size_t>(sampled_count, keep) + 1);
        for (uint32_t i = 0; i < sampled_count && result.size() < keep; ++i) {
            const float probability = sampled_probs[i];
            if (probability > 0.0f && std::isfinite(probability)) {
                result.push_back({ static_cast<int32_t>(sampled_tokens[i]),
                                   std::log(probability) });
            }
        }
        if (result.empty()) {
            return {};
        }
        bool selected_present = false;
        for (const auto & item : result) {
            selected_present = selected_present || item.token == sampled;
        }
        if (!selected_present) {
            for (uint32_t i = 0; i < sampled_count; ++i) {
                if (sampled_tokens[i] == sampled && sampled_probs[i] > 0.0f &&
                    std::isfinite(sampled_probs[i])) {
                    result.push_back({ static_cast<int32_t>(sampled),
                                       std::log(sampled_probs[i]) });
                    break;
                }
            }
        }
        return result;
    }

    struct candidate {
        llama_token token;
        double logprob;
    };
    std::vector<candidate> candidates;
    candidates.reserve(n_vocab);
    float max_logit = -std::numeric_limits<float>::infinity();
    for (uint32_t token = 0; token < n_vocab; ++token) {
        max_logit = std::max(max_logit, logits[token]);
    }
    double normalizer = 0.0;
    for (uint32_t token = 0; token < n_vocab; ++token) {
        if (!std::isfinite(logits[token])) {
            continue;
        }
        const double log_weight = static_cast<double>(logits[token] - max_logit);
        normalizer += std::exp(log_weight);
        candidates.push_back({ static_cast<llama_token>(token), log_weight });
    }
    if (normalizer <= 0.0 || candidates.empty()) {
        return {};
    }
    const double log_normalizer = std::log(normalizer);
    for (auto & item : candidates) {
        item.logprob -= log_normalizer;
    }
    const size_t keep = std::max<size_t>(1, config.top_logprobs);
    const size_t limit = std::min(keep, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + limit, candidates.end(),
                      [](const candidate & lhs, const candidate & rhs) {
                          return lhs.logprob > rhs.logprob;
                      });
    std::vector<potluck::token_logprob> result;
    result.reserve(limit + 1);
    bool selected_present = false;
    for (size_t i = 0; i < limit; ++i) {
        result.push_back({ static_cast<int32_t>(candidates[i].token),
                           static_cast<float>(candidates[i].logprob) });
        selected_present = selected_present || candidates[i].token == sampled;
    }
    if (!selected_present) {
        const auto found = std::find_if(candidates.begin() + limit, candidates.end(),
                                         [sampled](const candidate & item) {
                                             return item.token == sampled;
                                         });
        if (found != candidates.end()) {
            result.push_back({ static_cast<int32_t>(found->token),
                               static_cast<float>(found->logprob) });
        }
    }
    return result;
}


void clear_stage(potluck::stage_model & stage) {
    if (stage.ctx != nullptr) {
        llama_memory_clear(llama_get_memory(stage.ctx), true);
    }
}

void clear_stages(std::vector<potluck::stage_model> & stages) {
    for (potluck::stage_model & stage : stages) {
        clear_stage(stage);
    }
}

void append_error_payload(potluck::message & message, const std::string & text) {
    message.payload.assign(text.begin(), text.end());
    if (message.payload.size() > potluck::max_payload_bytes) {
        message.payload.resize(potluck::max_payload_bytes);
    }
}
bool read_curve_bootstrap(potluck::curve_bootstrap_credentials & credentials,
                          std::string & error) {
    std::array<uint8_t, potluck::curve_bootstrap_record_size> record = {};
    const auto scrub_record = [&] {
        volatile uint8_t * bytes = record.data();
        for (size_t index = 0; index < record.size(); ++index) {
            bytes[index] = 0;
        }
    };
    size_t offset = 0;
    while (offset < record.size()) {
        const ssize_t count = read(STDIN_FILENO, record.data() + offset,
                                   record.size() - offset);
        if (count > 0) {
            offset += static_cast<size_t>(count);
            continue;
        }
        if (count < 0 && errno == EINTR) {
            continue;
        }
        error = count == 0
            ? "CURVE bootstrap stdin ended before the complete record"
            : "CURVE bootstrap stdin read failed";
        scrub_record();
        return false;
    }
    const bool decoded = potluck::decode_curve_bootstrap(
        record.data(), record.size(), credentials, error);
    scrub_record();
    return decoded;
}
bool validate_shard(const potluck::stage_model & stage, uint32_t index, uint32_t count,
                    uint32_t start, uint32_t end, std::string & error) {
    struct expected_value {
        const char * key;
        uint32_t value;
    };
    const expected_value expected[] = {
        { "potluck.shard.index", index },
        { "potluck.shard.count", count },
        { "potluck.shard.start", start },
        { "potluck.shard.end", end },
    };
    for (const expected_value & item : expected) {
        char value[32] = {};
        if (llama_model_meta_val_str(stage.model, item.key, value, sizeof(value)) <= 0) {
            error = std::string("shard is missing metadata key ") + item.key;
            return false;
        }
        uint32_t parsed = 0;
        if (!parse_u32(value, parsed) || parsed != item.value) {
            error = std::string("shard metadata mismatch for ") + item.key +
                " (expected " + std::to_string(item.value) + ", got " + value + ")";
            return false;
        }
    }
    return true;
}



} // namespace

int main(int argc, char ** argv) {
    std::vector<potluck::stage_model> stages;
    std::unordered_map<int32_t, llama_sampler *> samplers;
    std::unordered_map<int32_t, potluck::slot_config> sampler_configs;
    bool backend_initialized = false;
    uint32_t launch_rank = 0;
    potluck::curve_bootstrap_credentials curve_credentials;
    bool have_curve_credentials = false;
    potluck::ring_peer peer;
    potluck::ring_sender result_sender;
    std::mutex inbox_mutex;
    std::condition_variable inbox_cv;
    std::deque<potluck::message> inbox;
    std::string inbox_error;
    std::atomic<bool> receiver_stopping{false};
    std::thread receiver_thread;

    auto cleanup = [&]() noexcept {
        receiver_stopping.store(true);
        if (receiver_thread.joinable()) {
            receiver_thread.join();
        }
        for (auto & entry : samplers) {
            if (entry.second != nullptr) {
                llama_sampler_free(entry.second);
            }
        }
        samplers.clear();
        sampler_configs.clear();
        for (potluck::stage_model & stage : stages) {
            potluck::stage_free(stage);
        }
        stages.clear();
        if (have_curve_credentials) {
            potluck::scrub_curve_credentials(curve_credentials);
            have_curve_credentials = false;
        }
        if (backend_initialized) {
            llama_backend_free();
            backend_initialized = false;
        }
    };
    try {
        if (argc == 2 && std::string(argv[1]) == "--probe") {
            llama_backend_init();
            potluck::accel_profile probe;
            std::string probe_error;
            const bool probed = probe_local_accel(probe, probe_error);
            llama_backend_free();
            if (!probed) {
                fail(probe_error);
            }
            std::printf("potluck-probe kind=%s accel_free=%llu accel_total=%llu host_free=%llu host_total=%llu\n",
                        accel_kind_name(probe.kind),
                        static_cast<unsigned long long>(probe.free_bytes),
                        static_cast<unsigned long long>(probe.total_bytes),
                        static_cast<unsigned long long>(probe.host_free_bytes),
                        static_cast<unsigned long long>(probe.host_total_bytes));
            std::fflush(stdout);
            return 0;
        }
        if (argc < 10) {
            fail("usage: potluck-worker <model.gguf> --bind <endpoint> --next <endpoint> --result <endpoint> --rank N --credentials-stdin");
        }

        const std::string model_path = argv[1];
        bool have_bind = false;
        bool have_next = false;
        bool have_result = false;
        bool have_rank = false;
        bool have_credentials_stdin = false;
        std::string bind_endpoint;
        std::string next_endpoint;
        std::string result_endpoint;
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--credentials-stdin") {
                if (have_credentials_stdin) {
                    fail("duplicate --credentials-stdin");
                }
                have_credentials_stdin = true;
                continue;
            }
            if (i + 1 >= argc) {
                fail("missing value for " + arg);
            }
            const std::string value = argv[++i];
            if (arg == "--bind") {
                bind_endpoint = value;
                have_bind = true;
            } else if (arg == "--next") {
                next_endpoint = value;
                have_next = true;
            } else if (arg == "--result") {
                result_endpoint = value;
                have_result = true;
            } else if (arg == "--rank") {
                if (!parse_u32(value, launch_rank)) {
                    fail("invalid --rank value: " + value);
                }
                have_rank = true;
            } else {
                fail("unknown controller argument: " + arg);
            }
        }
        if (!have_bind || bind_endpoint.empty() || !have_next || next_endpoint.empty() ||
            !have_result || result_endpoint.empty() || !have_rank || !have_credentials_stdin) {
            fail("controller endpoints, rank, and CURVE bootstrap are required");
        }

        std::string error;
        if (!read_curve_bootstrap(curve_credentials, error)) {
            fail("security bootstrap failed: " + error);
        }
        have_curve_credentials = true;
        if (curve_credentials.rank != launch_rank) {
            fail("security bootstrap rank mismatch");
        }
        potluck::accel_profile profile;
        profile.rank = launch_rank;
        llama_backend_init();
        backend_initialized = true;
        if (!probe_local_accel(profile, error)) {
            fail("cannot probe accelerator: " + error);
        }

        potluck::curve_public_key_list allowed_clients = {
            curve_credentials.previous_server_key,
            curve_credentials.controller_public_key
        };
        peer = potluck::ring_peer::bind(
            bind_endpoint, curve_credentials.keypair, next_endpoint,
            curve_credentials.next_server_key, allowed_clients, error);
        if (!peer.valid()) {
            fail("cannot bind ring receiver or connect next peer: " + error);
        }
        std::printf("WORKER rank %u bound %s next %s\n", launch_rank,
                    peer.endpoint().c_str(), peer.next_endpoint().c_str());
        std::fflush(stdout);
        potluck::curve_client_credentials result_credentials;
        result_credentials.keypair = curve_credentials.keypair;
        result_credentials.server_public_key = curve_credentials.result_server_key;
        result_sender = potluck::ring_sender::connect(
            result_endpoint, result_credentials, error);
        potluck::scrub_curve_keypair(result_credentials.keypair);
        result_credentials.server_public_key.clear();
        if (!result_sender.valid()) {
            fail("cannot connect result sender: " + error);
        }

        constexpr int startup_timeout_ms = 120000;
        constexpr int decode_timeout_ms = 600000;
        if (!peer.set_timeouts(startup_timeout_ms, startup_timeout_ms, error)) {
            fail("cannot set ring startup timeouts: " + error);
        }
        if (!result_sender.set_send_timeout(startup_timeout_ms, error)) {
            fail("cannot set result startup timeout: " + error);
        }
        if (curve_credentials.peer_count == 0 ||
            launch_rank >= curve_credentials.peer_count ||
            curve_credentials.generation == 0) {
            fail("security bootstrap topology metadata is invalid");
        }
        potluck::message adjacency_hello;
        adjacency_hello.type = potluck::message_type::hello;
        adjacency_hello.rank = launch_rank;
        adjacency_hello.sequence = curve_credentials.generation;
        if (!peer.send(adjacency_hello, error)) {
            fail("security adjacency handshake send failed: " + error);
        }
        potluck::message predecessor_hello;
        if (!peer.receive(predecessor_hello, error)) {
            fail("security adjacency handshake receive failed: " + error);
        }
        const uint32_t expected_predecessor =
            (launch_rank + curve_credentials.peer_count - 1) %
            curve_credentials.peer_count;
        if (predecessor_hello.type != potluck::message_type::hello ||
            predecessor_hello.rank != expected_predecessor ||
            predecessor_hello.sequence != curve_credentials.generation) {
            fail("security adjacency handshake predecessor mismatch");
        }

        std::vector<uint8_t> profile_payload;
        if (!potluck::encode_accel_profile(profile, profile_payload)) {
            fail("cannot encode accelerator profile");
        }
        potluck::message profile_message;
        profile_message.type = potluck::message_type::profile_result;
        profile_message.rank = launch_rank;
        profile_message.payload = std::move(profile_payload);
        if (!result_sender.send(profile_message, error)) {
            fail("cannot send accelerator profile: " + error);
        }
        std::printf("WORKER rank %u accelerator %s free %llu MiB total %llu MiB\n",
                    launch_rank, accel_kind_name(profile.kind),
                    static_cast<unsigned long long>(profile.free_bytes / (1024u * 1024u)),
                    static_cast<unsigned long long>(profile.total_bytes / (1024u * 1024u)));
        std::fflush(stdout);

        potluck::message config_message;
        if (!peer.receive(config_message, error)) {
            fail("cannot receive node_config: " + error);
        }
        if (config_message.type != potluck::message_type::node_config) {
            fail("expected node_config before ring traffic");
        }

        potluck::node_config config;
        if (!potluck::decode_config(config_message.payload.data(), config_message.payload.size(),
                                    config, error)) {
            fail("cannot decode node_config: " + error);
        }
        if (config.index != launch_rank) {
            fail("node_config rank does not match controller launch rank");
        }
        if (config.n_workers == 0 || config.index >= config.n_workers ||
            config.n_layer == 0 || config.windows.empty()) {
            fail("node_config has invalid ring dimensions");
        }
        uint32_t next_layer = 0;
        for (const potluck::ring_window & window : config.windows) {
            if (window.owner >= config.n_workers || window.start != next_layer ||
                window.start >= window.end || window.end > config.n_layer) {
                fail("node_config contains an invalid ordered window route");
            }
            next_layer = window.end;
        }
        if (next_layer != config.n_layer) {
            fail("node_config windows do not cover the model");
        }

        const uint32_t n_ctx = config.n_ctx == 0 ? 4096 : config.n_ctx;
        const uint32_t n_seq_max = config.n_seq_max == 0 ? 1 : config.n_seq_max;
        const uint32_t n_ubatch = config.n_ubatch == 0 ? 512 : config.n_ubatch;
        stages.resize(config.windows.size());
        size_t owned_windows = 0;
        for (size_t i = 0; i < config.windows.size(); ++i) {
            const potluck::ring_window & window = config.windows[i];
            if (window.owner != config.index) {
                continue;
            }
            ++owned_windows;
            const bool tail = window.end == config.n_layer;
            const std::string shard_path = model_path + ".potluck-" + std::to_string(i) +
                "of" + std::to_string(config.windows.size()) + ".gguf";
            if (!potluck::stage_load(stages[i], shard_path, window.start, window.end,
                                     /*embeddings=*/false, n_ctx, n_seq_max, n_ubatch,
                                     error, tail, window.n_gpu_layers, nullptr,
                                     /*explicit_gpu_head=*/false, /*single_thread=*/false)) {
                fail("window [" + std::to_string(window.start) + "," +
                     std::to_string(window.end) + ") shard load failed: " + error);
            }
            if (!validate_shard(stages[i], static_cast<uint32_t>(i),
                                static_cast<uint32_t>(config.windows.size()),
                                window.start, window.end, error)) {
                fail("window [" + std::to_string(window.start) + "," +
                     std::to_string(window.end) + ") " + error);
            }
        }
        const bool trace_prp = std::getenv("POTLUCK_TRACE_PRP") != nullptr;
        const bool force_prefetch = std::getenv("POTLUCK_PREFETCH_FORCE") != nullptr;
        std::vector<uint32_t> owner_round;
        if (trace_prp) {
            owner_round.resize(config.windows.size());
            std::unordered_map<uint32_t, uint32_t> rounds;
            for (size_t i = 0; i < config.windows.size(); ++i) {
                owner_round[i] = rounds[config.windows[i].owner]++;
            }
        }
        const auto trace_window_event = [&](uint64_t sequence, size_t window,
                                            const char * event, size_t bytes) {
            if (!trace_prp) {
                return;
            }
            trace_prp_event(true, config.index, sequence, static_cast<uint32_t>(window),
                            owner_round[window], event, bytes);
        };
        const auto prefetch_next_owned = [&](size_t current_window, uint64_t sequence,
                                             bool startup) {
            for (size_t offset = 1; offset <= config.windows.size(); ++offset) {
                const size_t next = (current_window + offset) % config.windows.size();
                if (config.windows[next].owner != config.index || stages[next].model == nullptr) {
                    continue;
                }
                trace_window_event(sequence, next,
                                   startup ? "startup_prefetch_begin" : "prefetch_begin", 0);
                const size_t bytes = stages[next].model->prefetch(force_prefetch);
                trace_window_event(sequence, next,
                                   startup ? "startup_prefetch_end" : "prefetch_end", bytes);
                return;
            }
        };
        prefetch_next_owned(config.windows.size() - 1, 0, true);


        std::printf("WORKER rank %u/%u loaded %zu owned windows\n",
                    config.index, config.n_workers, owned_windows);
        std::fflush(stdout);
        potluck::message ready;
        ready.type = potluck::message_type::ready;
        ready.rank = config.index;
        ready.sequence = 1;
        if (!result_sender.send(ready, error)) {
            fail("cannot report worker readiness: " + error);
        }

        constexpr int peer_receive_timeout_ms = 250;
        constexpr int peer_send_timeout_ms = 5000;
        if (!peer.set_timeouts(peer_receive_timeout_ms, peer_send_timeout_ms, error)) {
            fail("cannot set ring decode timeouts: " + error);
        }
        if (!result_sender.set_send_timeout(decode_timeout_ms, error)) {
            fail("cannot set result decode timeout: " + error);
        }
        receiver_thread = std::thread([&] {
            std::string heartbeat_error;
            potluck::curve_client_credentials heartbeat_credentials;
            heartbeat_credentials.keypair = curve_credentials.keypair;
            heartbeat_credentials.server_public_key = curve_credentials.result_server_key;
            potluck::ring_sender heartbeat_sender =
                potluck::ring_sender::connect(result_endpoint, heartbeat_credentials,
                                              heartbeat_error);
            potluck::scrub_curve_keypair(heartbeat_credentials.keypair);
            heartbeat_credentials.server_public_key.clear();
            if (!heartbeat_sender.valid() ||
                !heartbeat_sender.set_send_timeout(peer_send_timeout_ms, heartbeat_error)) {
                {
                    std::lock_guard<std::mutex> lock(inbox_mutex);
                    inbox_error = "cannot create heartbeat result sender: " + heartbeat_error;
                }
                inbox_cv.notify_one();
                return;
            }
            for (;;) {
                potluck::message received;
                std::string receive_error;
                if (!peer.receive(received, receive_error)) {
                    if (receiver_stopping.load()) {
                        break;
                    }
                    if (receive_error.find("timeout") != std::string::npos) {
                        continue;
                    }
                    {
                        std::lock_guard<std::mutex> lock(inbox_mutex);
                        inbox_error = "ring receive failed: " + receive_error;
                    }
                    inbox_cv.notify_one();
                    break;
                }
                if (received.type == potluck::message_type::heartbeat) {
                    potluck::message acknowledgement;
                    acknowledgement.type = potluck::message_type::heartbeat_ack;
                    acknowledgement.rank = config.index;
                    acknowledgement.sequence = received.sequence;
                    if (!heartbeat_sender.send(acknowledgement, receive_error)) {
                        {
                            std::lock_guard<std::mutex> lock(inbox_mutex);
                            inbox_error = "cannot acknowledge heartbeat: " + receive_error;
                        }
                        inbox_cv.notify_one();
                        break;
                    }
                    continue;
                }
                {
                    std::lock_guard<std::mutex> lock(inbox_mutex);
                    inbox.push_back(std::move(received));
                }
                inbox_cv.notify_one();
            }
        });

        for (;;) {
            potluck::message message;
            {
                std::unique_lock<std::mutex> lock(inbox_mutex);
                inbox_cv.wait(lock, [&] {
                    return !inbox.empty() || !inbox_error.empty();
                });
                if (!inbox_error.empty()) {
                    fail(inbox_error);
                }
                message = std::move(inbox.front());
                inbox.pop_front();
            }
            if (message.type == potluck::message_type::reset) {
                clear_stages(stages);
                break;
            }
            if (message.type != potluck::message_type::batch_decode &&
                message.type != potluck::message_type::batch_result &&
                message.type != potluck::message_type::slot_config) {
                fail("unsupported ring message type " +
                     std::to_string(static_cast<unsigned>(message.type)));
            }

            const uint32_t window_index = message.flags;
            if (window_index >= config.windows.size()) {
                fail("ring message has out-of-range global window index " +
                     std::to_string(window_index));
            }
            const potluck::ring_window & window = config.windows[window_index];
            if (window.owner != config.index) {
                if (!peer.send(message, error)) {
                    fail("cannot forward window " + std::to_string(window_index) +
                         " to next peer: " + error);
                }
                continue;
            }

            const bool tail = window_index + 1 == config.windows.size();
            if (message.type == potluck::message_type::slot_config) {
                if (!tail) {
                    message.flags = window_index + 1;
                    if (!peer.send(message, error)) {
                        fail("cannot forward slot config to next peer: " + error);
                    }
                    continue;
                }
                potluck::slot_config slot;
                if (!potluck::decode_slot_config(message.payload.data(), message.payload.size(),
                                                 slot, error)) {
                    fail("invalid slot config: " + error);
                }
                if (static_cast<uint32_t>(slot.seq) >= n_seq_max) {
                    fail("slot config sequence exceeds configured slots");
                }
                llama_sampler * replacement =
                    make_sampler(slot, stages[window_index].n_vocab, error);
                if (!error.empty()) {
                    fail(error);
                }
                const auto existing = samplers.find(slot.seq);
                if (existing != samplers.end()) {
                    llama_sampler_free(existing->second);
                    samplers.erase(existing);
                }
                sampler_configs[slot.seq] = slot;
                if (replacement != nullptr) {
                    samplers.emplace(slot.seq, replacement);
                }

                continue;
            }
            if (trace_prp) {
                trace_window_event(message.sequence, window_index, "receive",
                                   message.payload.size());
            }

            potluck::stage_model & stage = stages[window_index];

            if (stage.ctx == nullptr) {
                fail("owned ring window was not loaded: " + std::to_string(window_index));
            }
            if (trace_prp) {
                trace_window_event(message.sequence, window_index, "compute_begin", 0);
            }
            const bool first_window = window.start == 0;

            potluck::message output;
            output.rank = config.index;
            output.sequence = message.sequence;
            output.flags = window_index + 1;

                const bool from_head = message.type == potluck::message_type::batch_decode;
                if (from_head != first_window) {
                    fail("batch message type does not match global window position");
                }
                std::vector<int32_t> positions;
                std::vector<int32_t> sequences;
                std::vector<int32_t> tokens;
                std::vector<float> hidden;
                int32_t clear_seq = -1;
                int32_t trim_seq = -1;
                int32_t trim_to = -1;
                uint32_t n_logits = 0;
                const size_t n_embd_hint = from_head ? 0 : stage.n_embd;
                if (!potluck::decode_batch_payload(message.payload.data(), message.payload.size(),
                                                   n_embd_hint, clear_seq, trim_seq, trim_to,
                                                   n_logits, positions, sequences, tokens, hidden, error)) {
                    fail("invalid batch payload at window " + std::to_string(window_index) +
                         ": " + error);
                }
                if (n_logits > positions.size()) {
                    fail("batch logits count exceeds entries at window " +
                         std::to_string(window_index));
                }
                if (tail && n_logits > n_seq_max) {
                    fail("batch logits count exceeds configured slots at window " +
                         std::to_string(window_index));
                }
                if (positions.size() > n_ubatch) {
                    fail("batch entry count exceeds configured ubatch at window " +
                         std::to_string(window_index));
                }
                for (size_t i = 0; i < positions.size(); ++i) {
                    if (positions[i] < 0) {
                        fail("batch position is negative");
                    }
                }
                for (const int32_t sequence : sequences) {
                    if (sequence < 0 || static_cast<uint32_t>(sequence) >= n_seq_max) {
                        fail("batch sequence id is outside configured slots");
                    }
                }
                if (clear_seq >= 0 && static_cast<uint32_t>(clear_seq) >= n_seq_max) {
                    fail("batch clear sequence id is outside configured slots");
                }
                if (trim_seq >= 0 && static_cast<uint32_t>(trim_seq) >= n_seq_max) {
                    fail("batch trim sequence id is outside configured slots");
                }
                if (trim_to >= 0 && trim_seq < 0) {
                    fail("batch trim sequence id is missing");
                }
                llama_memory_t memory = llama_get_memory(stage.ctx);
                if (clear_seq == -2) {
                    llama_memory_clear(memory, true);
                } else if (clear_seq >= 0) {
                    if (!llama_memory_seq_rm(memory, clear_seq, -1, -1)) {
                        fail("batch clear sequence is unsupported");
                    }
                }
                if (trim_to >= 0) {
                    if (!llama_memory_seq_rm(memory, trim_seq, trim_to, -1)) {
                        fail("batch sequence trim is unsupported");
                    }
                }
                if (positions.empty()) {
                    output.type = potluck::message_type::batch_result;
                    output.dtype = potluck::data_type::i32;
                    if (!potluck::encode_batch_payload(
                            positions, sequences, tokens, nullptr, 0,
                            clear_seq, trim_seq, trim_to, 0, output.payload)) {
                        fail("cannot encode clear-only batch at window " +
                             std::to_string(window_index));
                    }
                    if (trace_prp) {
                        trace_window_event(message.sequence, window_index, "compute_end", 0);
                    }
                    if (tail) {
                        if (!result_sender.send(output, error)) {
                            fail("cannot send completed clear-only result to head: " + error);
                        }
                    } else if (!peer.send(output, error)) {
                        fail("cannot send clear-only window result " +
                             std::to_string(output.flags) + " to next peer: " + error);
                    }
                    if (trace_prp) {
                        trace_window_event(message.sequence, window_index, "send",
                                           output.payload.size());
                    }
                    prefetch_next_owned(window_index, message.sequence, false);

                    continue;
                }

                const uint32_t n_entries = static_cast<uint32_t>(positions.size());
                int decode_rc;
                if (from_head) {
                    decode_rc = potluck::stage_decode_tokens_batch(
                        stage, tokens.data(), positions.data(), sequences.data(), n_entries,
                        tail ? n_logits : n_entries);
                } else {
                    decode_rc = potluck::stage_decode_hidden_batch(
                        stage, hidden.data(), positions.data(), sequences.data(), n_entries, n_logits);
                }
                if (decode_rc != 0) {
                    fail("batch decode failed at window " + std::to_string(window_index));
                }

                if (tail) {
                    std::vector<int32_t> results(n_entries, 0);
                    potluck::batch_logprobs output_logprobs(n_logits);
                    bool want_logprobs = false;
                    for (uint32_t i = n_entries - n_logits; i < n_entries; ++i) {
                        const float * logits = llama_get_logits_ith(stage.ctx, static_cast<int32_t>(i));
                        if (logits == nullptr) {
                            fail("tail batch produced no logits at window " +
                                 std::to_string(window_index));
                        }
                        const auto sampler = samplers.find(sequences[i]);
                        const auto config_it = sampler_configs.find(sequences[i]);
                        const potluck::slot_config sampling =
                            config_it == sampler_configs.end()
                                ? potluck::slot_config{}
                                : config_it->second;
                        const llama_token sampled = sampler == samplers.end()
                            ? static_cast<llama_token>(potluck::argmax_token(logits, stage.n_vocab))
                            : llama_sampler_sample(sampler->second, stage.ctx,
                                                   static_cast<int32_t>(i));
                        results[i] = static_cast<int32_t>(sampled);
                        if (sampling.logprobs) {
                            want_logprobs = true;
                            output_logprobs[i - (n_entries - n_logits)] =
                                make_logprobs(stage.ctx, static_cast<int32_t>(i), logits,
                                              stage.n_vocab, sampling, sampled);
                        }
                        if (sampler != samplers.end()) {
                            llama_sampler_accept(sampler->second, sampled);
                        }
                    }
                    output.type = want_logprobs
                        ? potluck::message_type::batch_result_logprobs
                        : potluck::message_type::batch_result;
                    output.dtype = potluck::data_type::i32;
                    if (!potluck::encode_batch_payload(positions, sequences, results, nullptr, 0,
                                                       clear_seq, trim_seq, trim_to, n_logits,
                                                       output.payload)) {
                        fail("cannot encode token batch at window " + std::to_string(window_index));
                    }
                    if (want_logprobs) {
                        std::vector<uint8_t> metadata;
                        if (!potluck::encode_batch_logprobs(output_logprobs, metadata)) {
                            fail("cannot encode token logprobs at window " +
                                 std::to_string(window_index));
                        }
                        if (output.payload.size() > potluck::max_payload_bytes ||
                            metadata.size() > potluck::max_payload_bytes - output.payload.size()) {
                            fail("token logprob payload exceeds transport limit");
                        }
                        output.shape = { output.payload.size() };
                        output.payload.insert(output.payload.end(), metadata.begin(), metadata.end());
                    }
                } else {
                    output.type = potluck::message_type::batch_result;
                    const size_t hidden_width = static_cast<size_t>(stage.n_embd);
                    if (hidden_width == 0 ||
                        n_entries > potluck::max_payload_bytes /
                            (sizeof(float) * hidden_width)) {
                        fail("batch hidden output exceeds payload limit at window " +
                             std::to_string(window_index));
                    }
                    std::vector<float> output_hidden(
                        static_cast<size_t>(n_entries) * hidden_width);
                    for (uint32_t i = 0; i < n_entries; ++i) {
                        const float * hidden_row = llama_get_embeddings_ith(stage.ctx, static_cast<int32_t>(i));
                        if (hidden_row == nullptr) {
                            fail("batch window produced no embeddings at window " +
                                 std::to_string(window_index));
                        }
                        std::memcpy(output_hidden.data() + static_cast<size_t>(i) * stage.n_embd,
                                    hidden_row, sizeof(float) * stage.n_embd);
                    }
                    output.dtype = potluck::data_type::f32;
                    if (!potluck::encode_batch_payload(positions, sequences, std::vector<int32_t>{},
                                                       output_hidden.data(), stage.n_embd,
                                                       clear_seq, trim_seq, trim_to, n_logits,
                                                       output.payload)) {
                        fail("cannot encode hidden batch at window " + std::to_string(window_index));
                    }
                }
            if (trace_prp) {
                trace_window_event(message.sequence, window_index, "compute_end", 0);
            }
            if (tail) {
                if (!result_sender.send(output, error)) {
                    fail("cannot send completed result to head: " + error);
                }
            } else if (!peer.send(output, error)) {
                fail("cannot send window " + std::to_string(output.flags) +
                     " to next peer: " + error);
            }
            if (trace_prp) {
                trace_window_event(message.sequence, window_index, "send",
                                   output.payload.size());
            }
            prefetch_next_owned(window_index, message.sequence, false);
        }

        cleanup();
        return 0;
    } catch (const std::exception & exception) {
        if (result_sender.valid()) {
            potluck::message error_message;
            error_message.type = potluck::message_type::error;
            error_message.rank = launch_rank;
            append_error_payload(error_message, exception.what());
            std::string ignored;
            (void) result_sender.send(error_message, ignored);
        }
        std::fprintf(stderr, "worker: %s\n", exception.what());
        std::fflush(stderr);
        cleanup();
        return 1;
    }
}
