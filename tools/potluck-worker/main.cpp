// potluck-worker: controller-private direct ZeroMQ ring peer.
#include "potluck-transport.h"
#include "potluck_runtime.h"

#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"

#include <atomic>
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
#include <vector>

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
llama_sampler * make_sampler(const potluck::slot_config & config, std::string & error) {
    if (!std::isfinite(config.temp) || config.temp < 0.0f ||
        !std::isfinite(config.top_p) || config.top_p < 0.0f ||
        config.top_p > 1.0f) {
        error = "slot sampler has invalid temperature or top_p";
        return nullptr;
    }
    if (config.temp == 0.0f) {
        return nullptr;
    }
    if (config.top_k > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        error = "slot sampler top_k is too large";
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
    if (config.top_k > 0 && !add(llama_sampler_init_top_k(static_cast<int32_t>(config.top_k)))) {
        return nullptr;
    }
    if (config.top_p > 0.0f && config.top_p < 1.0f &&
        !add(llama_sampler_init_top_p(config.top_p, 1))) {
        return nullptr;
    }
    if (!add(llama_sampler_init_temp(config.temp))) {
        return nullptr;
    }
    if (!add(llama_sampler_init_dist(config.seed))) {
        return nullptr;
    }
    return chain;
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

} // namespace

int main(int argc, char ** argv) {
    std::vector<potluck::stage_model> stages;
    std::unordered_map<int32_t, llama_sampler *> samplers;
    bool backend_initialized = false;
    uint32_t launch_rank = 0;
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
        for (potluck::stage_model & stage : stages) {
            potluck::stage_free(stage);
        }
        stages.clear();
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
            fail("usage: potluck-worker <model.gguf> --bind <endpoint> --next <endpoint> --result <endpoint> --rank N");
        }

        const std::string model_path = argv[1];
        if (model_path.empty() || model_path[0] == '-') {
            fail("first argument must be a model path");
        }
        std::string bind_endpoint;
        std::string next_endpoint;
        std::string result_endpoint;
        bool have_bind = false;
        bool have_next = false;
        bool have_result = false;
        bool have_rank = false;

        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
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
            !have_result || result_endpoint.empty() || !have_rank) {
            fail("controller endpoints and rank are required");
        }

        std::string error;
        potluck::accel_profile profile;
        profile.rank = launch_rank;
        llama_backend_init();
        backend_initialized = true;
        if (!probe_local_accel(profile, error)) {
            fail("cannot probe accelerator: " + error);
        }

        peer = potluck::ring_peer::bind(bind_endpoint, next_endpoint, error);
        if (!peer.valid()) {
            fail("cannot bind ring receiver or connect next peer: " + error);
        }
        result_sender = potluck::ring_sender::connect(result_endpoint, error);
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

        std::printf("WORKER rank %u bound %s next %s\n", launch_rank,
                    peer.endpoint().c_str(), peer.next_endpoint().c_str());
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
            if (!potluck::stage_load(stages[i], model_path, window.start, window.end,
                                     /*embeddings=*/false, n_ctx, n_seq_max, n_ubatch,
                                     error, tail, window.n_gpu_layers, nullptr,
                                     /*explicit_gpu_head=*/false, /*single_thread=*/false)) {
                fail("window [" + std::to_string(window.start) + "," +
                     std::to_string(window.end) + ") load failed: " + error);
            }
        }
        const bool trace_prefetch = std::getenv("POTLUCK_TRACE_PREFETCH") != nullptr;
        std::vector<bool> traced_compute(config.windows.size(), false);
        const auto prefetch_next_owned = [&](size_t current_window) {
            for (size_t offset = 1; offset <= config.windows.size(); ++offset) {
                const size_t next = (current_window + offset) % config.windows.size();
                if (config.windows[next].owner != config.index || stages[next].model == nullptr) {
                    continue;
                }
                const size_t bytes = stages[next].model->prefetch();
                std::printf("WORKER rank %u prefetched window %zu (%zu bytes)\n",
                            config.index, next, bytes);
                std::fflush(stdout);
                return;
            }
        };
        prefetch_next_owned(config.windows.size() - 1);


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
            potluck::ring_sender heartbeat_sender =
                potluck::ring_sender::connect(result_endpoint, heartbeat_error);
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
                message.type != potluck::message_type::slot_config &&
                message.type != potluck::message_type::token &&
                message.type != potluck::message_type::hidden_state) {
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
                llama_sampler * replacement = make_sampler(slot, error);
                if (!error.empty()) {
                    fail(error);
                }
                const auto existing = samplers.find(slot.seq);
                if (existing != samplers.end()) {
                    llama_sampler_free(existing->second);
                    samplers.erase(existing);
                }
                if (replacement != nullptr) {
                    samplers.emplace(slot.seq, replacement);
                }
                continue;
            }

            potluck::stage_model & stage = stages[window_index];
            if (stage.ctx == nullptr) {
                fail("owned ring window was not loaded: " + std::to_string(window_index));
            }
            const bool first_window = window.start == 0;
            const bool batch = message.type == potluck::message_type::batch_decode ||
                               message.type == potluck::message_type::batch_result;

            potluck::message output;
            output.rank = config.index;
            output.sequence = message.sequence;
            output.flags = window_index + 1;

            if (batch) {
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
                    if (tail) {
                        if (!result_sender.send(output, error)) {
                            fail("cannot send completed clear-only result to head: " + error);
                        }
                    } else if (!peer.send(output, error)) {
                        fail("cannot send clear-only window result " +
                             std::to_string(output.flags) + " to next peer: " + error);
                    }
                    continue;
                }

                if (trace_prefetch && !traced_compute[window_index]) {
                    std::printf("WORKER rank %u computing window %u\n", config.index, window_index);
                    std::fflush(stdout);
                    traced_compute[window_index] = true;
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

                output.type = potluck::message_type::batch_result;
                if (tail) {
                    std::vector<int32_t> results(n_entries, 0);
                    for (uint32_t i = n_entries - n_logits; i < n_entries; ++i) {
                        const float * logits = llama_get_logits_ith(stage.ctx, static_cast<int32_t>(i));
                        if (logits == nullptr) {
                            fail("tail batch produced no logits at window " +
                                 std::to_string(window_index));
                        }
                        const auto sampler = samplers.find(sequences[i]);
                        results[i] = sampler == samplers.end()
                            ? potluck::argmax_token(logits, stage.n_vocab)
                            : static_cast<int32_t>(
                                  llama_sampler_sample(sampler->second, stage.ctx,
                                                       static_cast<int32_t>(i)));
                    }
                    output.dtype = potluck::data_type::i32;
                    if (!potluck::encode_batch_payload(positions, sequences, results, nullptr, 0,
                                                       clear_seq, trim_seq, trim_to, n_logits,
                                                       output.payload)) {
                        fail("cannot encode token batch at window " + std::to_string(window_index));
                    }
                } else {
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
            } else {
                if (first_window && message.type != potluck::message_type::token) {
                    fail("first ring window expected token input");
                }
                if (!first_window && message.type != potluck::message_type::hidden_state) {
                    fail("non-first ring window expected hidden input");
                }
                if (message.sequence > std::numeric_limits<uint32_t>::max()) {
                    fail("single-token position exceeds uint32 range");
                }
                const uint32_t position = static_cast<uint32_t>(message.sequence);
                const int decode_rc = first_window
                    ? [&] {
                        if (message.payload.size() != sizeof(uint32_t)) {
                            fail("token payload has invalid size");
                        }
                        uint32_t token = 0;
                        std::memcpy(&token, message.payload.data(), sizeof(token));
                        return potluck::stage_decode_token(stage, static_cast<llama_token>(token), position);
                    }()
                    : [&] {
                        if (message.payload.size() != sizeof(float) * stage.n_embd) {
                            fail("hidden payload has invalid size");
                        }
                        return potluck::stage_decode_hidden(
                            stage, reinterpret_cast<const float *>(message.payload.data()), position);
                    }();
                if (decode_rc != 0) {
                    fail("single-token decode failed at window " + std::to_string(window_index));
                }

                if (tail) {
                    const float * logits = llama_get_logits_ith(stage.ctx, -1);
                    if (logits == nullptr) {
                        fail("tail token decode produced no logits at window " +
                             std::to_string(window_index));
                    }
                    llama_sampler * sampler = nullptr;
                    if (samplers.size() == 1) {
                        sampler = samplers.begin()->second;
                    }
                    const uint32_t token = static_cast<uint32_t>(
                        sampler == nullptr
                            ? potluck::argmax_token(logits, stage.n_vocab)
                            : llama_sampler_sample(sampler, stage.ctx, -1));
                    output.type = potluck::message_type::token;
                    output.dtype = potluck::data_type::i32;
                    output.shape = {1};
                    output.payload.resize(sizeof(token));
                    std::memcpy(output.payload.data(), &token, sizeof(token));
                } else {
                    const float * hidden = llama_get_embeddings_ith(stage.ctx, -1);
                    if (hidden == nullptr) {
                        fail("window produced no hidden state at window " +
                             std::to_string(window_index));
                    }
                    output.type = potluck::message_type::hidden_state;
                    output.dtype = potluck::data_type::f32;
                    output.shape = {1, stage.n_embd};
                    output.payload.assign(reinterpret_cast<const uint8_t *>(hidden),
                                          reinterpret_cast<const uint8_t *>(hidden) +
                                          sizeof(float) * stage.n_embd);
                }
            }
            if (owned_windows > 1) {
                prefetch_next_owned(window_index);
            }

            if (tail) {
                if (!result_sender.send(output, error)) {
                    fail("cannot send completed result to head: " + error);
                }
            } else if (!peer.send(output, error)) {
                fail("cannot send window " + std::to_string(output.flags) +
                     " to next peer: " + error);
            }
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
