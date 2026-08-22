// potluck-worker: controller-private direct ZeroMQ ring peer.
//
// The controller supplies the model shard, endpoints, and rank. The worker
// receives one complete ordered window route and executes only its windows.
// Activation data moves from one physical peer to the next; completed results
// return to the head through the result sender.

#include "potluck-transport.h"
#include "gguf.h"
#include "potluck_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
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
            return "accelerator";
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

bool validate_shard(const std::string & path, uint32_t start, uint32_t end, std::string & error) {
    ggml_context * meta = nullptr;
    gguf_init_params params = { true, &meta };
    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (ctx == nullptr) {
        error = "cannot read GGUF metadata from " + path;
        return false;
    }
    const int count_id = gguf_find_key(ctx, "potluck.shard.count");
    if (count_id < 0) {
        gguf_free(ctx);
        if (meta != nullptr) {
            ggml_free(meta);
        }
        return true;
    }
    const int shard_start_id = gguf_find_key(ctx, "potluck.shard.start");
    const int shard_end_id = gguf_find_key(ctx, "potluck.shard.end");
    if (shard_start_id < 0 || shard_end_id < 0 ||
        gguf_get_kv_type(ctx, count_id) != GGUF_TYPE_UINT32 ||
        gguf_get_kv_type(ctx, shard_start_id) != GGUF_TYPE_UINT32 ||
        gguf_get_kv_type(ctx, shard_end_id) != GGUF_TYPE_UINT32) {
        error = "invalid potluck shard metadata in " + path;
        gguf_free(ctx);
        if (meta != nullptr) {
            ggml_free(meta);
        }
        return false;
    }
    const uint32_t shard_start = gguf_get_val_u32(ctx, shard_start_id);
    const uint32_t shard_end = gguf_get_val_u32(ctx, shard_end_id);
    const bool inside = start >= shard_start && end <= shard_end && start < end;
    if (!inside) {
        error = "assigned layers [" + std::to_string(start) + "," + std::to_string(end) +
                ") but shard file " + path + " holds [" + std::to_string(shard_start) +
                "," + std::to_string(shard_end) + ")";
    }
    gguf_free(ctx);
    if (meta != nullptr) {
        ggml_free(meta);
    }
    return inside;
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
    llama_sampler * sampler = nullptr;
    bool backend_initialized = false;
    uint32_t launch_rank = 0;
    potluck::ring_peer peer;
    potluck::ring_sender result_sender;

    auto cleanup = [&]() noexcept {
        if (sampler != nullptr) {
            llama_sampler_free(sampler);
            sampler = nullptr;
        }
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

        llama_backend_init();
        backend_initialized = true;

        std::string error;
        peer = potluck::ring_peer::bind(bind_endpoint, next_endpoint, error);
        if (!peer.valid()) {
            fail("cannot bind ring receiver or connect next peer: " + error);
        }
        result_sender = potluck::ring_sender::connect(result_endpoint, error);
        if (!result_sender.valid()) {
            fail("cannot connect result sender: " + error);
        }

        constexpr int startup_timeout_ms = 120000;
        constexpr int decode_timeout_ms = 300000;
        if (!peer.set_timeouts(startup_timeout_ms, startup_timeout_ms, error)) {
            fail("cannot set ring startup timeouts: " + error);
        }
        if (!result_sender.set_send_timeout(startup_timeout_ms, error)) {
            fail("cannot set result startup timeout: " + error);
        }

        // Report this device's accelerator so the head can plan layer placement.
        potluck::accel_profile profile;
        profile.rank = launch_rank;
        const enum ggml_backend_dev_type wanted[] = {
            GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_ACCEL
        };
        for (size_t ti = 0; ti < 2 && profile.kind == potluck::accel_kind::none; ++ti) {
            for (size_t ri = 0; ri < ggml_backend_reg_count(); ++ri) {
                ggml_backend_reg_t reg = ggml_backend_reg_get(ri);
                for (size_t di = 0; di < ggml_backend_reg_dev_count(reg); ++di) {
                    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, di);
                    if (ggml_backend_dev_type(dev) != wanted[ti]) {
                        continue;
                    }
                    size_t free_bytes = 0;
                    size_t total_bytes = 0;
                    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
                    if (total_bytes == 0) {
                        continue;
                    }
                    profile.free_bytes = free_bytes;
                    profile.total_bytes = total_bytes;
                    const std::string name = ggml_backend_dev_name(dev);
                    const std::string description = ggml_backend_dev_description(dev);
                    if (name.rfind("MTL", 0) == 0 || description.find("Metal") != std::string::npos) {
                        profile.kind = potluck::accel_kind::metal;
                    } else if (name.rfind("CUDA", 0) == 0 || name.rfind("ROCm", 0) == 0 ||
                               name.rfind("MUSA", 0) == 0 ||
                               description.find("CUDA") != std::string::npos) {
                        profile.kind = potluck::accel_kind::cuda;
                    } else {
                        profile.kind = potluck::accel_kind::other;
                    }
                    if (profile.kind != potluck::accel_kind::none) {
                        break;
                    }
                }
            }
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
        std::printf("WORKER rank %u accelerator %s free %zu MiB total %zu MiB\n",
                    launch_rank, accel_kind_name(profile.kind),
                    profile.free_bytes / (1024u * 1024u), profile.total_bytes / (1024u * 1024u));
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
            if (!validate_shard(model_path, window.start, window.end, error)) {
                fail(error);
            }
            const bool tail = window.end == config.n_layer;
            if (!potluck::stage_load(stages[i], model_path, window.start, window.end,
                                     /*embeddings=*/false, n_ctx, n_seq_max, n_ubatch,
                                     error, tail, window.n_gpu_layers, nullptr,
                                     /*explicit_gpu_head=*/false, /*single_thread=*/false)) {
                fail("window [" + std::to_string(window.start) + "," +
                     std::to_string(window.end) + ") load failed: " + error);
            }
        }

        const size_t tail_index = config.windows.size() - 1;
        if (config.windows[tail_index].owner == config.index) {
            llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
            sampler = llama_sampler_chain_init(sampler_params);
            if (sampler == nullptr) {
                fail("cannot create tail sampler");
            }
            if (config.temp > 0.0f) {
                llama_sampler_chain_add(sampler, llama_sampler_init_temp(config.temp));
                if (config.top_p > 0.0f && config.top_p < 1.0f) {
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(config.top_p, 1));
                }
                llama_sampler_chain_add(sampler, llama_sampler_init_dist(config.seed));
            } else {
                llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
            }
        }

        std::printf("WORKER rank %u/%u loaded %zu owned windows\n",
                    config.index, config.n_workers, owned_windows);
        std::fflush(stdout);

        potluck::message ready;
        ready.type = potluck::message_type::ready;
        ready.rank = config.index;
        ready.sequence = 1;
        if (!result_sender.send(ready, error)) {
            fail("cannot report worker ready: " + error);
        }
        if (!peer.set_timeouts(decode_timeout_ms, decode_timeout_ms, error)) {
            fail("cannot set ring decode timeouts: " + error);
        }
        if (!result_sender.set_send_timeout(decode_timeout_ms, error)) {
            fail("cannot set result decode timeout: " + error);
        }

        for (;;) {
            potluck::message message;
            if (!peer.receive(message, error)) {
                fail("ring receive failed: " + error);
            }
            if (message.type == potluck::message_type::reset) {
                clear_stages(stages);
                break;
            }
            if (message.type != potluck::message_type::batch_decode &&
                message.type != potluck::message_type::batch_result &&
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

            potluck::stage_model & stage = stages[window_index];
            if (stage.ctx == nullptr) {
                fail("owned ring window was not loaded: " + std::to_string(window_index));
            }
            const bool first_window = window.start == 0;
            const bool tail = window_index + 1 == config.windows.size();
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
                if ((!positions.empty() &&
                     (sequences.size() != positions.size() ||
                      (from_head ? tokens.size() != positions.size()
                                 : hidden.size() != positions.size() * stage.n_embd))) ||
                    (positions.empty() && clear_seq == -1)) {
                    fail("invalid batch entry count at window " + std::to_string(window_index));
                }
                if (n_logits > positions.size()) {
                    fail("batch logits count exceeds entries at window " +
                         std::to_string(window_index));
                }
                llama_memory_t memory = llama_get_memory(stage.ctx);
                if (clear_seq == -2) {
                    llama_memory_clear(memory, true);
                } else if (clear_seq >= 0) {
                    (void) llama_memory_seq_rm(memory, clear_seq, -1, -1);
                }
                if (trim_to >= 0) {
                    (void) llama_memory_seq_rm(memory, trim_seq, trim_to, -1);
                }

                if (positions.empty()) {
                    output.type = potluck::message_type::batch_result;
                    output.dtype = potluck::data_type::i32;
                    if (!potluck::encode_batch_payload({}, {}, {}, nullptr, 0,
                                                       clear_seq, trim_seq, trim_to, 0, output.payload)) {
                        fail("cannot encode control batch at window " + std::to_string(window_index));
                    }
                } else {
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
                    if (sampler == nullptr) {
                        fail("tail window has no sampler");
                    }
                    std::vector<int32_t> results(n_entries, 0);
                    for (uint32_t i = n_entries - n_logits; i < n_entries; ++i) {
                        const float * logits = llama_get_logits_ith(stage.ctx, static_cast<int32_t>(i));
                        if (logits == nullptr) {
                            fail("tail batch produced no logits at window " +
                                 std::to_string(window_index));
                        }
                        results[i] = potluck::argmax_token(logits, stage.n_vocab);
                    }
                    output.dtype = potluck::data_type::i32;
                    if (!potluck::encode_batch_payload(positions, sequences, results, nullptr, 0,
                                                       clear_seq, trim_seq, trim_to, n_logits, output.payload)) {
                        fail("cannot encode token batch at window " + std::to_string(window_index));
                    }
                } else {
                    std::vector<float> output_hidden(n_entries * stage.n_embd);
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
                                                       clear_seq, trim_seq, trim_to, n_logits, output.payload)) {
                        fail("cannot encode hidden batch at window " + std::to_string(window_index));
                    }
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
                    if (sampler == nullptr) {
                        fail("tail window has no sampler");
                    }
                    const uint32_t token = static_cast<uint32_t>(
                        llama_sampler_sample(sampler, stage.ctx, -1));
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
