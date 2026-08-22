#pragma once

// Shared stage-runtime helpers for the integrated direct-peer ring.
//
// A stage owns one or more model-layer windows. Ring traffic is handled by
// direct ZeroMQ links between adjacent peers.

#include "llama.h"

#include <sys/stat.h>

 #include <algorithm>
 #include <cstdint>
 #include <cstdio>
 #include <cstring>
 #include <cstdlib>
 #include <string>
 #include <vector>

namespace potluck {

struct stage_model {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_batch batch {};
    uint32_t n_embd = 0;
    uint32_t n_vocab = 0;
    uint32_t n_layer = 0;
    uint32_t start = 0;
    uint32_t end = 0;
    bool compute_embeddings = false;
};

// Hidden-state batches use embeddings rather than tokens. For M-RoPE, llama.cpp
// expects four position sections in the embedding batch, not one position per
// token as it accepts for text batches.
inline uint32_t stage_n_pos_per_embd(const stage_model & sm) {
    const auto rope_type = llama_model_rope_type(sm.model);
    return rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1;
}

inline void stage_prepare_hidden_positions(llama_batch & batch, uint32_t n_tokens, uint32_t n_pos) {
    if (n_pos == 1) {
        return;
    }
    std::free(batch.pos);
    batch.pos = static_cast<llama_pos *>(std::malloc(sizeof(llama_pos) * n_tokens * n_pos));
}

inline void stage_set_position(llama_batch & batch, uint32_t index, llama_pos pos,
                               uint32_t n_tokens, uint32_t n_pos) {
    if (n_pos == 1) {
        batch.pos[index] = pos;
        return;
    }
    batch.pos[index] = pos;
    batch.pos[n_tokens + index] = pos;
    batch.pos[2*n_tokens + index] = pos;
    batch.pos[3*n_tokens + index] = 0;
}

// Size of the GGUF on disk, from stat(). Used where llama_model_size() would
// return 0 because the model was loaded with no_alloc (metadata only).
inline uint64_t model_file_bytes(const std::string & path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) {
        return 0;
    }
    return static_cast<uint64_t>(st.st_size);
}

inline llama_context_params stage_context_params(uint32_t start, uint32_t end, bool embeddings,
                                                 uint32_t n_ctx, uint32_t n_seq_max, uint32_t n_ubatch,
                                                 bool tail, bool single_thread = false) {
    llama_context_params params = llama_context_default_params();
    if (single_thread || std::getenv("POTLUCK_SINGLE_THREAD") != nullptr) {
        params.n_threads = 1;
        params.n_threads_batch = 1;
    }
    params.n_ctx = n_ctx;
    // Legacy node_config carries these cluster-wide values. The integrated
    // direct-peer ZeroMQ server must select and distribute equivalent bounded
    // values automatically.
    // n_ubatch = 1 preserves the numerics-parity setting used by tests.
    params.n_ubatch = std::max<uint32_t>(1, n_ubatch);
    params.n_batch = std::max<uint32_t>(256, params.n_ubatch);
    // Keep the configured number of sequences. llama.cpp treats n_ctx as the
    // total context across those sequences; forcing 64 slots silently reduced
    // the default server's per-request context to 1/64 (256 tokens at 4096).
    params.n_seq_max = std::max<uint32_t>(1, n_seq_max);
    params.n_outputs_max = embeddings ? params.n_batch : std::max<uint32_t>(1, n_seq_max);
    params.embeddings = embeddings;
    params.swa_full = n_seq_max > 1;
    params.potluck_layer_start = start;
    params.potluck_layer_end = end;
    return params;
}

inline llama_model_params stage_model_params(uint32_t start, uint32_t end) {
    llama_model_params params = llama_model_default_params();
    params.potluck_layer_start = start;
    params.potluck_layer_end = end;
    return params;
}

// Loads a stage model and context. n_gpu_layers is window-relative: a positive
// value offloads that many of the stage's layers to the first GPU device
// (CUDA, else Metal/ACCEL); a negative value offloads the whole window; zero
inline bool stage_load(stage_model & sm, const std::string & path, uint32_t start, uint32_t end,
                       bool embeddings, uint32_t n_ctx, uint32_t n_seq_max, uint32_t n_ubatch,
                       std::string & error, bool tail = false, int32_t n_gpu_layers = 0,
                       const std::vector<uint32_t> * explicit_gpu_layers = nullptr,
                       bool explicit_gpu_head = false, bool single_thread = false) {
    sm.start = start;
    sm.end = end;

    llama_model_params params = llama_model_default_params();
    // llama_model_default_params() uses -1 (all layers). Potluck's zero
    // means CPU-only, so make that explicit before applying a positive or
    // negative window-relative override below.
    params.n_gpu_layers = 0;
    // Keep CPU-only stages off the Metal backend entirely. A null device list
    // still makes llama.cpp initialize every discovered GPU even with zero
    // offloaded layers; several workers then contend during scheduler setup.
    const bool explicit_gpu = explicit_gpu_layers != nullptr;
    const bool explicit_has_gpu = explicit_gpu &&
        (!explicit_gpu_layers->empty() || explicit_gpu_head);
    std::vector<ggml_backend_dev_t> cpu_devices;
    if ((n_gpu_layers == 0 && !explicit_has_gpu) || (explicit_gpu && !explicit_has_gpu)) {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            error = "CPU backend is unavailable";
            return false;
        }
        cpu_devices = {cpu_dev, nullptr};
        params.devices = cpu_devices.data();
    }
    params.potluck_layer_start = start;
    params.potluck_layer_end = end;
    // Per-tensor buffer overrides keep a partial window's placement
    // window-relative. The explicit full-model form is used by the parity
    // reference to mirror the chain's per-worker CPU/GPU map, including a
    // CPU tail LM head when the tail worker has no GPU budget.
    std::vector<std::string> patterns;
    std::vector<llama_model_tensor_buft_override> overrides;
    std::string device_name;
    uint32_t off = 0;
    if (explicit_has_gpu) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (dev == nullptr) {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL);
        }
        if (dev == nullptr) {
            error = "GPU offload requested but no GPU device is available";
            return false;
        }
        const ggml_backend_buffer_type_t gpu_buft = ggml_backend_dev_buffer_type(dev);
        patterns.reserve(explicit_gpu_layers->size() + (explicit_gpu_head ? 2u : 0u));
        for (uint32_t layer : *explicit_gpu_layers) {
            patterns.push_back("blk\\." + std::to_string(layer) + "\\.");
        }
        if (explicit_gpu_head) {
            patterns.push_back("output_norm\\.");
            patterns.push_back("output\\.");
        }
        overrides.reserve(patterns.size() + 1);
        for (const std::string & p : patterns) {
            llama_model_tensor_buft_override o;
            o.pattern = p.c_str();
            o.buft = gpu_buft;
            overrides.push_back(o);
        }
        llama_model_tensor_buft_override term;
        term.pattern = nullptr;
        term.buft = nullptr;
        overrides.push_back(term);
        params.tensor_buft_overrides = overrides.data();
        device_name = ggml_backend_dev_name(dev);
        off = static_cast<uint32_t>(explicit_gpu_layers->size());
    } else if (n_gpu_layers != 0) {
        if (end == 0) {
            // Full-model stage: use the plain layer count. The override list
            // is only needed to match a partial window.
            params.n_gpu_layers = n_gpu_layers;
        } else {
            ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            if (dev == nullptr) {
                dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL);
            }
            if (dev == nullptr) {
                error = "GPU offload requested but no GPU device is available";
                return false;
            }
            const ggml_backend_buffer_type_t gpu_buft = ggml_backend_dev_buffer_type(dev);
            const uint32_t span = end - start;
            off = (n_gpu_layers < 0 || static_cast<uint32_t>(n_gpu_layers) >= span)
                      ? span
                      : static_cast<uint32_t>(n_gpu_layers);
            patterns.reserve(off + 2);
            for (uint32_t i = 0; i < off; ++i) {
                patterns.push_back("blk\\." + std::to_string(start + i) + "\\.");
            }
            if (tail && off == span) {
                patterns.push_back("output_norm\\.");
                patterns.push_back("output\\.");
            }
            overrides.reserve(patterns.size() + 1);
            for (const std::string & p : patterns) {
                llama_model_tensor_buft_override o;
                o.pattern = p.c_str();
                o.buft = gpu_buft;
                overrides.push_back(o);
            }
            llama_model_tensor_buft_override term;
            term.pattern = nullptr;
            term.buft = nullptr;
            overrides.push_back(term);
            params.tensor_buft_overrides = overrides.data();
            device_name = ggml_backend_dev_name(dev);
        }
    }

    sm.model = llama_model_load_from_file(path.c_str(), params);
    if (sm.model == nullptr) {
        error = "failed to load model: " + path;
        return false;
    }
    if (!device_name.empty()) {
        const bool offload_head = explicit_gpu_head || (tail && off == (end - start));
        std::fprintf(stderr, "stage: offloaded %u/%u window layers (+%u head tensors) to %s\n",
                     off, end - start, offload_head ? 2u : 0u, device_name.c_str());
    }
    sm.n_embd = static_cast<uint32_t>(llama_model_n_embd(sm.model));
    sm.n_vocab = static_cast<uint32_t>(llama_vocab_n_tokens(llama_model_get_vocab(sm.model)));
    sm.n_layer = static_cast<uint32_t>(llama_model_n_layer(sm.model));

    // A stage that does not reach the final layer must emit its hidden state to
    // the next stage. The hidden state is read back through llama_get_embeddings_ith,
    // which only returns a pointer when the context was built with embeddings
    // enabled (the embeddings buffer is shared with the logits output path).
    // The full-model/final stage does not emit, so it keeps embeddings off.
    const bool is_emitter = (end != 0 && end < sm.n_layer);
    sm.compute_embeddings = embeddings || is_emitter;

    sm.ctx = llama_init_from_model(sm.model, stage_context_params(start, end, sm.compute_embeddings,
                                                                  n_ctx, n_seq_max, n_ubatch, tail,
                                                                  single_thread));
    if (sm.ctx == nullptr) {
        error = "failed to allocate context for stage " + std::to_string(start) + ":" + std::to_string(end);
        llama_model_free(sm.model);
        sm.model = nullptr;
        return false;
    }
    // When the stage consumes a hidden state (start > 0) the batch needs an
    // embedding buffer; otherwise it needs a token buffer.
    sm.batch = llama_batch_init(1, start > 0 ? static_cast<int32_t>(sm.n_embd) : 0, 1);
    if (start > 0) {
        stage_prepare_hidden_positions(sm.batch, 1, stage_n_pos_per_embd(sm));
    }
    return true;
}


inline void stage_free(stage_model & sm) {
    if (sm.batch.n_tokens > 0 || sm.batch.token != nullptr || sm.batch.embd != nullptr) {
        llama_batch_free(sm.batch);
    }
    if (sm.ctx) llama_free(sm.ctx);
    if (sm.model) llama_model_free(sm.model);
    sm = stage_model{};
}
// Decodes a single token at the given position. On success returns 0 and leaves
// the logits available via llama_get_logits_ith(ctx, 0).
//
// NOTE: the current llama.cpp defers output synchronization, so the logits /
// embeddings buffers are only guaranteed valid after llama_synchronize(). We
// synchronize here so callers can read llama_get_logits_ith / llama_get_embeddings_ith
// immediately after this returns (matching the behavior of older llama.cpp, which
// synchronized internally).
inline int stage_decode_token(stage_model & sm, llama_token token, uint32_t pos) {
    sm.batch.n_tokens = 1;
    sm.batch.token[0] = token;
    sm.batch.pos[0] = static_cast<int32_t>(pos);
    sm.batch.n_seq_id[0] = 1;
    sm.batch.seq_id[0][0] = 0;
    sm.batch.logits[0] = 1;
    const int rc = llama_decode(sm.ctx, sm.batch);
    if (rc == 0) {
        llama_synchronize(sm.ctx);
    }
    return rc;
}

// Decodes a hidden-state vector at the given position (only valid for a stage
// that starts after layer 0).
inline int stage_decode_hidden(stage_model & sm, const float * hidden, uint32_t pos) {
    sm.batch.n_tokens = 1;
    std::memcpy(sm.batch.embd, hidden, sizeof(float) * sm.n_embd);
    stage_set_position(sm.batch, 0, static_cast<llama_pos>(pos), 1, stage_n_pos_per_embd(sm));
    sm.batch.n_seq_id[0] = 1;
    sm.batch.seq_id[0][0] = 0;
    sm.batch.logits[0] = 1;
    const int rc = llama_decode(sm.ctx, sm.batch);
    if (rc == 0) {
        llama_synchronize(sm.ctx);
    }
    return rc;
}

// §11/§12 batched decode: decodes `n` tokens in a single llama_decode call.
// Every entry carries its own position and sequence id, so one round can
// advance several independent conversations (dynamic batching) or run one
// sequence's draft span in parallel (speculative decoding). The batch is
// built fresh per call; llama_decode splits it into ubatches internally.
// On success the per-entry logits/embeddings are readable via
// llama_get_logits_ith / llama_get_embeddings_ith(ctx, i) after synchronizing.
inline int stage_decode_tokens_batch(stage_model & sm, const int32_t * tokens,
                                     const int32_t * pos, const int32_t * seq, uint32_t n,
                                     uint32_t n_logits) {
    llama_batch b = llama_batch_init(n, 0, 1);
    b.n_tokens = n;
    for (uint32_t i = 0; i < n; ++i) {
        b.token[i] = static_cast<llama_token>(tokens[i]);
        b.pos[i] = pos[i];
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = seq[i];
        // Only the trailing n_logits entries request logits: every requested
        // row counts against the context's n_outputs_max, which is n_seq_max
        // for a tail stage. A prefill therefore computes just the last entry's
        // logits; dynamic-batching rounds request all of them. An emitter
        // stage (stage 0 of a split) must instead pass n_logits = n: its
        // embeddings context requires every entry to be an output.
        b.logits[i] = (i + n_logits >= n) ? 1 : 0;
    }
    const int rc = llama_decode(sm.ctx, b);
    if (rc == 0) {
        llama_synchronize(sm.ctx);
    }
    llama_batch_free(b);
    return rc;
}

// Same as stage_decode_tokens_batch but consumes hidden-state rows instead of
// token ids (valid for a stage that starts after layer 0). `hidden` holds
// n * sm.n_embd floats.
//
// An emitter stage (compute_embeddings) is an embeddings context, and
// llama.cpp requires every batch entry to be an output in embeddings mode
// (the entries are reordered into the embeddings buffer row by row), so it
// requests logits for all entries. The windowed graph has no LM head, so no
// logits rows are computed and the output buffer holds only the embeddings
// rows (n_outputs_max = n_batch). A tail stage requests only the trailing
// n_logits entries, which its n_outputs_max = n_seq_max covers.
inline int stage_decode_hidden_batch(stage_model & sm, const float * hidden,
                                     const int32_t * pos, const int32_t * seq, uint32_t n,
                                     uint32_t n_logits) {
    llama_batch b = llama_batch_init(n, static_cast<int32_t>(sm.n_embd), 1);
    const uint32_t n_pos = stage_n_pos_per_embd(sm);
    stage_prepare_hidden_positions(b, n, n_pos);
    b.n_tokens = n;
    for (uint32_t i = 0; i < n; ++i) {
        std::memcpy(b.embd + static_cast<size_t>(i) * sm.n_embd,
                    hidden + static_cast<size_t>(i) * sm.n_embd,
                    sizeof(float) * sm.n_embd);
        stage_set_position(b, i, static_cast<llama_pos>(pos[i]), n, n_pos);
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = seq[i];
        b.logits[i] = sm.compute_embeddings || (i + n_logits >= n) ? 1 : 0;
    }
    const int rc = llama_decode(sm.ctx, b);
    if (rc == 0) {
        llama_synchronize(sm.ctx);
    }
    llama_batch_free(b);
    return rc;
}

inline int argmax_token(const float * logits, uint32_t n_vocab) {
    return static_cast<int>(std::max_element(logits, logits + n_vocab) - logits);
}

} // namespace potluck
