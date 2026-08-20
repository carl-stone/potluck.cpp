#pragma once

// Shared stage-runtime helpers for the prima-style static pipeline.
//
// A "stage" owns a contiguous half-open range of model layers [start, end).
//   - A head stage starts at layer 0 and emits the hidden state after its last
//     layer (no LM head) when end < n_layer.
//   - A tail stage ends at layer n_layer and runs the LM head producing logits.
//   - A middle stage (future work) would consume a hidden state and emit one.
//
// This header is intentionally small and dependency-light so it can be shared
// between the prima-stage coordinator tool and the test suite.

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace prima {

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

inline llama_context_params stage_context_params(uint32_t start, uint32_t end, bool embeddings, uint32_t n_ctx) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    // A single llama_decode must accept one whole batched window (§11/§12),
    // so n_batch is sized for it; n_ubatch stays 1 so llama.cpp splits the
    // window into single-token ubatches, keeping batched numerics identical
    // to the isolated single-token path (recurrent cleanup per ubatch).
    params.n_batch = 256;
    params.n_ubatch = 1;
    // §12 dynamic batching: concurrent requests are distinct sequences in one
    // llama_decode per round, so the context must accept more than sequence 0
    // and reserve outputs to match (llama.cpp asserts n_outputs_max >=
    // n_seq_max when it reserves the output buffer).
    params.n_seq_max = 64;
    params.n_outputs_max = 64;
    params.embeddings = embeddings;
    params.prima_layer_start = start;
    params.prima_layer_end = end;
    return params;
}

inline llama_model_params stage_model_params(uint32_t start, uint32_t end) {
    llama_model_params params = llama_model_default_params();
    params.prima_layer_start = start;
    params.prima_layer_end = end;
    return params;
}

// Loads a stage model and context. n_gpu_layers is window-relative: a positive
// value offloads that many of the stage's layers to the first GPU device
// (CUDA, else Metal/ACCEL); a negative value offloads the whole window; zero
// keeps the stage on CPU. The tail stage also pulls the LM head onto the
// device. Set n_gpu_layers=0 for CPU-only behavior (the default).
inline bool stage_load(stage_model & sm, const std::string & path, uint32_t start, uint32_t end,
                       bool embeddings, uint32_t n_ctx, std::string & error,
                       bool tail = false, int32_t n_gpu_layers = 0) {
    sm.start = start;
    sm.end = end;

    llama_model_params params = llama_model_default_params();
    params.prima_layer_start = start;
    params.prima_layer_end = end;

    // Window tensors are named blk.{i}.*; offload them via the loader's
    // per-tensor buffer overrides so placement matches the stage window (the
    // public n_gpu_layers counts from layer 0, which a mid-chain worker does
    // not hold). The override list is NULL-terminated and stored by the loader
    // as a pointer, so it must outlive llama_model_load_from_file.
    std::vector<std::string> patterns;
    std::vector<llama_model_tensor_buft_override> overrides;
    std::string device_name;
    uint32_t off = 0;
    if (n_gpu_layers != 0) {
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
            if (tail) {
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
        std::fprintf(stderr, "stage: offloaded %u/%u window layers (+%u head tensors) to %s\n",
                     off, end - start, tail ? 2u : 0u, device_name.c_str());
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

    sm.ctx = llama_init_from_model(sm.model, stage_context_params(start, end, sm.compute_embeddings, n_ctx));
    if (sm.ctx == nullptr) {
        error = "failed to allocate context for stage " + std::to_string(start) + ":" + std::to_string(end);
        llama_model_free(sm.model);
        sm.model = nullptr;
        return false;
    }

    // When the stage consumes a hidden state (start > 0) the batch needs an
    // embedding buffer; otherwise it needs a token buffer.
    sm.batch = llama_batch_init(1, start > 0 ? static_cast<int32_t>(sm.n_embd) : 0, 1);
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

// §11/§12 batched decode: decodes `n` tokens in a single llama_decode call.
// Every entry carries its own position and sequence id, so one round can
// advance several independent conversations (dynamic batching) or run one
// sequence's draft span in parallel (speculative decoding). The batch is
// built fresh per call; llama_decode splits it into ubatches internally.
// On success the per-entry logits/embeddings are readable via
// llama_get_logits_ith / llama_get_embeddings_ith(ctx, i) after synchronizing.
inline int stage_decode_tokens_batch(stage_model & sm, const int32_t * tokens,
                                     const int32_t * pos, const int32_t * seq, uint32_t n) {
    llama_batch b = llama_batch_init(n, 0, 1);
    b.n_tokens = n;
    for (uint32_t i = 0; i < n; ++i) {
        b.token[i] = static_cast<llama_token>(tokens[i]);
        b.pos[i] = pos[i];
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = seq[i];
        b.logits[i] = 1;
    }
    const int rc = llama_decode(sm.ctx, b);
    llama_batch_free(b);
    if (rc == 0) {
        llama_synchronize(sm.ctx);
    }
    return rc;
}

// Same as stage_decode_tokens_batch but consumes hidden-state rows instead of
// token ids (valid for a stage that starts after layer 0). `hidden` holds
// n * sm.n_embd floats.
inline int stage_decode_hidden_batch(stage_model & sm, const float * hidden,
                                     const int32_t * pos, const int32_t * seq, uint32_t n) {
    llama_batch b = llama_batch_init(n, static_cast<int32_t>(sm.n_embd), 1);
    b.n_tokens = n;
    for (uint32_t i = 0; i < n; ++i) {
        std::memcpy(b.embd + static_cast<size_t>(i) * sm.n_embd,
                    hidden + static_cast<size_t>(i) * sm.n_embd,
                    sizeof(float) * sm.n_embd);
        b.pos[i] = pos[i];
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = seq[i];
        b.logits[i] = 1;
    }
    const int rc = llama_decode(sm.ctx, b);
    llama_batch_free(b);
    if (rc == 0) {
        llama_synchronize(sm.ctx);
    }
    return rc;
}

inline int argmax_token(const float * logits, uint32_t n_vocab) {
    return static_cast<int>(std::max_element(logits, logits + n_vocab) - logits);
}

} // namespace prima
