#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace potluck {

constexpr uint32_t protocol_magic = 0x50544c4b; // "PTLK"
constexpr uint16_t protocol_version = 5;
constexpr size_t max_payload_bytes = 256u * 1024u * 1024u;

enum class message_type : uint16_t {
    hello = 1,
    batch_meta = 2,
    hidden_state = 3,
    token = 4,
    reset = 5,
    node_config = 6,
    ready = 7,
    profile_result = 8,
    // §11/§12 batched decode: one round decodes many (pos, seq, token)
    // entries in a single llama_decode per stage. The head sends batch_decode
    // to stage 0 (tokens in the payload); every other stage receives
    // batch_result (hidden states for its input); the tail answers
    // batch_result with one argmax token per entry.
    batch_decode = 9,
    batch_result = 10,
    error = 255,
};

enum class data_type : uint16_t {
    none = 0,
    f32 = 1,
    f16 = 2,
    i32 = 3,
};

struct message {
    message_type type = message_type::error;
    uint32_t flags = 0;
    uint32_t rank = 0;
    uint64_t sequence = 0;
    data_type dtype = data_type::none;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> payload;
};

struct node_addr {
    std::string host;
    uint16_t port = 0;
};

// Node schedule handed to worker 0 by the coordinator and forwarded unchanged
// down the chain; each worker reads its own sliver from `bounds` by `index`.
struct node_config {
    uint32_t n_workers = 0;
    uint32_t index = 0;            // this worker's stage index in [0, n_workers)
    uint32_t n_ctx = 0;
    uint32_t n_seq_max = 1;   // context sequences (1 unless the head runs --batch N)
    uint32_t n_ubatch = 512;   // multi-token internal batches; prompts still cross once
    std::vector<uint32_t> bounds;  // size n_workers+1; worker i owns [bounds[i], bounds[i+1])
    std::vector<node_addr> workers; // the full chain, workers[0] == the far head
    bool tail = false;             // last stage: reports tokens back to the head
    node_addr head_link;           // head back-link listener (used by the tail)
    uint32_t seed = 0;             // RNG seed for the tail's sampler
    float temp  = 0.0f;            // >0 enables temperature sampling; <=0 means greedy
    float top_p = 0.0f;            // >0 && <1 narrows the distribution (with temp); otherwise disabled
    std::vector<int32_t> ngl;      // per-stage GPU offload (window-relative) layers, size n_workers; empty = use worker default
    // Piped-ring mode: when non-empty, this worker owns these windows (each a
    // half-open [start,end) layer range) instead of one contiguous slice, and
    // serves window-decode requests from the coordinator over a single channel.
    // Empty means the default static contiguous-window pipeline.
    std::vector<std::pair<uint32_t, uint32_t>> ring;
};
// Per-stage metrics returned by a live benchmark request. The coordinator
// sends an empty profile_result message; workers aggregate these records while
// keeping the decode chain alive for subsequent HTTP requests.
struct worker_bench_metrics {
    uint32_t index = 0;
    uint32_t start = 0;
    uint32_t end = 0;
    float decode_tok_s = 0.0f;
    float peak_rss_mb = 0.0f;
    uint64_t decoded_positions = 0;
};

bool encode_worker_bench_metrics(const std::vector<worker_bench_metrics> & metrics,
                                 std::vector<uint8_t> & out);
bool decode_worker_bench_metrics(const uint8_t * data, size_t size,
                                 std::vector<worker_bench_metrics> & metrics,
                                 std::string & error);


// Returns an empty vector when the message cannot be encoded safely.
std::vector<uint8_t> encode_frame(const message & message);

// The frame is self-contained: [u64 body_size][body].
// No bytes are consumed on failure; error describes the first violation.
bool decode_frame(const uint8_t * data, size_t size, message & output, std::string & error);

// Serialize a node schedule into a message payload.
bool encode_config(const node_config & config, std::vector<uint8_t> & out);
bool decode_config(const uint8_t * data, size_t size, node_config & config, std::string & error);

// §11/§12 batched-decode payload helpers. A batch message carries, per entry,
// a position and a sequence id (dynamic batching mixes sequences; speculative
// decoding uses one sequence with consecutive positions), followed by the
// data array: tokens (stage-0 request and tail reply) or hidden states
// (n_entries * n_embd floats, forwarded between stages). Either `tokens` or
// `hidden` must be non-empty. `clear` != 0 tells each stage to reset its KV
// (llama_kv_cache_clear) before decoding: speculative decoding re-prefills the
// committed prefix from scratch every round so it is correct even for
// `n_logits` (0..n) is how many TRAILING entries request logits; only those
// rows count against n_outputs_max. A batched prefill needs just the last
// entry (whose argmax is discarded anyway); dynamic-batching rounds need
// every entry (each is a distinct sequence's next-token prediction).
// `clear` and `trim_to` are the requested KV reset/trim (`trim_to < 0` when
// it is not needed (dynamic batching only ever advances positions).
bool encode_batch_payload(const std::vector<int32_t> & pos,
                          const std::vector<int32_t> & seq,
                          const std::vector<int32_t> & tokens,
                          const float * hidden, size_t n_embd,
                          int32_t clear, int32_t trim_to, uint32_t n_logits,
                          std::vector<uint8_t> & out);

// Decode a batch payload into tokens (n_embd == 0) or hidden states
// (n_embd == this stage's embedding width). Returns false on any mismatch.
bool decode_batch_payload(const uint8_t * data, size_t size, size_t n_embd,
                          int32_t & clear, int32_t & trim_to, uint32_t & n_logits,
                          std::vector<int32_t> & pos,
                          std::vector<int32_t> & seq,
                          std::vector<int32_t> & tokens,
                          std::vector<float> & hidden,
                          std::string & error);

} // namespace potluck
