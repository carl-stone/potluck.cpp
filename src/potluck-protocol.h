#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace potluck {

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
    // One ring pass decodes many sequence positions.
    batch_decode = 9,
    batch_result = 10,
    slot_config = 11,
    // Control liveness uses the sequence as a nonce and never enters the data path.
    heartbeat = 12,
    heartbeat_ack = 13,
    batch_result_logprobs = 14,
    error = 255
};

enum class data_type : uint16_t {
    none = 0,
    f32 = 1,
    f16 = 2,
    i32 = 3
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

struct ring_window {
    uint32_t owner = 0;
    uint32_t start = 0;
    uint32_t end = 0;
    int32_t n_gpu_layers = -1;
};

struct node_config {
    uint32_t n_workers = 0;
    uint32_t index = 0;
    uint32_t n_layer = 0;
    uint32_t n_ctx = 0;
    uint32_t n_seq_max = 1;
    uint32_t n_ubatch = 512;
    uint32_t seed = 0;
    float temp = 0.0f;
    float top_p = 0.0f;
    std::vector<ring_window> windows;
};

// A sampled token and the log probability reported for it.
struct token_logprob {
    int32_t token = 0;
    float logprob = 0.0f;
};

using batch_logprobs = std::vector<std::vector<token_logprob>>;

// Metrics returned by a live worker benchmark.
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

// Accelerator capability a worker reports before ring configuration.
enum class accel_kind : uint8_t {
    none = 0,
    metal = 1,
    cuda = 2,
    other = 3,
};

struct accel_profile {
    uint32_t rank = 0;
    accel_kind kind = accel_kind::none;
    uint64_t free_bytes = 0;
    uint64_t total_bytes = 0;
    uint64_t host_free_bytes = 0;
    uint64_t host_total_bytes = 0;
};

bool encode_accel_profile(const accel_profile & profile, std::vector<uint8_t> & out);
bool decode_accel_profile(const uint8_t * data, size_t size, accel_profile & profile,
                          std::string & error);

struct slot_config {
    int32_t seq = 0;
    float temp = 0.0f;
    float top_p = 0.0f;
    uint32_t top_k = 0;
    uint32_t seed = 0;
    float min_p = 0.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    float repeat_penalty = 1.0f;
    int32_t penalty_last_n = 64;
    bool logprobs = false;
    uint32_t top_logprobs = 0;
};

bool encode_slot_config(const slot_config & config, std::vector<uint8_t> & out);
bool decode_slot_config(const uint8_t * data, size_t size, slot_config & config,
                        std::string & error);

bool encode_batch_logprobs(const batch_logprobs & values, std::vector<uint8_t> & out);
bool decode_batch_logprobs(const uint8_t * data, size_t size, batch_logprobs & values,
                           std::string & error);


// Serialize a node schedule into a message payload.
bool encode_config(const node_config & config, std::vector<uint8_t> & out);
bool decode_config(const uint8_t * data, size_t size, node_config & config, std::string & error);


// A batch carries positions and sequence ids plus tokens or hidden states.
// clear_seq clears one sequence (-2 clears all, -1 clears none); trim_to
// truncates trim_seq when non-negative, and n_logits selects trailing rows.
// A non-empty clear_seq batch may have zero entries for clear-only cleanup.
bool encode_batch_payload(const std::vector<int32_t> & pos,
                          const std::vector<int32_t> & seq,
                          const std::vector<int32_t> & tokens,
                          const float * hidden, size_t n_embd,
                          int32_t clear_seq, int32_t trim_seq, int32_t trim_to,
                          uint32_t n_logits,
                          std::vector<uint8_t> & out);

// Decode a batch payload into tokens (n_embd == 0) or hidden states
// (n_embd == this stage's embedding width). Returns false on any mismatch.
bool decode_batch_payload(const uint8_t * data, size_t size, size_t n_embd,
                          int32_t & clear_seq, int32_t & trim_seq, int32_t & trim_to,
                          uint32_t & n_logits,
                          std::vector<int32_t> & pos,
                          std::vector<int32_t> & seq,
                          std::vector<int32_t> & tokens,
                          std::vector<float> & hidden,
                          std::string & error);

} // namespace potluck
