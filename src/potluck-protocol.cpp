#include "potluck-protocol.h"

#include <cstring>

namespace potluck {
namespace {



void append_u32(std::vector<uint8_t> & out, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

void append_u64(std::vector<uint8_t> & out, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}


bool read_u32(const uint8_t * data, size_t size, size_t & offset, uint32_t & value, std::string & error) {
    if (data == nullptr || offset > size || size - offset < 4) {
        error = "truncated u32";
        return false;
    }
    value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= static_cast<uint32_t>(data[offset + i]) << (8 * i);
    }
    offset += 4;
    return true;
}

bool read_u64(const uint8_t * data, size_t size, size_t & offset, uint64_t & value, std::string & error) {
    if (data == nullptr || offset > size || size - offset < 8) {
        error = "truncated u64";
        return false;
    }
    value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(data[offset + i]) << (8 * i);
    }
    offset += 8;
    return true;
}





// Sampler params travel as IEEE-754 bit patterns, little-endian u32, so the
// wire format stays fixed-width across architectures.
void append_f32(std::vector<uint8_t> & out, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32(out, bits);
}

bool read_f32(const uint8_t * data, size_t size, size_t & offset, float & value, std::string & error) {
    uint32_t bits = 0;
    if (!read_u32(data, size, offset, bits, error)) {
        return false;
    }
    std::memcpy(&value, &bits, sizeof(value));
    return true;
}
} // namespace
bool encode_config(const node_config & config, std::vector<uint8_t> & out) {
    constexpr uint32_t config_version = 1;
    if (config.n_workers == 0 || config.index >= config.n_workers ||
        config.n_layer == 0 || config.windows.empty() ||
        config.windows.size() > config.n_layer) {
        return false;
    }
    uint32_t next_layer = 0;
    for (const ring_window & window : config.windows) {
        if (window.owner >= config.n_workers || window.start != next_layer ||
            window.start >= window.end || window.end > config.n_layer) {
            return false;
        }
        next_layer = window.end;
    }
    if (next_layer != config.n_layer) {
        return false;
    }

    out.clear();
    append_u32(out, config_version);
    append_u32(out, config.n_workers);
    append_u32(out, config.index);
    append_u32(out, config.n_layer);
    append_u32(out, config.n_ctx);
    append_u32(out, config.n_seq_max);
    append_u32(out, config.n_ubatch);
    append_u32(out, config.seed);
    append_f32(out, config.temp);
    append_f32(out, config.top_p);
    append_u32(out, static_cast<uint32_t>(config.windows.size()));
    for (const ring_window & window : config.windows) {
        append_u32(out, window.owner);
        append_u32(out, window.start);
        append_u32(out, window.end);
        append_u32(out, static_cast<uint32_t>(window.n_gpu_layers));
    }
    return true;
}

bool decode_config(const uint8_t * data, size_t size, node_config & config, std::string & error) {
    constexpr uint32_t config_version = 1;
    error.clear();
    config = node_config{};
    size_t offset = 0;
    uint32_t version = 0;
    if (!read_u32(data, size, offset, version, error)) {
        return false;
    }
    if (version != config_version) {
        error = "unsupported ring config version";
        return false;
    }
    if (!read_u32(data, size, offset, config.n_workers, error) ||
        !read_u32(data, size, offset, config.index, error) ||
        !read_u32(data, size, offset, config.n_layer, error) ||
        !read_u32(data, size, offset, config.n_ctx, error) ||
        !read_u32(data, size, offset, config.n_seq_max, error) ||
        !read_u32(data, size, offset, config.n_ubatch, error) ||
        !read_u32(data, size, offset, config.seed, error) ||
        !read_f32(data, size, offset, config.temp, error) ||
        !read_f32(data, size, offset, config.top_p, error)) {
        return false;
    }
    if (config.n_workers == 0 || config.index >= config.n_workers || config.n_layer == 0) {
        error = "invalid ring config dimensions";
        return false;
    }

    uint32_t n_windows = 0;
    if (!read_u32(data, size, offset, n_windows, error)) {
        return false;
    }
    if (n_windows == 0 || n_windows > config.n_layer) {
        error = "invalid ring window count";
        return false;
    }
    config.windows.resize(n_windows);
    uint32_t next_layer = 0;
    for (ring_window & window : config.windows) {
        uint32_t raw_gpu_layers = 0;
        if (!read_u32(data, size, offset, window.owner, error) ||
            !read_u32(data, size, offset, window.start, error) ||
            !read_u32(data, size, offset, window.end, error) ||
            !read_u32(data, size, offset, raw_gpu_layers, error)) {
            return false;
        }
        window.n_gpu_layers = static_cast<int32_t>(raw_gpu_layers);
        if (window.owner >= config.n_workers || window.start != next_layer ||
            window.start >= window.end || window.end > config.n_layer) {
            error = "invalid ring window";
            return false;
        }
        next_layer = window.end;
    }
    if (next_layer != config.n_layer) {
        error = "ring windows do not cover the model";
        return false;
    }
    if (offset != size) {
        error = "trailing ring config bytes";
        return false;
    }
    return true;
}

bool encode_worker_bench_metrics(const std::vector<worker_bench_metrics> & metrics,
                                 std::vector<uint8_t> & out) {
    constexpr uint32_t metrics_magic = 0x314d4245; // "EBM1"
    if (metrics.size() > (max_payload_bytes - 8) / 28) {
        return false;
    }
    out.clear();
    out.reserve(8 + metrics.size() * 28);
    append_u32(out, metrics_magic);
    append_u32(out, static_cast<uint32_t>(metrics.size()));
    for (const worker_bench_metrics & metric : metrics) {
        append_u32(out, metric.index);
        append_u32(out, metric.start);
        append_u32(out, metric.end);
        append_f32(out, metric.decode_tok_s);
        append_f32(out, metric.peak_rss_mb);
        append_u64(out, metric.decoded_positions);
    }
    return true;
}

bool decode_worker_bench_metrics(const uint8_t * data, size_t size,
                                 std::vector<worker_bench_metrics> & metrics,
                                 std::string & error) {
    constexpr uint32_t metrics_magic = 0x314d4245; // "EBM1"
    metrics.clear();
    size_t offset = 0;
    uint32_t magic = 0;
    uint32_t count = 0;
    if (!read_u32(data, size, offset, magic, error) ||
        !read_u32(data, size, offset, count, error)) {
        return false;
    }
    if (magic != metrics_magic) {
        error = "invalid worker benchmark metrics magic";
        return false;
    }
    if (count > (max_payload_bytes - 8) / 28 ||
        size != 8 + static_cast<size_t>(count) * 28) {
        error = "worker benchmark metrics size mismatch";
        return false;
    }
    metrics.resize(count);
    for (worker_bench_metrics & metric : metrics) {
        if (!read_u32(data, size, offset, metric.index, error) ||
            !read_u32(data, size, offset, metric.start, error) ||
            !read_u32(data, size, offset, metric.end, error) ||
            !read_f32(data, size, offset, metric.decode_tok_s, error) ||
            !read_f32(data, size, offset, metric.peak_rss_mb, error) ||
            !read_u64(data, size, offset, metric.decoded_positions, error)) {
            return false;
        }
    }
    return offset == size;
}

bool encode_accel_profile(const accel_profile & profile, std::vector<uint8_t> & out) {
    constexpr uint32_t profile_magic = 0x32504145; // "EAP2"
    if (static_cast<uint32_t>(profile.kind) > static_cast<uint32_t>(accel_kind::other)) {
        return false;
    }
    out.clear();
    append_u32(out, profile_magic);
    append_u32(out, profile.rank);
    out.push_back(static_cast<uint8_t>(profile.kind));
    append_u64(out, profile.free_bytes);
    append_u64(out, profile.total_bytes);
    append_u64(out, profile.host_free_bytes);
    append_u64(out, profile.host_total_bytes);
    return true;
}

bool decode_accel_profile(const uint8_t * data, size_t size, accel_profile & profile,
                          std::string & error) {
    constexpr uint32_t profile_magic = 0x32504145; // "EAP2"
    constexpr size_t profile_bytes = 41;
    profile = accel_profile{};
    error.clear();
    if (size != profile_bytes) {
        error = "accelerator profile size mismatch";
        return false;
    }
    size_t offset = 0;
    uint32_t magic = 0;
    uint32_t rank = 0;
    uint64_t free_bytes = 0;
    uint64_t total_bytes = 0;
    uint64_t host_free_bytes = 0;
    uint64_t host_total_bytes = 0;
    if (!read_u32(data, size, offset, magic, error)) {
        return false;
    }
    if (magic != profile_magic) {
        error = "invalid accelerator profile magic";
        return false;
    }
    if (!read_u32(data, size, offset, rank, error)) {
        return false;
    }
    if (data == nullptr || offset >= size) {
        error = "truncated accelerator profile kind";
        return false;
    }
    const uint8_t kind = data[offset++];
    if (kind > static_cast<uint8_t>(accel_kind::other)) {
        error = "unknown accelerator kind";
        return false;
    }
    if (!read_u64(data, size, offset, free_bytes, error) ||
        !read_u64(data, size, offset, total_bytes, error) ||
        !read_u64(data, size, offset, host_free_bytes, error) ||
        !read_u64(data, size, offset, host_total_bytes, error)) {
        return false;
    }
    if (offset != size) {
        error = "accelerator profile size mismatch";
        return false;
    }
    profile.rank = rank;
    profile.kind = static_cast<accel_kind>(kind);
    profile.free_bytes = free_bytes;
    profile.total_bytes = total_bytes;
    profile.host_free_bytes = host_free_bytes;
    profile.host_total_bytes = host_total_bytes;
    return true;
}
bool encode_slot_config(const slot_config & config, std::vector<uint8_t> & out) {
    constexpr uint32_t slot_magic = 0x31544c53; // "SLT1"
    if (config.seq < 0) {
        return false;
    }
    out.clear();
    out.reserve(24);
    append_u32(out, slot_magic);
    append_u32(out, static_cast<uint32_t>(config.seq));
    append_f32(out, config.temp);
    append_f32(out, config.top_p);
    append_u32(out, config.top_k);
    append_u32(out, config.seed);
    return true;
}

bool decode_slot_config(const uint8_t * data, size_t size, slot_config & config,
                        std::string & error) {
    constexpr uint32_t slot_magic = 0x31544c53; // "SLT1"
    constexpr size_t slot_bytes = 24;
    config = slot_config{};
    error.clear();
    if (size != slot_bytes) {
        error = "slot config size mismatch";
        return false;
    }
    size_t offset = 0;
    uint32_t magic = 0;
    uint32_t raw_seq = 0;
    if (!read_u32(data, size, offset, magic, error) ||
        !read_u32(data, size, offset, raw_seq, error)) {
        return false;
    }
    if (magic != slot_magic) {
        error = "invalid slot config magic";
        return false;
    }
    config.seq = static_cast<int32_t>(raw_seq);
    if (config.seq < 0 ||
        !read_f32(data, size, offset, config.temp, error) ||
        !read_f32(data, size, offset, config.top_p, error) ||
        !read_u32(data, size, offset, config.top_k, error) ||
        !read_u32(data, size, offset, config.seed, error)) {
        if (config.seq < 0) {
            error = "invalid slot config sequence";
        }
        return false;
    }
    if (offset != size) {
        error = "slot config size mismatch";
        return false;
    }
    return true;
}


bool encode_batch_payload(const std::vector<int32_t> & pos,
                          const std::vector<int32_t> & seq,
                          const std::vector<int32_t> & tokens,
                          const float * hidden, size_t n_embd,
                          int32_t clear_seq, int32_t trim_seq, int32_t trim_to,
                          uint32_t n_logits,
                          std::vector<uint8_t> & out) {
    const bool has_tokens = !tokens.empty();
    const bool has_hidden = hidden != nullptr && n_embd > 0;
    const bool clear_only = pos.empty() && seq.empty() && tokens.empty() &&
        !has_hidden && clear_seq != -1 && trim_seq == -1 && trim_to == -1 &&
        n_logits == 0;
    if ((!clear_only && has_tokens == has_hidden) ||
        (!clear_only && (pos.empty() || seq.size() != pos.size() ||
                         n_logits > pos.size() ||
                         (has_tokens && tokens.size() != pos.size()))) ||
        (clear_only && (clear_seq < -2 || clear_seq == -1)) ||
        clear_seq < -2 || trim_seq < -1 || trim_to < -1 ||
        (trim_to >= 0 && trim_seq < 0)) {
        return false;
    }
    const size_t n = pos.size();
    if (n > (max_payload_bytes - 20) / 8) {
        return false;
    }
    size_t n_data = 0;
    if (clear_only) {
        n_data = 0;
    } else if (has_tokens) {
        n_data = n;
    } else {
        if (n_embd > max_payload_bytes / sizeof(float) ||
            n > max_payload_bytes / (sizeof(float) * n_embd)) {
            return false;
        }
        n_data = n * n_embd;
    }
    if (n_data > (max_payload_bytes - 20 - 8 * n) / sizeof(float)) {
        return false;
    }
    out.clear();
    out.reserve(20 + 8 * n + sizeof(float) * n_data);
    append_u32(out, static_cast<uint32_t>(n));
    append_u32(out, static_cast<uint32_t>(clear_seq));
    append_u32(out, static_cast<uint32_t>(trim_seq));
    append_u32(out, static_cast<uint32_t>(trim_to));
    append_u32(out, n_logits);
    for (int32_t v : pos) {
        append_u32(out, static_cast<uint32_t>(v));
    }
    for (int32_t v : seq) {
        append_u32(out, static_cast<uint32_t>(v));
    }
    if (has_tokens) {
        for (int32_t v : tokens) {
            append_u32(out, static_cast<uint32_t>(v));
        }
    } else {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(hidden);
        out.insert(out.end(), bytes, bytes + n_data * sizeof(float));
    }
    return true;
}

bool decode_batch_payload(const uint8_t * data, size_t size, size_t n_embd,
                          int32_t & clear_seq, int32_t & trim_seq, int32_t & trim_to,
                          uint32_t & n_logits,
                          std::vector<int32_t> & pos,
                          std::vector<int32_t> & seq,
                          std::vector<int32_t> & tokens,
                          std::vector<float> & hidden,
                          std::string & error) {
    pos.clear();
    seq.clear();
    tokens.clear();
    hidden.clear();
    clear_seq = -1;
    trim_seq = -1;
    trim_to = -1;
    n_logits = 0;
    error.clear();
    if (data == nullptr || size > max_payload_bytes || size < 20) {
        error = "batch payload size mismatch";
        return false;
    }
    size_t offset = 0;
    uint32_t n = 0;
    uint32_t raw_clear_seq = 0;
    uint32_t raw_trim_seq = 0;
    uint32_t raw_trim_to = 0;
    if (!read_u32(data, size, offset, n, error)) {
        return false;
    }
    if (!read_u32(data, size, offset, raw_clear_seq, error) ||
        !read_u32(data, size, offset, raw_trim_seq, error) ||
        !read_u32(data, size, offset, raw_trim_to, error) ||
        !read_u32(data, size, offset, n_logits, error)) {
        return false;
    }
    const size_t n_entries = n;
    clear_seq = static_cast<int32_t>(raw_clear_seq);
    trim_seq = static_cast<int32_t>(raw_trim_seq);
    trim_to = static_cast<int32_t>(raw_trim_to);
    if (clear_seq < -2 || trim_seq < -1 || trim_to < -1 ||
        (trim_to >= 0 && trim_seq < 0)) {
        error = "invalid batch clear or trim controls";
        return false;
    }
    if (n == 0) {
        if (clear_seq == -1 || trim_seq != -1 || trim_to != -1 || n_logits != 0 ||
            offset != size) {
            error = "invalid clear-only batch";
            return false;
        }
        return true;
    }
    if (n_logits > n_entries || n_entries > (max_payload_bytes - 20) / 8) {
        error = "invalid batch entry count";
        return false;
    }
    const size_t fixed_bytes = 20 + 8 * n_entries;
    if (size < fixed_bytes) {
        error = "truncated batch entries";
        return false;
    }
    pos.resize(n_entries);
    seq.resize(n_entries);
    for (size_t i = 0; i < n_entries; ++i) {
        uint32_t v = 0;
        if (!read_u32(data, size, offset, v, error)) {
            return false;
        }
        pos[i] = static_cast<int32_t>(v);
    }
    for (size_t i = 0; i < n_entries; ++i) {
        uint32_t v = 0;
        if (!read_u32(data, size, offset, v, error)) {
            return false;
        }
        seq[i] = static_cast<int32_t>(v);
    }
    const size_t data_bytes = size - offset;
    size_t expected_bytes = 0;
    if (n_embd == 0) {
        if (n_entries > max_payload_bytes / sizeof(int32_t)) {
            error = "batch token payload size mismatch";
            return false;
        }
        expected_bytes = n_entries * sizeof(int32_t);
        if (data_bytes != expected_bytes) {
            error = "batch token payload size mismatch";
            return false;
        }
        tokens.resize(n_entries);
        for (size_t i = 0; i < n_entries; ++i) {
            uint32_t v = 0;
            if (!read_u32(data, size, offset, v, error)) {
                return false;
            }
            tokens[i] = static_cast<int32_t>(v);
        }
    } else {
        if (n_embd > max_payload_bytes / sizeof(float) ||
            n_entries > max_payload_bytes / (sizeof(float) * n_embd)) {
            error = "batch hidden payload size mismatch";
            return false;
        }
        expected_bytes = n_entries * n_embd * sizeof(float);
        if (data_bytes != expected_bytes) {
            error = "batch hidden payload size mismatch";
            return false;
        }
        hidden.resize(n_entries * n_embd);
        std::memcpy(hidden.data(), data + offset, data_bytes);
        offset += data_bytes;
    }
    return offset == size;
}

} // namespace potluck
