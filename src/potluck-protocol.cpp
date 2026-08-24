#include "potluck-protocol.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
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
    constexpr uint32_t config_version = 3;
    if (config.n_workers == 0 || config.index >= config.n_workers ||
        config.n_layer == 0 || config.windows.empty() ||
        config.windows.size() > config.n_layer ||
        static_cast<uint32_t>(config.prefetch) >
            static_cast<uint32_t>(prefetch_mode::force) ||
        config.n_cycles == 0) {
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
    append_u32(out, config.n_rs_seq);
    append_u32(out, config.seed);
    append_f32(out, config.temp);
    append_f32(out, config.top_p);
    append_u32(out, static_cast<uint32_t>(config.prefetch));
    append_u32(out, config.n_cycles);
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
    constexpr uint32_t config_version = 3;
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
    uint32_t raw_prefetch = 0;
    if (!read_u32(data, size, offset, config.n_workers, error) ||
        !read_u32(data, size, offset, config.index, error) ||
        !read_u32(data, size, offset, config.n_layer, error) ||
        !read_u32(data, size, offset, config.n_ctx, error) ||
        !read_u32(data, size, offset, config.n_seq_max, error) ||
        !read_u32(data, size, offset, config.n_ubatch, error) ||
        !read_u32(data, size, offset, config.n_rs_seq, error) ||
        !read_u32(data, size, offset, config.seed, error) ||
        !read_f32(data, size, offset, config.temp, error) ||
        !read_f32(data, size, offset, config.top_p, error) ||
        !read_u32(data, size, offset, raw_prefetch, error) ||
        !read_u32(data, size, offset, config.n_cycles, error)) {
        return false;
    }
    if (config.n_workers == 0 || config.index >= config.n_workers || config.n_layer == 0 ||
        raw_prefetch > static_cast<uint32_t>(prefetch_mode::force) ||
        config.n_cycles == 0) {
        error = "invalid ring config dimensions";
        return false;
    }
    config.prefetch = static_cast<prefetch_mode>(raw_prefetch);

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

bool encode_device_profile(const device_profile & profile, std::vector<uint8_t> & out) {
    constexpr uint32_t profile_magic = 0x33504445; // "EDP3"
    constexpr uint32_t max_profile_vector_count = 4096;
    constexpr size_t profile_fixed_bytes = 70;
    const auto valid_vector = [](const std::vector<float> & values) {
        if (values.size() > max_profile_vector_count) {
            return false;
        }
        for (const float value : values) {
            if (!std::isfinite(value) || value < 0.0f) {
                return false;
            }
        }
        return true;
    };
    if (static_cast<uint32_t>(profile.kind) > static_cast<uint32_t>(accel_kind::other) ||
        static_cast<uint32_t>(profile.os) > static_cast<uint32_t>(os_kind::linux_os) ||
        !valid_vector(profile.cpu_gflops) || !valid_vector(profile.accel_gflops) ||
        !std::isfinite(profile.mem_copy_delay_ms) || profile.mem_copy_delay_ms < 0.0f ||
        !std::isfinite(profile.accel_copy_delay_ms) || profile.accel_copy_delay_ms < 0.0f ||
        !std::isfinite(profile.disk_read_seq_gbps) || profile.disk_read_seq_gbps < 0.0f ||
        !std::isfinite(profile.disk_read_rnd_gbps) || profile.disk_read_rnd_gbps < 0.0f) {
        return false;
    }
    const size_t vector_bytes =
        (profile.cpu_gflops.size() + profile.accel_gflops.size()) * sizeof(float);
    const size_t bytes = profile_fixed_bytes + vector_bytes;
    if (bytes > max_payload_bytes) {
        return false;
    }
    out.clear();
    out.reserve(bytes);
    append_u32(out, profile_magic);
    append_u32(out, profile.rank);
    out.push_back(static_cast<uint8_t>(profile.kind));
    out.push_back(static_cast<uint8_t>(profile.os));
    append_u64(out, profile.free_bytes);
    append_u64(out, profile.total_bytes);
    append_u64(out, profile.host_free_bytes);
    append_u64(out, profile.host_total_bytes);
    append_u32(out, static_cast<uint32_t>(profile.cpu_gflops.size()));
    for (const float value : profile.cpu_gflops) {
        append_f32(out, value);
    }
    append_u32(out, static_cast<uint32_t>(profile.accel_gflops.size()));
    for (const float value : profile.accel_gflops) {
        append_f32(out, value);
    }
    append_f32(out, profile.mem_copy_delay_ms);
    append_f32(out, profile.accel_copy_delay_ms);
    append_f32(out, profile.disk_read_seq_gbps);
    append_f32(out, profile.disk_read_rnd_gbps);
    append_u32(out, profile.n_cpu_threads);
    return true;
}

bool decode_device_profile(const uint8_t * data, size_t size, device_profile & profile,
                           std::string & error) {
    constexpr uint32_t profile_magic = 0x33504445; // "EDP3"
    constexpr uint32_t max_profile_vector_count = 4096;
    constexpr size_t profile_fixed_bytes = 70;
    profile = device_profile{};
    error.clear();
    if (size > max_payload_bytes) {
        error = "device profile size mismatch";
        return false;
    }
    size_t offset = 0;
    uint32_t magic = 0;
    if (!read_u32(data, size, offset, magic, error)) {
        return false;
    }
    if (magic != profile_magic) {
        error = "invalid device profile magic";
        return false;
    }
    if (size < profile_fixed_bytes) {
        error = "device profile size mismatch";
        return false;
    }
    if (!read_u32(data, size, offset, profile.rank, error)) {
        return false;
    }
    if (data == nullptr || offset + 2 > size) {
        error = "truncated device profile kinds";
        return false;
    }
    const uint32_t raw_kind = data[offset++];
    const uint32_t raw_os = data[offset++];
    if (raw_kind > static_cast<uint32_t>(accel_kind::other)) {
        error = "unknown accelerator kind";
        return false;
    }
    if (raw_os > static_cast<uint32_t>(os_kind::linux_os)) {
        error = "unknown operating system kind";
        return false;
    }
    profile.kind = static_cast<accel_kind>(raw_kind);
    profile.os = static_cast<os_kind>(raw_os);
    if (!read_u64(data, size, offset, profile.free_bytes, error) ||
        !read_u64(data, size, offset, profile.total_bytes, error) ||
        !read_u64(data, size, offset, profile.host_free_bytes, error) ||
        !read_u64(data, size, offset, profile.host_total_bytes, error)) {
        return false;
    }
    const auto read_vector = [&](std::vector<float> & values) {
        uint32_t count = 0;
        if (!read_u32(data, size, offset, count, error)) {
            return false;
        }
        if (count > max_profile_vector_count || offset > size ||
            count > (size - offset) / sizeof(float)) {
            error = "invalid device profile vector count";
            return false;
        }
        values.resize(count);
        for (float & value : values) {
            if (!read_f32(data, size, offset, value, error)) {
                return false;
            }
            if (!std::isfinite(value) || value < 0.0f) {
                error = "invalid device profile vector value";
                return false;
            }
        }
        return true;
    };
    if (!read_vector(profile.cpu_gflops) || !read_vector(profile.accel_gflops) ||
        !read_f32(data, size, offset, profile.mem_copy_delay_ms, error) ||
        !read_f32(data, size, offset, profile.accel_copy_delay_ms, error) ||
        !read_f32(data, size, offset, profile.disk_read_seq_gbps, error) ||
        !read_f32(data, size, offset, profile.disk_read_rnd_gbps, error) ||
        !read_u32(data, size, offset, profile.n_cpu_threads, error)) {
        return false;
    }
    if (!std::isfinite(profile.mem_copy_delay_ms) || profile.mem_copy_delay_ms < 0.0f ||
        !std::isfinite(profile.accel_copy_delay_ms) || profile.accel_copy_delay_ms < 0.0f ||
        !std::isfinite(profile.disk_read_seq_gbps) || profile.disk_read_seq_gbps < 0.0f ||
        !std::isfinite(profile.disk_read_rnd_gbps) || profile.disk_read_rnd_gbps < 0.0f) {
        error = "invalid device profile metrics";
        return false;
    }
    if (offset != size) {
        error = "device profile size mismatch";
        return false;
    }
    return true;
}
bool encode_slot_config(const slot_config & config, std::vector<uint8_t> & out) {
    constexpr uint32_t slot_magic = 0x31544c53; // "SLT1"
    if (config.seq < 0 || !std::isfinite(config.temp) || config.temp < 0.0f ||
        !std::isfinite(config.top_p) || config.top_p < 0.0f || config.top_p > 1.0f ||
        !std::isfinite(config.min_p) || config.min_p < 0.0f || config.min_p > 1.0f ||
        !std::isfinite(config.presence_penalty) ||
        !std::isfinite(config.frequency_penalty) ||
        !std::isfinite(config.repeat_penalty) || config.repeat_penalty <= 0.0f ||
        config.penalty_last_n < -1 ||
        (config.top_logprobs != 0 && !config.logprobs)) {
        return false;
    }
    out.clear();
    out.reserve(52);
    append_u32(out, slot_magic);
    append_u32(out, static_cast<uint32_t>(config.seq));
    append_f32(out, config.temp);
    append_f32(out, config.top_p);
    append_u32(out, config.top_k);
    append_u32(out, config.seed);
    append_f32(out, config.min_p);
    append_f32(out, config.presence_penalty);
    append_f32(out, config.frequency_penalty);
    append_f32(out, config.repeat_penalty);
    append_u32(out, static_cast<uint32_t>(config.penalty_last_n));
    append_u32(out, config.logprobs ? 1u : 0u);
    append_u32(out, config.top_logprobs);
    return true;
}

bool decode_slot_config(const uint8_t * data, size_t size, slot_config & config,
                        std::string & error) {
    constexpr uint32_t slot_magic = 0x31544c53; // "SLT1"
    constexpr size_t slot_bytes = 52;
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
    uint32_t raw_penalty_last_n = 0;
    uint32_t raw_logprobs = 0;
    if (config.seq < 0 ||
        !read_f32(data, size, offset, config.temp, error) ||
        !read_f32(data, size, offset, config.top_p, error) ||
        !read_u32(data, size, offset, config.top_k, error) ||
        !read_u32(data, size, offset, config.seed, error) ||
        !read_f32(data, size, offset, config.min_p, error) ||
        !read_f32(data, size, offset, config.presence_penalty, error) ||
        !read_f32(data, size, offset, config.frequency_penalty, error) ||
        !read_f32(data, size, offset, config.repeat_penalty, error) ||
        !read_u32(data, size, offset, raw_penalty_last_n, error) ||
        !read_u32(data, size, offset, raw_logprobs, error) ||
        !read_u32(data, size, offset, config.top_logprobs, error)) {
        if (config.seq < 0) {
            error = "invalid slot config sequence";
        }
        return false;
    }
    config.penalty_last_n = static_cast<int32_t>(raw_penalty_last_n);
    if (raw_logprobs > 1 ||
        !std::isfinite(config.temp) || config.temp < 0.0f ||
        !std::isfinite(config.top_p) || config.top_p < 0.0f || config.top_p > 1.0f ||
        !std::isfinite(config.min_p) || config.min_p < 0.0f || config.min_p > 1.0f ||
        !std::isfinite(config.presence_penalty) ||
        !std::isfinite(config.frequency_penalty) ||
        !std::isfinite(config.repeat_penalty) || config.repeat_penalty <= 0.0f ||
        config.penalty_last_n < -1 ||
        (raw_logprobs == 0 && config.top_logprobs != 0)) {
        error = "invalid slot sampler parameters";
        return false;
    }
    config.logprobs = raw_logprobs != 0;
    if (offset != size) {
        error = "slot config size mismatch";
        return false;
    }
    return true;
}
bool encode_batch_logprobs(const batch_logprobs & values, std::vector<uint8_t> & out) {
    constexpr uint32_t logprob_magic = 0x3142504c; // "LPB1"
    if (values.size() > UINT32_MAX) {
        return false;
    }
    size_t bytes = 8;
    for (const auto & entry : values) {
        if (entry.size() > (max_payload_bytes - bytes - 4) / 8) {
            return false;
        }
        bytes += 4 + entry.size() * 8;
        for (const token_logprob & value : entry) {
            if (!std::isfinite(value.logprob)) {
                return false;
            }
        }
    }
    out.clear();
    out.reserve(bytes);
    append_u32(out, logprob_magic);
    append_u32(out, static_cast<uint32_t>(values.size()));
    for (const auto & entry : values) {
        append_u32(out, static_cast<uint32_t>(entry.size()));
        for (const token_logprob & value : entry) {
            append_u32(out, static_cast<uint32_t>(value.token));
            append_f32(out, value.logprob);
        }
    }
    return true;
}

bool decode_batch_logprobs(const uint8_t * data, size_t size, batch_logprobs & values,
                           std::string & error) {
    constexpr uint32_t logprob_magic = 0x3142504c; // "LPB1"
    values.clear();
    error.clear();
    size_t offset = 0;
    uint32_t magic = 0;
    uint32_t count = 0;
    if (!read_u32(data, size, offset, magic, error) ||
        !read_u32(data, size, offset, count, error)) {
        return false;
    }
    if (magic != logprob_magic || count > max_payload_bytes / 8) {
        error = "invalid batch logprobs header";
        return false;
    }
    values.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t n = 0;
        if (!read_u32(data, size, offset, n, error) ||
            offset > size || n > (size - offset) / 8) {
            error = "invalid batch logprobs count";
            return false;
        }
        std::vector<token_logprob> entry;
        entry.reserve(n);
        for (uint32_t j = 0; j < n; ++j) {
            uint32_t raw_token = 0;
            token_logprob value;
            if (!read_u32(data, size, offset, raw_token, error) ||
                !read_f32(data, size, offset, value.logprob, error)) {
                return false;
            }
            if (!std::isfinite(value.logprob)) {
                error = "invalid batch logprob value";
                return false;
            }
            value.token = static_cast<int32_t>(raw_token);
            entry.push_back(value);
        }
        values.push_back(std::move(entry));
    }
    if (offset != size) {
        error = "batch logprobs size mismatch";
        return false;
    }
    return true;
}

bool encode_batch_payload(const std::vector<int32_t> & pos,
                          const std::vector<int32_t> & seq,
                          const std::vector<int32_t> & tokens,
                          const std::vector<int32_t> & draft_tokens,
                          uint32_t accepted_count,
                          const float * hidden, size_t n_embd,
                          int32_t clear_seq, int32_t trim_seq, int32_t trim_to,
                          uint32_t n_logits,
                          std::vector<uint8_t> & out) {
    constexpr uint32_t batch_magic = 0x31505442; // "BTP1"
    constexpr uint32_t batch_version = 1;
    constexpr size_t batch_header_bytes = 36;
    const size_t n = pos.size();
    const size_t n_draft = draft_tokens.size();
    const bool has_tokens = !tokens.empty();
    const bool has_hidden = hidden != nullptr && n_embd > 0;
    const bool clear_only = n == 0 && seq.empty() && tokens.empty() &&
        draft_tokens.empty() && !has_hidden && accepted_count == 0 &&
        clear_seq != -1 && trim_seq == -1 && trim_to == -1 && n_logits == 0;
    if (accepted_count > n_draft ||
        (!clear_only && has_tokens == has_hidden) ||
        (!clear_only && (n == 0 || seq.size() != n ||
                         n_logits > n ||
                         (has_tokens && tokens.size() != n))) ||
        (clear_only && (clear_seq < -2 || clear_seq == -1)) ||
        clear_seq < -2 || trim_seq < -1 || trim_to < -1 ||
        (trim_to >= 0 && trim_seq < 0)) {
        return false;
    }
    if (n > UINT32_MAX || n_draft > 4096 ||
        n > (max_payload_bytes - batch_header_bytes) / 8 ||
        n_draft > (max_payload_bytes - batch_header_bytes) / sizeof(int32_t)) {
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
    const size_t base_bytes = batch_header_bytes + 8 * n + 4 * n_draft;
    if (base_bytes > max_payload_bytes ||
        n_data > (max_payload_bytes - base_bytes) / sizeof(float)) {
        return false;
    }
    const size_t payload_bytes = base_bytes + sizeof(float) * n_data;
    out.clear();
    out.reserve(payload_bytes);
    append_u32(out, batch_magic);
    append_u32(out, batch_version);
    append_u32(out, static_cast<uint32_t>(n));
    append_u32(out, static_cast<uint32_t>(clear_seq));
    append_u32(out, static_cast<uint32_t>(trim_seq));
    append_u32(out, static_cast<uint32_t>(trim_to));
    append_u32(out, n_logits);
    append_u32(out, static_cast<uint32_t>(n_draft));
    append_u32(out, accepted_count);
    for (const int32_t value : pos) {
        append_u32(out, static_cast<uint32_t>(value));
    }
    for (const int32_t value : seq) {
        append_u32(out, static_cast<uint32_t>(value));
    }
    for (const int32_t value : draft_tokens) {
        append_u32(out, static_cast<uint32_t>(value));
    }
    if (has_tokens) {
        for (const int32_t value : tokens) {
            append_u32(out, static_cast<uint32_t>(value));
        }
    } else if (has_hidden) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(hidden);
        out.insert(out.end(), bytes, bytes + n_data * sizeof(float));
    }
    return out.size() == payload_bytes;
}

bool decode_batch_payload(const uint8_t * data, size_t size, size_t n_embd,
                          int32_t & clear_seq, int32_t & trim_seq, int32_t & trim_to,
                          uint32_t & n_logits,
                          std::vector<int32_t> & pos,
                          std::vector<int32_t> & seq,
                          std::vector<int32_t> & tokens,
                          std::vector<int32_t> & draft_tokens,
                          uint32_t & accepted_count,
                          std::vector<float> & hidden,
                          std::string & error) {
    constexpr uint32_t batch_magic = 0x31505442; // "BTP1"
    constexpr uint32_t batch_version = 1;
    constexpr uint32_t max_batch_draft_count = 4096;
    constexpr size_t batch_header_bytes = 36;
    pos.clear();
    seq.clear();
    tokens.clear();
    draft_tokens.clear();
    hidden.clear();
    clear_seq = -1;
    trim_seq = -1;
    trim_to = -1;
    n_logits = 0;
    accepted_count = 0;
    error.clear();
    if (data == nullptr || size > max_payload_bytes || size < sizeof(uint32_t)) {
        error = "batch payload size mismatch";
        return false;
    }
    size_t offset = 0;
    uint32_t magic = 0;
    uint32_t version = 0;
    if (!read_u32(data, size, offset, magic, error)) {
        return false;
    }
    if (magic != batch_magic) {
        error = "invalid batch payload magic";
        return false;
    }
    if (!read_u32(data, size, offset, version, error)) {
        return false;
    }
    if (version != batch_version) {
        error = "unsupported batch payload version";
        return false;
    }
    uint32_t n = 0;
    uint32_t raw_clear_seq = 0;
    uint32_t raw_trim_seq = 0;
    uint32_t raw_trim_to = 0;
    uint32_t n_draft = 0;
    if (!read_u32(data, size, offset, n, error) ||
        !read_u32(data, size, offset, raw_clear_seq, error) ||
        !read_u32(data, size, offset, raw_trim_seq, error) ||
        !read_u32(data, size, offset, raw_trim_to, error) ||
        !read_u32(data, size, offset, n_logits, error) ||
        !read_u32(data, size, offset, n_draft, error) ||
        !read_u32(data, size, offset, accepted_count, error)) {
        return false;
    }
    clear_seq = static_cast<int32_t>(raw_clear_seq);
    trim_seq = static_cast<int32_t>(raw_trim_seq);
    trim_to = static_cast<int32_t>(raw_trim_to);
    if (clear_seq < -2 || trim_seq < -1 || trim_to < -1 ||
        (trim_to >= 0 && trim_seq < 0)) {
        error = "invalid batch clear or trim controls";
        return false;
    }
    if (n_draft > max_batch_draft_count || accepted_count > n_draft) {
        error = accepted_count > n_draft
            ? "invalid batch accepted count" : "invalid batch draft count";
        return false;
    }
    if (n == 0) {
        if (clear_seq == -1 || trim_seq != -1 || trim_to != -1 || n_logits != 0 ||
            n_draft != 0 || accepted_count != 0 || offset != size) {
            error = "invalid clear-only batch";
            return false;
        }
        return true;
    }
    if (n_logits > n || n > (max_payload_bytes - batch_header_bytes) / 8) {
        error = "invalid batch entry count";
        return false;
    }
    const size_t fixed_bytes = batch_header_bytes + 8 * static_cast<size_t>(n) +
        4 * static_cast<size_t>(n_draft);
    if (size < fixed_bytes) {
        error = "truncated batch entries";
        return false;
    }
    pos.resize(n);
    seq.resize(n);
    for (size_t i = 0; i < n; ++i) {
        uint32_t value = 0;
        if (!read_u32(data, size, offset, value, error)) {
            return false;
        }
        pos[i] = static_cast<int32_t>(value);
    }
    for (size_t i = 0; i < n; ++i) {
        uint32_t value = 0;
        if (!read_u32(data, size, offset, value, error)) {
            return false;
        }
        seq[i] = static_cast<int32_t>(value);
    }
    draft_tokens.resize(n_draft);
    for (int32_t & value : draft_tokens) {
        uint32_t raw_value = 0;
        if (!read_u32(data, size, offset, raw_value, error)) {
            return false;
        }
        value = static_cast<int32_t>(raw_value);
    }
    const size_t data_bytes = size - offset;
    size_t expected_bytes = 0;
    if (n_embd == 0) {
        if (n > max_payload_bytes / sizeof(int32_t)) {
            error = "batch token payload size mismatch";
            return false;
        }
        expected_bytes = static_cast<size_t>(n) * sizeof(int32_t);
        if (data_bytes != expected_bytes) {
            error = "batch token payload size mismatch";
            return false;
        }
        tokens.resize(n);
        for (int32_t & value : tokens) {
            uint32_t raw_value = 0;
            if (!read_u32(data, size, offset, raw_value, error)) {
                return false;
            }
            value = static_cast<int32_t>(raw_value);
        }
    } else {
        if (n_embd > max_payload_bytes / sizeof(float) ||
            n > max_payload_bytes / (sizeof(float) * n_embd)) {
            error = "batch hidden payload size mismatch";
            return false;
        }
        expected_bytes = static_cast<size_t>(n) * n_embd * sizeof(float);
        if (data_bytes != expected_bytes) {
            error = "batch hidden payload size mismatch";
            return false;
        }
        hidden.resize(static_cast<size_t>(n) * n_embd);
        std::memcpy(hidden.data(), data + offset, data_bytes);
        offset += data_bytes;
    }
    return offset == size;
}

} // namespace potluck
