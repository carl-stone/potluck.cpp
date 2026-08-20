#include "prima-distributed-protocol.h"

#include <cstring>
#include <limits>

namespace prima {
namespace {

constexpr size_t frame_prefix_bytes = sizeof(uint64_t);
constexpr size_t body_header_bytes = 4 + 2 + 2 + 4 + 4 + 8 + 2 + 2 + 8;

void append_u16(std::vector<uint8_t> & out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value));
    out.push_back(static_cast<uint8_t>(value >> 8));
}

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

bool read_u16(const uint8_t * data, size_t size, size_t & offset, uint16_t & value, std::string & error) {
    if (offset > size || size - offset < 2) {
        error = "truncated u16";
        return false;
    }
    value = static_cast<uint16_t>(data[offset]) |
            static_cast<uint16_t>(data[offset + 1]) << 8;
    offset += 2;
    return true;
}

bool read_u32(const uint8_t * data, size_t size, size_t & offset, uint32_t & value, std::string & error) {
    if (offset > size || size - offset < 4) {
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
    if (offset > size || size - offset < 8) {
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

bool valid_message_type(uint16_t value) {
    return value == static_cast<uint16_t>(message_type::hello) ||
           value == static_cast<uint16_t>(message_type::batch_meta) ||
           value == static_cast<uint16_t>(message_type::hidden_state) ||
           value == static_cast<uint16_t>(message_type::token) ||
           value == static_cast<uint16_t>(message_type::reset) ||
           value == static_cast<uint16_t>(message_type::node_config) ||
           value == static_cast<uint16_t>(message_type::ready) ||
           value == static_cast<uint16_t>(message_type::profile_result) ||
           value == static_cast<uint16_t>(message_type::batch_decode) ||
           value == static_cast<uint16_t>(message_type::batch_result) ||
           value == static_cast<uint16_t>(message_type::error);
}

bool valid_data_type(uint16_t value) {
    return value <= static_cast<uint16_t>(data_type::i32);
}

} // namespace

std::vector<uint8_t> encode_frame(const message & message) {
    if (message.payload.size() > max_payload_bytes || message.shape.size() > std::numeric_limits<uint16_t>::max()) {
        return {};
    }

    const uint64_t dims_bytes = static_cast<uint64_t>(message.shape.size()) * sizeof(uint64_t);
    const uint64_t body_size = body_header_bytes + dims_bytes + message.payload.size();
    if (body_size > static_cast<uint64_t>(frame_prefix_bytes) + max_payload_bytes ||
        body_size > std::numeric_limits<size_t>::max() - frame_prefix_bytes) {
        return {};
    }

    std::vector<uint8_t> frame;
    frame.reserve(frame_prefix_bytes + static_cast<size_t>(body_size));
    append_u64(frame, body_size);
    append_u32(frame, protocol_magic);
    append_u16(frame, protocol_version);
    append_u16(frame, static_cast<uint16_t>(message.type));
    append_u32(frame, message.flags);
    append_u32(frame, message.rank);
    append_u64(frame, message.sequence);
    append_u16(frame, static_cast<uint16_t>(message.dtype));
    append_u16(frame, static_cast<uint16_t>(message.shape.size()));
    append_u64(frame, message.payload.size());
    for (uint64_t dimension : message.shape) {
        append_u64(frame, dimension);
    }
    frame.insert(frame.end(), message.payload.begin(), message.payload.end());
    return frame;
}

bool decode_frame(const uint8_t * data, size_t size, message & output, std::string & error) {
    error.clear();
    if (data == nullptr || size < frame_prefix_bytes) {
        error = "truncated frame prefix";
        return false;
    }

    size_t offset = 0;
    uint64_t body_size = 0;
    if (!read_u64(data, size, offset, body_size, error)) {
        return false;
    }
    if (body_size > frame_prefix_bytes + max_payload_bytes || body_size > size - frame_prefix_bytes) {
        error = "truncated frame body";
        return false;
    }
    if (body_size < body_header_bytes) {
        error = "truncated frame header";
        return false;
    }

    const size_t body_end = frame_prefix_bytes + static_cast<size_t>(body_size);
    if (body_end > size) {
        error = "truncated frame body";
        return false;
    }

    uint32_t magic = 0;
    uint16_t version = 0;
    uint16_t type = 0;
    uint16_t dtype = 0;
    uint16_t dimensions = 0;
    uint64_t payload_size = 0;
    if (!read_u32(data, body_end, offset, magic, error) ||
        !read_u16(data, body_end, offset, version, error) ||
        !read_u16(data, body_end, offset, type, error) ||
        !read_u32(data, body_end, offset, output.flags, error) ||
        !read_u32(data, body_end, offset, output.rank, error) ||
        !read_u64(data, body_end, offset, output.sequence, error) ||
        !read_u16(data, body_end, offset, dtype, error) ||
        !read_u16(data, body_end, offset, dimensions, error) ||
        !read_u64(data, body_end, offset, payload_size, error)) {
        return false;
    }

    if (magic != protocol_magic) {
        error = "invalid protocol magic";
        return false;
    }
    if (version != protocol_version) {
        error = "unsupported protocol version";
        return false;
    }
    if (!valid_message_type(type)) {
        error = "invalid message type";
        return false;
    }
    if (!valid_data_type(dtype)) {
        error = "invalid data type";
        return false;
    }
    if (payload_size > max_payload_bytes) {
        error = "payload exceeds limit";
        return false;
    }

    const uint64_t required = static_cast<uint64_t>(offset) +
                              static_cast<uint64_t>(dimensions) * sizeof(uint64_t) +
                              payload_size;
    if (required > body_end) {
        error = "truncated frame contents";
        return false;
    }

    output.type = static_cast<message_type>(type);
    output.dtype = static_cast<data_type>(dtype);
    output.shape.clear();
    output.shape.reserve(dimensions);
    for (uint16_t i = 0; i < dimensions; ++i) {
        uint64_t dimension = 0;
        if (!read_u64(data, body_end, offset, dimension, error)) {
            return false;
        }
        output.shape.push_back(dimension);
    }

    output.payload.assign(data + offset, data + offset + static_cast<size_t>(payload_size));
    return true;
}
// ---------------------------------------------------------------------------
// Node schedule (config) encoding. All integers are little-endian.
//   u32 n_workers, u32 index, u32 start, u32 end, u32 n_ctx
//   u8  tail
//   u8  has_head_link; if set: u16 head_port + u16 host_len + host bytes
//   u8  downstream;    if set: u16 port + u16 host_len + host bytes
// ---------------------------------------------------------------------------

static void append_str(std::vector<uint8_t> & out, const std::string & value) {
    append_u16(out, static_cast<uint16_t>(value.size()));
    out.insert(out.end(), value.begin(), value.end());
}

static bool read_str(const uint8_t * data, size_t size, size_t & offset, std::string & value, std::string & error) {
    uint16_t length = 0;
    if (!read_u16(data, size, offset, length, error)) {
        return false;
    }
    if (offset > size || size - offset < length) {
        error = "truncated config string";
        return false;
    }
    value.assign(reinterpret_cast<const char *>(data + offset), length);
    offset += length;
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
bool encode_config(const node_config & config, std::vector<uint8_t> & out) {
    if (config.bounds.size() != config.n_workers + 1 ||
        config.workers.size() != config.n_workers) {
        return false;
    }
    out.clear();
    append_u32(out, config.n_workers);
    append_u32(out, config.index);
    append_u32(out, config.n_ctx);
    out.push_back(config.tail ? 1 : 0);
    append_u16(out, config.head_link.port);
    append_str(out, config.head_link.host);
    append_u32(out, static_cast<uint32_t>(config.bounds.size()));
    for (uint32_t b : config.bounds) {
        append_u32(out, b);
    }
    for (const node_addr & addr : config.workers) {
        append_u16(out, addr.port);
        append_str(out, addr.host);
    }
    append_u32(out, config.seed);
    append_f32(out, config.temp);
    append_f32(out, config.top_p);
    if (config.ngl.empty()) {
        append_u32(out, 0);
    } else {
        if (config.ngl.size() != config.n_workers) {
            return false;
        }
        append_u32(out, static_cast<uint32_t>(config.ngl.size()));
        for (int32_t v : config.ngl) {
            append_u32(out, static_cast<uint32_t>(v));
        }
    }
    append_u32(out, static_cast<uint32_t>(config.ring.size()));
    for (const auto & w : config.ring) {
        append_u32(out, w.first);
        append_u32(out, w.second);
    }
    return true;
}

bool decode_config(const uint8_t * data, size_t size, node_config & config, std::string & error) {
    error.clear();
    config = node_config{};
    size_t offset = 0;

    if (!read_u32(data, size, offset, config.n_workers, error) ||
        !read_u32(data, size, offset, config.index, error) ||
        !read_u32(data, size, offset, config.n_ctx, error)) {
        return false;
    }

    uint8_t flag = 0;
    if (offset >= size + 1) {
        error = "truncated config tail flag";
        return false;
    }
    flag = data[offset++];
    if (flag > 1) {
        error = "invalid config tail flag";
        return false;
    }
    config.tail = flag == 1;
    if (!read_u16(data, size, offset, config.head_link.port, error) ||
        !read_str(data, size, offset, config.head_link.host, error)) {
        return false;
    }

    uint32_t n_bounds = 0;
    if (!read_u32(data, size, offset, n_bounds, error)) {
        return false;
    }
    if (n_bounds != config.n_workers + 1) {
        error = "config bounds size mismatch";
        return false;
    }
    config.bounds.resize(n_bounds);
    for (uint32_t & b : config.bounds) {
        if (!read_u32(data, size, offset, b, error)) {
            return false;
        }
    }

    config.workers.resize(config.n_workers);
    for (node_addr & addr : config.workers) {
        if (!read_u16(data, size, offset, addr.port, error) ||
            !read_str(data, size, offset, addr.host, error)) {
            return false;
        }
    }
    if (!read_u32(data, size, offset, config.seed, error) ||
        !read_f32(data, size, offset, config.temp, error) ||
        !read_f32(data, size, offset, config.top_p, error)) {
        return false;
    }
    uint32_t n_ngl = 0;
    if (!read_u32(data, size, offset, n_ngl, error)) {
        return false;
    }
    if (n_ngl != 0) {
        if (n_ngl != config.n_workers) {
            error = "config ngl count mismatch";
            return false;
        }
        config.ngl.resize(n_ngl);
        for (int32_t & v : config.ngl) {
            uint32_t raw = 0;
            if (!read_u32(data, size, offset, raw, error)) {
                return false;
            }
            v = static_cast<int32_t>(raw);
        }
    }
    uint32_t n_ring = 0;
    if (!read_u32(data, size, offset, n_ring, error)) {
        return false;
    }
    config.ring.resize(n_ring);
    for (auto & w : config.ring) {
        if (!read_u32(data, size, offset, w.first, error) ||
            !read_u32(data, size, offset, w.second, error)) {
            return false;
        }
    }
    return true;
}

bool encode_batch_payload(const std::vector<int32_t> & pos,
                          const std::vector<int32_t> & seq,
                          const std::vector<int32_t> & tokens,
                          const float * hidden, size_t n_embd,
                          int32_t clear, int32_t trim_to,
                          std::vector<uint8_t> & out) {
    const bool has_tokens = !tokens.empty();
    const bool has_hidden = hidden != nullptr && n_embd > 0;
    if (has_tokens == has_hidden) {
        return false; // exactly one data array must be provided
    }
    const size_t n = pos.size();
    if (n == 0 || seq.size() != n) {
        return false;
    }
    const size_t n_data = has_tokens ? n : n * n_embd;
    if (n_data > (max_payload_bytes / 4)) {
        return false;
    }
    out.clear();
    out.reserve(12 + 8 * n + 4 * n_data);
    append_u32(out, static_cast<uint32_t>(n));
    append_u32(out, static_cast<uint32_t>(clear));
    append_u32(out, static_cast<uint32_t>(trim_to));
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
                          int32_t & clear, int32_t & trim_to,
                          std::vector<int32_t> & pos,
                          std::vector<int32_t> & seq,
                          std::vector<int32_t> & tokens,
                          std::vector<float> & hidden,
                          std::string & error) {
    pos.clear();
    seq.clear();
    tokens.clear();
    hidden.clear();
    clear = 0;
    trim_to = -1;
    size_t offset = 0;
    uint32_t n = 0;
    if (!read_u32(data, size, offset, n, error) || n == 0) {
        return false;
    }
    const size_t n_entries = n;
    uint32_t raw_clear = 0, raw_trim = 0;
    if (!read_u32(data, size, offset, raw_clear, error) ||
        !read_u32(data, size, offset, raw_trim, error)) {
        return false;
    }
    clear = static_cast<int32_t>(raw_clear);
    trim_to = static_cast<int32_t>(raw_trim);
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
    if (n_embd == 0) {
        if (data_bytes != n_entries * sizeof(int32_t)) {
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
        if (data_bytes != n_entries * n_embd * sizeof(float)) {
            error = "batch hidden payload size mismatch";
            return false;
        }
        hidden.resize(n_entries * n_embd);
        std::memcpy(hidden.data(), data + offset, data_bytes);
    }
    return true;
}

} // namespace prima
