#include "../src/prima-distributed-protocol.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Runtime check that survives both Debug and Release (-DNDEBUG) builds. assert()
// is compiled out under -DNDEBUG, so these tests would otherwise verify nothing
// in the default CMake config.
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            std::abort();                                                                 \
        }                                                                                 \
    } while (0)

int main() {
    // Round-trip a representative message.
    prima::message message;
    message.type = prima::message_type::hidden_state;
    message.rank = 2;
    message.sequence = 17;
    message.dtype = prima::data_type::f32;
    message.shape = {1, 3, 1024};
    message.payload.resize(12);
    for (size_t i = 0; i < message.payload.size(); ++i) {
        message.payload[i] = static_cast<uint8_t>(i * 7);
    }

    const std::vector<uint8_t> frame = prima::encode_frame(message);
    CHECK(frame.size() > message.payload.size());

    prima::message decoded;
    std::string error;
    CHECK(prima::decode_frame(frame.data(), frame.size(), decoded, error));
    CHECK(decoded.type == message.type);
    CHECK(decoded.rank == message.rank);
    CHECK(decoded.sequence == message.sequence);
    CHECK(decoded.dtype == message.dtype);
    CHECK(decoded.shape == message.shape);
    CHECK(decoded.payload == message.payload);

    // A corrupted magic is rejected with a descriptive error.
    std::vector<uint8_t> bad_magic = frame;
    bad_magic[8] ^= 0xff;
    CHECK(!prima::decode_frame(bad_magic.data(), bad_magic.size(), decoded, error));
    CHECK(error.find("magic") != std::string::npos);

    // A truncated frame is rejected (shorter than the declared prefix).
    std::vector<uint8_t> truncated(frame.begin(), frame.end() - 1);
    CHECK(!prima::decode_frame(truncated.data(), truncated.size(), decoded, error));
    CHECK(error.find("truncated") != std::string::npos);

    // An oversized payload cannot be encoded.
    prima::message too_large = message;
    too_large.payload.resize(prima::max_payload_bytes + 1);
    CHECK(prima::encode_frame(too_large).empty());

    // Every message type + data type must round-trip.
    for (const auto type : {
             prima::message_type::hello,
             prima::message_type::batch_meta,
             prima::message_type::hidden_state,
             prima::message_type::token,
             prima::message_type::reset,
             prima::message_type::error,
         }) {
        for (const auto dtype : {
                 prima::data_type::none,
                 prima::data_type::f32,
                 prima::data_type::f16,
                 prima::data_type::i32,
             }) {
            prima::message m;
            m.type = type;
            m.flags = 0xa5;
            m.rank = 3;
            m.sequence = 123456789;
            m.dtype = dtype;
            m.shape = {2, 64};
            m.payload = {1, 2, 3, 4, 5};
            const std::vector<uint8_t> f = prima::encode_frame(m);
            prima::message d;
            CHECK(prima::decode_frame(f.data(), f.size(), d, error));
            CHECK(d.type == type);
            CHECK(d.flags == 0xa5);
            CHECK(d.rank == 3);
            CHECK(d.sequence == 123456789);
            CHECK(d.dtype == dtype);
            CHECK(d.shape == m.shape);
            CHECK(d.payload == m.payload);
        }
    }

    // An empty payload is a legal (if unusual) message.
    {
        prima::message m;
        m.type = prima::message_type::reset;
        m.payload.clear();
        const std::vector<uint8_t> f = prima::encode_frame(m);
        prima::message d;
        CHECK(prima::decode_frame(f.data(), f.size(), d, error));
        CHECK(d.type == prima::message_type::reset);
        CHECK(d.payload.empty());
    }

    return 0;
}
