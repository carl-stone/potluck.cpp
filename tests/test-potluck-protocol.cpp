#include "../src/potluck-protocol.h"

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
    potluck::message message;
    message.type = potluck::message_type::hidden_state;
    message.rank = 2;
    message.sequence = 17;
    message.dtype = potluck::data_type::f32;
    message.shape = {1, 3, 1024};
    message.payload.resize(12);
    for (size_t i = 0; i < message.payload.size(); ++i) {
        message.payload[i] = static_cast<uint8_t>(i * 7);
    }

    const std::vector<uint8_t> frame = potluck::encode_frame(message);
    CHECK(frame.size() > message.payload.size());

    potluck::message decoded;
    std::string error;
    CHECK(potluck::decode_frame(frame.data(), frame.size(), decoded, error));
    CHECK(decoded.type == message.type);
    CHECK(decoded.rank == message.rank);
    CHECK(decoded.sequence == message.sequence);
    CHECK(decoded.dtype == message.dtype);
    CHECK(decoded.shape == message.shape);
    CHECK(decoded.payload == message.payload);

    // A corrupted magic is rejected with a descriptive error.
    std::vector<uint8_t> bad_magic = frame;
    bad_magic[8] ^= 0xff;
    CHECK(!potluck::decode_frame(bad_magic.data(), bad_magic.size(), decoded, error));
    CHECK(error.find("magic") != std::string::npos);

    // A peer using another wire version fails loudly and names the versions.
    std::vector<uint8_t> bad_version = frame;
    bad_version[12] = 99;
    bad_version[13] = 0;
    CHECK(!potluck::decode_frame(bad_version.data(), bad_version.size(), decoded, error));
    CHECK(error == "protocol version mismatch: local " + std::to_string(potluck::protocol_version) +
                   ", peer 99; rebuild both sides from the same commit");

    // A truncated frame is rejected (shorter than the declared prefix).
    std::vector<uint8_t> truncated(frame.begin(), frame.end() - 1);
    CHECK(!potluck::decode_frame(truncated.data(), truncated.size(), decoded, error));
    CHECK(error.find("truncated") != std::string::npos);

    // An oversized payload cannot be encoded.
    potluck::message too_large = message;
    too_large.payload.resize(potluck::max_payload_bytes + 1);
    CHECK(potluck::encode_frame(too_large).empty());

    // Every message type + data type must round-trip.
    for (const auto type : {
             potluck::message_type::hello,
             potluck::message_type::batch_meta,
             potluck::message_type::hidden_state,
             potluck::message_type::token,
             potluck::message_type::reset,
             potluck::message_type::error,
         }) {
        for (const auto dtype : {
                 potluck::data_type::none,
                 potluck::data_type::f32,
                 potluck::data_type::f16,
                 potluck::data_type::i32,
             }) {
            potluck::message m;
            m.type = type;
            m.flags = 0xa5;
            m.rank = 3;
            m.sequence = 123456789;
            m.dtype = dtype;
            m.shape = {2, 64};
            m.payload = {1, 2, 3, 4, 5};
            const std::vector<uint8_t> f = potluck::encode_frame(m);
            potluck::message d;
            CHECK(potluck::decode_frame(f.data(), f.size(), d, error));
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
        potluck::message m;
        m.type = potluck::message_type::reset;
        m.payload.clear();
        const std::vector<uint8_t> f = potluck::encode_frame(m);
        potluck::message d;
        CHECK(potluck::decode_frame(f.data(), f.size(), d, error));
        CHECK(d.type == potluck::message_type::reset);
        CHECK(d.payload.empty());
    }

    // Live benchmark metrics round-trip with fixed-width wire fields.
    {
        const std::vector<potluck::worker_bench_metrics> expected = {
            {0, 0, 12, 305.25f, 849.9f, 20},
            {1, 12, 24, 167.75f, 1112.5f, 20},
        };
        std::vector<uint8_t> payload;
        CHECK(potluck::encode_worker_bench_metrics(expected, payload));
        std::vector<potluck::worker_bench_metrics> actual;
        CHECK(potluck::decode_worker_bench_metrics(payload.data(), payload.size(), actual, error));
        CHECK(actual.size() == expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            CHECK(actual[i].index == expected[i].index);
            CHECK(actual[i].start == expected[i].start);
            CHECK(actual[i].end == expected[i].end);
            CHECK(actual[i].decode_tok_s == expected[i].decode_tok_s);
            CHECK(actual[i].peak_rss_mb == expected[i].peak_rss_mb);
            CHECK(actual[i].decoded_positions == expected[i].decoded_positions);
        }
        payload.push_back(0);
        CHECK(!potluck::decode_worker_bench_metrics(payload.data(), payload.size(), actual, error));
    }

    return 0;
}
