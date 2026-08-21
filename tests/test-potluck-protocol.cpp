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
    std::string error;

    potluck::node_config expected;
    expected.n_workers = 2;
    expected.index = 1;
    expected.n_layer = 16;
    expected.n_ctx = 8192;
    expected.n_seq_max = 4;
    expected.n_ubatch = 256;
    expected.seed = 42;
    expected.temp = 0.7f;
    expected.top_p = 0.9f;
    expected.windows = {
        {0, 0, 4, 4},
        {1, 4, 10, 0},
        {0, 10, 16, 6},
    };

    std::vector<uint8_t> config_payload;
    CHECK(potluck::encode_config(expected, config_payload));
    potluck::node_config actual;
    CHECK(potluck::decode_config(config_payload.data(), config_payload.size(), actual, error));
    CHECK(actual.n_workers == expected.n_workers);
    CHECK(actual.index == expected.index);
    CHECK(actual.n_layer == expected.n_layer);
    CHECK(actual.n_ctx == expected.n_ctx);
    CHECK(actual.n_seq_max == expected.n_seq_max);
    CHECK(actual.n_ubatch == expected.n_ubatch);
    CHECK(actual.seed == expected.seed);
    CHECK(actual.temp == expected.temp);
    CHECK(actual.top_p == expected.top_p);
    CHECK(actual.windows.size() == expected.windows.size());
    for (size_t i = 0; i < expected.windows.size(); ++i) {
        CHECK(actual.windows[i].owner == expected.windows[i].owner);
        CHECK(actual.windows[i].start == expected.windows[i].start);
        CHECK(actual.windows[i].end == expected.windows[i].end);
        CHECK(actual.windows[i].n_gpu_layers == expected.windows[i].n_gpu_layers);
    }

    std::vector<uint8_t> bad_version = config_payload;
    bad_version[0] = 2;
    CHECK(!potluck::decode_config(bad_version.data(), bad_version.size(), actual, error));
    CHECK(error == "unsupported ring config version");

    potluck::node_config gap = expected;
    gap.windows[1].start = 5;
    CHECK(!potluck::encode_config(gap, config_payload));

    CHECK(potluck::encode_config(expected, config_payload));
    config_payload.push_back(0);
    CHECK(!potluck::decode_config(config_payload.data(), config_payload.size(), actual, error));
    CHECK(error == "trailing ring config bytes");

    const std::vector<int32_t> expected_pos = {0, 7};
    const std::vector<int32_t> expected_seq = {2, 3};
    const std::vector<int32_t> expected_tokens = {11, 13};
    std::vector<uint8_t> batch_payload;
    CHECK(potluck::encode_batch_payload(expected_pos, expected_seq, expected_tokens,
                                        nullptr, 0, 1, -1, 2, batch_payload));
    int32_t clear = 0;
    int32_t trim_to = 0;
    uint32_t n_logits = 0;
    std::vector<int32_t> pos;
    std::vector<int32_t> seq;
    std::vector<int32_t> tokens;
    std::vector<float> hidden;
    CHECK(potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                        clear, trim_to, n_logits, pos, seq, tokens, hidden, error));
    CHECK(clear == 1);
    CHECK(trim_to == -1);
    CHECK(n_logits == 2);
    CHECK(pos == expected_pos);
    CHECK(seq == expected_seq);
    CHECK(tokens == expected_tokens);
    CHECK(hidden.empty());
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

    // Accelerator profiles round-trip and reject damaged payloads.
    {
        const potluck::accel_profile expected = {3, potluck::accel_kind::cuda, 2147483648, 4294967296};
        std::vector<uint8_t> payload;
        CHECK(potluck::encode_accel_profile(expected, payload));
        CHECK(payload.size() == 25);
        potluck::accel_profile actual;
        CHECK(potluck::decode_accel_profile(payload.data(), payload.size(), actual, error));
        CHECK(actual.rank == expected.rank);
        CHECK(actual.kind == expected.kind);
        CHECK(actual.free_bytes == expected.free_bytes);
        CHECK(actual.total_bytes == expected.total_bytes);

        const potluck::accel_profile metal = {0, potluck::accel_kind::metal, 1, 2};
        CHECK(potluck::encode_accel_profile(metal, payload));
        CHECK(potluck::decode_accel_profile(payload.data(), payload.size(), actual, error));
        CHECK(actual.kind == potluck::accel_kind::metal);

        payload.push_back(0);
        CHECK(!potluck::decode_accel_profile(payload.data(), payload.size(), actual, error));
        CHECK(error == "accelerator profile size mismatch");
        payload.pop_back();
        payload[0] = 0;
        CHECK(!potluck::decode_accel_profile(payload.data(), payload.size(), actual, error));
        CHECK(error == "invalid accelerator profile magic");
        payload[0] = 0x45; // restore magic byte
        payload[8] = 9;    // kind byte out of range
        CHECK(!potluck::decode_accel_profile(payload.data(), payload.size(), actual, error));
        CHECK(error == "unknown accelerator kind");
    }

    return 0;
}
