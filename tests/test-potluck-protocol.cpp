#include "../src/potluck-protocol.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
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
                                        nullptr, 0, 1, 2, 7, 2, batch_payload));
    CHECK(batch_payload.size() == 44);
    CHECK(batch_payload[0] == 2 && batch_payload[4] == 1 &&
          batch_payload[8] == 2 && batch_payload[12] == 7 && batch_payload[16] == 2);
    int32_t clear_seq = 0;
    int32_t trim_seq = 0;
    int32_t trim_to = 0;
    uint32_t n_logits = 0;
    std::vector<int32_t> pos;
    std::vector<int32_t> seq;
    std::vector<int32_t> tokens;
    std::vector<float> hidden;
    CHECK(potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                        clear_seq, trim_seq, trim_to, n_logits,
                                        pos, seq, tokens, hidden, error));
    CHECK(clear_seq == 1);
    CHECK(trim_seq == 2);
    CHECK(trim_to == 7);
    CHECK(n_logits == 2);
    CHECK(pos == expected_pos);
    CHECK(seq == expected_seq);
    CHECK(tokens == expected_tokens);
    CHECK(hidden.empty());
    batch_payload.push_back(0);
    CHECK(!potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                         clear_seq, trim_seq, trim_to, n_logits,
                                         pos, seq, tokens, hidden, error));
    batch_payload.pop_back();
    batch_payload[4] = 0xfd;
    batch_payload[5] = 0xff;
    batch_payload[6] = 0xff;
    batch_payload[7] = 0xff;
    CHECK(!potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                         clear_seq, trim_seq, trim_to, n_logits,
                                         pos, seq, tokens, hidden, error));
    CHECK(potluck::encode_batch_payload(expected_pos, expected_seq, expected_tokens,
                                        nullptr, 0, -2, -1, -1, 0, batch_payload));
    CHECK(batch_payload.size() == 44);
    CHECK(potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                        clear_seq, trim_seq, trim_to, n_logits,
                                        pos, seq, tokens, hidden, error));
    CHECK(clear_seq == -2);
    CHECK(trim_seq == -1);
    CHECK(trim_to == -1);
    CHECK(n_logits == 0);
    CHECK(potluck::encode_batch_payload({}, {}, {}, nullptr, 0, 3, -1, -1, 0,
                                        batch_payload));
    CHECK(batch_payload.size() == 20);
    CHECK(potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                        clear_seq, trim_seq, trim_to, n_logits,
                                        pos, seq, tokens, hidden, error));
    CHECK(clear_seq == 3);
    CHECK(trim_seq == -1);
    CHECK(trim_to == -1);
    CHECK(n_logits == 0);
    CHECK(pos.empty() && seq.empty() && tokens.empty() && hidden.empty());
    batch_payload[8] = 0xfe;
    batch_payload[9] = 0xff;
    batch_payload[10] = 0xff;
    batch_payload[11] = 0xff;
    CHECK(!potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                         clear_seq, trim_seq, trim_to, n_logits,
                                         pos, seq, tokens, hidden, error));
    CHECK(error == "invalid batch clear or trim controls");
    CHECK(potluck::encode_batch_payload(expected_pos, expected_seq, expected_tokens,
                                        nullptr, 0, 1, 2, 7, 2, batch_payload));
    batch_payload[16] = 3;
    CHECK(!potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                         clear_seq, trim_seq, trim_to, n_logits,
                                         pos, seq, tokens, hidden, error));
    CHECK(error == "invalid batch entry count");
    {
        const float expected_hidden[] = {1.0f, 2.0f, 3.0f, 4.0f};
        CHECK(potluck::encode_batch_payload(expected_pos, expected_seq, {},
                                            expected_hidden, 2, -1, -1, -1, 1, batch_payload));
        CHECK(batch_payload.size() == 52);
        CHECK(potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 2,
                                            clear_seq, trim_seq, trim_to, n_logits,
                                            pos, seq, tokens, hidden, error));
        CHECK(clear_seq == -1);
        CHECK(trim_seq == -1);
        CHECK(trim_to == -1);
        CHECK(n_logits == 1);
        CHECK(pos == expected_pos);
        CHECK(seq == expected_seq);
        CHECK(tokens.empty());
        CHECK(hidden.size() == 4);
        for (size_t i = 0; i < hidden.size(); ++i) {
            CHECK(hidden[i] == expected_hidden[i]);
        }
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

    // Accelerator profiles round-trip and reject damaged payloads.
    {
        const potluck::accel_profile expected = {
            3, potluck::accel_kind::cuda, 2147483648, 4294967296, 8589934592, 17179869184
        };
        std::vector<uint8_t> payload;
        CHECK(potluck::encode_accel_profile(expected, payload));
        CHECK(payload.size() == 41);
        potluck::accel_profile actual;
        CHECK(potluck::decode_accel_profile(payload.data(), payload.size(), actual, error));
        CHECK(actual.rank == expected.rank);
        CHECK(actual.kind == expected.kind);
        CHECK(actual.free_bytes == expected.free_bytes);
        CHECK(actual.total_bytes == expected.total_bytes);
        CHECK(actual.host_free_bytes == expected.host_free_bytes);
        CHECK(actual.host_total_bytes == expected.host_total_bytes);

        const potluck::accel_profile metal = {0, potluck::accel_kind::metal, 1, 2, 3, 4};
        CHECK(potluck::encode_accel_profile(metal, payload));
        CHECK(potluck::decode_accel_profile(payload.data(), payload.size(), actual, error));
        CHECK(actual.kind == potluck::accel_kind::metal);
        CHECK(actual.host_free_bytes == 3);
        CHECK(actual.host_total_bytes == 4);

        payload.push_back(0);
        CHECK(!potluck::decode_accel_profile(payload.data(), payload.size(), actual, error));
        CHECK(error == "accelerator profile size mismatch");
        payload.resize(25);
        CHECK(!potluck::decode_accel_profile(payload.data(), payload.size(), actual, error));
        CHECK(error == "accelerator profile size mismatch");
        CHECK(potluck::encode_accel_profile(metal, payload));
        payload[0] = 0;
        CHECK(!potluck::decode_accel_profile(payload.data(), payload.size(), actual, error));
        CHECK(error == "invalid accelerator profile magic");
        payload[0] = 0x45;
        payload[8] = 9;
        CHECK(!potluck::decode_accel_profile(payload.data(), payload.size(), actual, error));
        CHECK(error == "unknown accelerator kind");
        CHECK(!potluck::decode_accel_profile(nullptr, payload.size(), actual, error));
        CHECK(error == "truncated u32");
    }
    {
        potluck::slot_config expected;
        expected.seq = 2;
        expected.temp = 1.25f;
        expected.top_p = 0.85f;
        expected.top_k = 40;
        expected.seed = 17;
        expected.min_p = 0.1f;
        expected.presence_penalty = 0.2f;
        expected.frequency_penalty = 0.3f;
        expected.repeat_penalty = 1.1f;
        expected.penalty_last_n = 32;
        expected.logprobs = true;
        expected.top_logprobs = 5;
        std::vector<uint8_t> payload;
        CHECK(potluck::encode_slot_config(expected, payload));
        CHECK(payload.size() == 52);
        potluck::slot_config actual;
        CHECK(potluck::decode_slot_config(payload.data(), payload.size(), actual, error));
        CHECK(actual.seq == expected.seq);
        CHECK(actual.temp == expected.temp);
        CHECK(actual.top_p == expected.top_p);
        CHECK(actual.top_k == expected.top_k);
        CHECK(actual.seed == expected.seed);
        CHECK(actual.min_p == expected.min_p);
        CHECK(actual.presence_penalty == expected.presence_penalty);
        CHECK(actual.frequency_penalty == expected.frequency_penalty);
        CHECK(actual.repeat_penalty == expected.repeat_penalty);
        CHECK(actual.penalty_last_n == expected.penalty_last_n);
        CHECK(actual.logprobs == expected.logprobs);
        CHECK(actual.top_logprobs == expected.top_logprobs);


        payload.push_back(0);
        CHECK(!potluck::decode_slot_config(payload.data(), payload.size(), actual, error));
        CHECK(error == "slot config size mismatch");
        payload.pop_back();
        payload[0] = 0;
        CHECK(!potluck::decode_slot_config(payload.data(), payload.size(), actual, error));
        CHECK(error == "invalid slot config magic");
        CHECK(!potluck::encode_slot_config({-1, 1.0f, 1.0f, 0, 0}, payload));
        CHECK(!potluck::encode_slot_config(
            potluck::slot_config{0, 1.0f, 1.0f, 0, 0, 1.1f, 0.0f, 0.0f, 1.0f, 64, false, 0},
            payload));
        CHECK(potluck::encode_slot_config(expected, payload));
        payload[4] = 0xff;
        payload[5] = 0xff;
        payload[6] = 0xff;
        payload[7] = 0xff;
        CHECK(!potluck::decode_slot_config(payload.data(), payload.size(), actual, error));
        CHECK(error == "invalid slot config sequence");
    }

    {
        const potluck::batch_logprobs expected = {
            {{1, -0.5f}, {2, -1.25f}},
            {}
        };
        std::vector<uint8_t> payload;
        CHECK(potluck::encode_batch_logprobs(expected, payload));
        potluck::batch_logprobs actual;
        CHECK(potluck::decode_batch_logprobs(payload.data(), payload.size(), actual, error));
        CHECK(actual.size() == expected.size());
        CHECK(actual[0].size() == 2);
        CHECK(actual[0][0].token == 1);
        CHECK(actual[0][0].logprob == -0.5f);
        CHECK(actual[0][1].token == 2);
        CHECK(actual[1].empty());
        payload.push_back(0);
        CHECK(!potluck::decode_batch_logprobs(payload.data(), payload.size(), actual, error));
        CHECK(error == "batch logprobs size mismatch");
        const potluck::batch_logprobs invalid = {{{1, std::numeric_limits<float>::quiet_NaN()}}};
        CHECK(!potluck::encode_batch_logprobs(invalid, payload));
    }

    {
        const potluck::message heartbeat{
            potluck::message_type::heartbeat, 0, 1, 0x123456789abcdef0ull,
            potluck::data_type::none, {}, {}
        };
        const potluck::message reply{
            potluck::message_type::heartbeat_ack, 0, heartbeat.rank,
            heartbeat.sequence, potluck::data_type::none, {}, {}
        };
        CHECK(heartbeat.type != reply.type);
        CHECK(heartbeat.sequence == reply.sequence);
        CHECK(static_cast<uint16_t>(heartbeat.type) == 12);
        CHECK(static_cast<uint16_t>(reply.type) == 13);
    }

    return 0;
}
