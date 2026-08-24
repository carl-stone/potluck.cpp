#include "../tools/potluck-server/internal.h"
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
    expected.n_rs_seq = 5;
    expected.seed = 42;
    expected.top_p = 0.9f;
    expected.prefetch = potluck::prefetch_mode::force;
    expected.n_cycles = 3;
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
    CHECK(actual.n_rs_seq == expected.n_rs_seq);
    CHECK(actual.seed == expected.seed);
    CHECK(actual.temp == expected.temp);
    CHECK(actual.top_p == expected.top_p);
    CHECK(actual.prefetch == expected.prefetch);
    CHECK(actual.n_cycles == expected.n_cycles);
    CHECK(actual.windows.size() == expected.windows.size());
    for (size_t i = 0; i < expected.windows.size(); ++i) {
        CHECK(actual.windows[i].owner == expected.windows[i].owner);
        CHECK(actual.windows[i].start == expected.windows[i].start);
        CHECK(actual.windows[i].end == expected.windows[i].end);
        CHECK(actual.windows[i].n_gpu_layers == expected.windows[i].n_gpu_layers);
    }

    std::vector<uint8_t> bad_version = config_payload;
    bad_version[0] = 1;
    CHECK(!potluck::decode_config(bad_version.data(), bad_version.size(), actual, error));
    CHECK(error == "unsupported ring config version");

    potluck::node_config gap = expected;
    gap.windows[1].start = 5;
    CHECK(!potluck::encode_config(gap, config_payload));

    CHECK(potluck::encode_config(expected, config_payload));
    config_payload.push_back(0);
    CHECK(!potluck::decode_config(config_payload.data(), config_payload.size(), actual, error));
    CHECK(error == "trailing ring config bytes");
    potluck::node_config invalid_config = expected;
    invalid_config.prefetch = static_cast<potluck::prefetch_mode>(99);
    CHECK(!potluck::encode_config(invalid_config, config_payload));
    invalid_config = expected;
    invalid_config.n_cycles = 0;
    CHECK(!potluck::encode_config(invalid_config, config_payload));

    const std::vector<int32_t> expected_pos = {0, 7};
    const std::vector<int32_t> expected_seq = {2, 3};
    const std::vector<int32_t> expected_tokens = {11, 13};
    const std::vector<int32_t> expected_drafts = {17, 19, 23};
    std::vector<uint8_t> batch_payload;
    CHECK(potluck::encode_batch_payload(expected_pos, expected_seq, expected_tokens,
                                        expected_drafts, 2, nullptr, 0,
                                        1, 2, 7, 2, batch_payload));
    CHECK(batch_payload[0] == 'B' && batch_payload[1] == 'T' &&
          batch_payload[2] == 'P' && batch_payload[3] == '1');
    CHECK(batch_payload[4] == 1 && batch_payload[8] == 2 &&
          batch_payload[12] == 1 && batch_payload[16] == 2 &&
          batch_payload[20] == 7 && batch_payload[24] == 2 &&
          batch_payload[28] == 3 && batch_payload[32] == 2);
    int32_t clear_seq = 0;
    int32_t trim_seq = 0;
    int32_t trim_to = 0;
    uint32_t n_logits = 0;
    uint32_t accepted_count = 0;
    std::vector<int32_t> pos;
    std::vector<int32_t> seq;
    std::vector<int32_t> tokens;
    std::vector<int32_t> drafts;
    std::vector<float> hidden;
    CHECK(potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                        clear_seq, trim_seq, trim_to, n_logits,
                                        pos, seq, tokens, drafts, accepted_count,
                                        hidden, error));
    CHECK(clear_seq == 1);
    CHECK(trim_seq == 2);
    CHECK(trim_to == 7);
    CHECK(n_logits == 2);
    CHECK(pos == expected_pos);
    CHECK(seq == expected_seq);
    CHECK(tokens == expected_tokens);
    CHECK(drafts == expected_drafts);
    CHECK(accepted_count == 2);
    CHECK(hidden.empty());
    {
        const std::vector<int32_t> partial_pos = {5, 12};
        const std::vector<int32_t> partial_seq = {1, 0};
        const std::vector<int32_t> partial_tokens = {29, 31};
        const std::vector<int32_t> partial_drafts = {37, 41, 43, 47};
        const uint32_t partial_accepted = 2;
        CHECK(partial_accepted > 0 && partial_accepted < partial_drafts.size());
        CHECK(potluck::encode_batch_payload(
            partial_pos, partial_seq, partial_tokens, partial_drafts,
            partial_accepted, nullptr, 0, 4, 5, 6, 2, batch_payload));
        CHECK(potluck::decode_batch_payload(
            batch_payload.data(), batch_payload.size(), 0,
            clear_seq, trim_seq, trim_to, n_logits, pos, seq, tokens, drafts,
            accepted_count, hidden, error));
        CHECK(clear_seq == 4 && trim_seq == 5 && trim_to == 6 && n_logits == 2);
        CHECK(pos == partial_pos);
        CHECK(seq == partial_seq);
        CHECK(tokens == partial_tokens);
        CHECK(drafts == partial_drafts);
        CHECK(accepted_count == partial_accepted);
    }
    CHECK(potluck::encode_batch_payload(expected_pos, expected_seq, expected_tokens,
                                        expected_drafts, 2, nullptr, 0,
                                        1, 2, 7, 2, batch_payload));
    batch_payload[0] = 0;
    CHECK(!potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                         clear_seq, trim_seq, trim_to, n_logits,
                                         pos, seq, tokens, drafts, accepted_count,
                                         hidden, error));
    CHECK(error == "invalid batch payload magic");
    CHECK(potluck::encode_batch_payload(expected_pos, expected_seq, expected_tokens,
                                        expected_drafts, 2, nullptr, 0,
                                        1, 2, 7, 2, batch_payload));
    batch_payload[4] = 2;
    CHECK(!potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                         clear_seq, trim_seq, trim_to, n_logits,
                                         pos, seq, tokens, drafts, accepted_count,
                                         hidden, error));
    CHECK(error == "unsupported batch payload version");
    CHECK(potluck::encode_batch_payload(expected_pos, expected_seq, expected_tokens,
                                        expected_drafts, 2, nullptr, 0,
                                        1, 2, 7, 2, batch_payload));
    batch_payload.push_back(0);
    CHECK(!potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                         clear_seq, trim_seq, trim_to, n_logits,
                                         pos, seq, tokens, drafts, accepted_count,
                                         hidden, error));
    batch_payload.pop_back();
    CHECK(potluck::encode_batch_payload(expected_pos, expected_seq, expected_tokens,
                                        expected_drafts, 2, nullptr, 0,
                                        -2, -1, -1, 0, batch_payload));
    CHECK(potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                        clear_seq, trim_seq, trim_to, n_logits,
                                        pos, seq, tokens, drafts, accepted_count,
                                        hidden, error));
    CHECK(clear_seq == -2);
    CHECK(trim_seq == -1);
    CHECK(trim_to == -1);
    CHECK(n_logits == 0);
    CHECK(drafts == expected_drafts && accepted_count == 2);
    CHECK(potluck::encode_batch_payload({}, {}, {}, {}, 0, nullptr, 0,
                                        3, -1, -1, 0, batch_payload));
    CHECK(batch_payload.size() == 36);
    CHECK(potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                        clear_seq, trim_seq, trim_to, n_logits,
                                        pos, seq, tokens, drafts, accepted_count,
                                        hidden, error));
    CHECK(clear_seq == 3);
    CHECK(trim_seq == -1);
    CHECK(trim_to == -1);
    CHECK(n_logits == 0 && accepted_count == 0);
    CHECK(pos.empty() && seq.empty() && tokens.empty() && drafts.empty() && hidden.empty());
    batch_payload[20] = 0xfe;
    batch_payload[21] = 0xff;
    batch_payload[22] = 0xff;
    batch_payload[23] = 0xff;
    CHECK(!potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                         clear_seq, trim_seq, trim_to, n_logits,
                                         pos, seq, tokens, drafts, accepted_count,
                                         hidden, error));
    CHECK(error == "invalid batch clear or trim controls");
    CHECK(!potluck::encode_batch_payload(expected_pos, expected_seq, expected_tokens,
                                         expected_drafts, 4, nullptr, 0,
                                         1, 2, 7, 2, batch_payload));
    CHECK(potluck::encode_batch_payload(expected_pos, expected_seq, expected_tokens,
                                        expected_drafts, 2, nullptr, 0,
                                        1, 2, 7, 2, batch_payload));
    batch_payload[8] = 3;
    CHECK(!potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 0,
                                         clear_seq, trim_seq, trim_to, n_logits,
                                         pos, seq, tokens, drafts, accepted_count,
                                         hidden, error));
    CHECK(error == "batch token payload size mismatch");
    {
        const float expected_hidden[] = {1.0f, 2.0f, 3.0f, 4.0f};
        CHECK(potluck::encode_batch_payload(expected_pos, expected_seq, {},
                                            {}, 0, expected_hidden, 2,
                                            -1, -1, -1, 1, batch_payload));
        CHECK(potluck::decode_batch_payload(batch_payload.data(), batch_payload.size(), 2,
                                            clear_seq, trim_seq, trim_to, n_logits,
                                            pos, seq, tokens, drafts, accepted_count,
                                            hidden, error));
        CHECK(pos == expected_pos);
        CHECK(seq == expected_seq);
        CHECK(tokens.empty() && drafts.empty() && accepted_count == 0);
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
        }
        payload.push_back(0);
        CHECK(!potluck::decode_worker_bench_metrics(payload.data(), payload.size(), actual, error));
    }

    // Device profiles round-trip and reject damaged payloads.
    {
        potluck::device_profile expected;
        expected.rank = 3;
        expected.kind = potluck::accel_kind::cuda;
        expected.os = potluck::os_kind::linux_os;
        expected.free_bytes = 2147483648;
        expected.total_bytes = 4294967296;
        expected.host_free_bytes = 8589934592;
        expected.host_total_bytes = 17179869184;
        expected.cpu_gflops = {1.5f, 2.5f};
        expected.accel_gflops = {10.0f, 20.0f, 30.0f};
        expected.mem_copy_delay_ms = 0.25f;
        expected.accel_copy_delay_ms = 0.5f;
        expected.disk_read_seq_gbps = 3.0f;
        expected.disk_read_rnd_gbps = 1.25f;
        expected.n_cpu_threads = 12;
        std::vector<uint8_t> payload;
        CHECK(potluck::encode_device_profile(expected, payload));
        CHECK(payload.size() == 90);
        potluck::device_profile actual;
        CHECK(potluck::decode_device_profile(payload.data(), payload.size(), actual, error));
        CHECK(actual.rank == expected.rank);
        CHECK(actual.kind == expected.kind);
        CHECK(actual.os == expected.os);
        CHECK(actual.free_bytes == expected.free_bytes);
        CHECK(actual.total_bytes == expected.total_bytes);
        CHECK(actual.host_free_bytes == expected.host_free_bytes);
        CHECK(actual.host_total_bytes == expected.host_total_bytes);
        CHECK(actual.cpu_gflops == expected.cpu_gflops);
        CHECK(actual.accel_gflops == expected.accel_gflops);
        CHECK(actual.mem_copy_delay_ms == expected.mem_copy_delay_ms);
        CHECK(actual.accel_copy_delay_ms == expected.accel_copy_delay_ms);
        CHECK(actual.disk_read_seq_gbps == expected.disk_read_seq_gbps);
        CHECK(actual.disk_read_rnd_gbps == expected.disk_read_rnd_gbps);
        CHECK(actual.n_cpu_threads == expected.n_cpu_threads);

        payload.push_back(0);
        CHECK(!potluck::decode_device_profile(payload.data(), payload.size(), actual, error));
        CHECK(error == "device profile size mismatch");
        payload.resize(25);
        CHECK(!potluck::decode_device_profile(payload.data(), payload.size(), actual, error));
        CHECK(error == "device profile size mismatch");
        CHECK(potluck::encode_device_profile(expected, payload));
        payload[0] = 'E';
        payload[1] = 'A';
        payload[2] = 'P';
        payload[3] = '2';
        CHECK(!potluck::decode_device_profile(payload.data(), payload.size(), actual, error));
        CHECK(error == "invalid device profile magic");
        CHECK(potluck::encode_device_profile(expected, payload));
        payload[8] = 9;
        CHECK(!potluck::decode_device_profile(payload.data(), payload.size(), actual, error));
        CHECK(error == "unknown accelerator kind");
        CHECK(potluck::encode_device_profile(expected, payload));
        payload[9] = 9;
        CHECK(!potluck::decode_device_profile(payload.data(), payload.size(), actual, error));
        CHECK(error == "unknown operating system kind");
        CHECK(potluck::encode_device_profile(expected, payload));
        payload[42] = 0xff;
        payload[43] = 0xff;
        payload[44] = 0xff;
        payload[45] = 0xff;
        CHECK(!potluck::decode_device_profile(payload.data(), payload.size(), actual, error));
        CHECK(error == "invalid device profile vector count");
        potluck::device_profile invalid = expected;
        invalid.cpu_gflops[0] = -1.0f;
        CHECK(!potluck::encode_device_profile(invalid, payload));
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
    {
        constexpr uint64_t gib = 1024ull * 1024ull * 1024ull;
        device_probe head;
        head.profile.host_total_bytes = 16ull * gib;
        head.profile.host_free_bytes = 5ull * gib;
        const head_participation_plan excluded =
            plan_head_participation(head, 4ull * gib, 3ull * gib);
        CHECK(excluded.budget == 1ull * gib);
        CHECK(!excluded.participates);

        head.profile.host_free_bytes = 12ull * gib;
        const head_participation_plan included =
            plan_head_participation(head, 4ull * gib, 3ull * gib);
        CHECK(included.budget == 8ull * gib);
        CHECK(included.participates);

        const head_participation_plan reserve_excluded =
            plan_head_participation(head, 11ull * gib, 3ull * gib);
        CHECK(reserve_excluded.budget == 1ull * gib);
        CHECK(!reserve_excluded.participates);

    }

    return 0;
}
