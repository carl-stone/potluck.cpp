#include "../tools/potluck-server/internal.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CHECK(condition)                                                                    \
    do {                                                                                    \
        if (!(condition)) {                                                                 \
            std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #condition, __FILE__, __LINE__); \
            std::abort();                                                                   \
        }                                                                                   \
    } while (0)

namespace {

void print_float_vector(const char * name, const std::vector<float> & values) {
    std::printf(" %s=[", name);
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            std::printf(",");
        }
        std::printf("%.9g", static_cast<double>(values[index]));
    }
    std::printf("]");
}

void print_probe(const char * name, double seconds, const device_probe & probe) {
    const potluck::device_profile & profile = probe.profile;
    std::printf(
        "%s duration_ms=%.3f ok=%d rank=%u kind=%u free_bytes=%llu total_bytes=%llu "
        "host_free_bytes=%llu host_total_bytes=%llu os=%u n_cpu_threads=%u "
        "mem_copy_delay_ms=%.9g accel_copy_delay_ms=%.9g disk_read_seq_gbps=%.9g "
        "disk_read_rnd_gbps=%.9g\n",
        name, seconds * 1000.0, probe.ok ? 1 : 0, profile.rank,
        static_cast<unsigned int>(profile.kind),
        static_cast<unsigned long long>(profile.free_bytes),
        static_cast<unsigned long long>(profile.total_bytes),
        static_cast<unsigned long long>(profile.host_free_bytes),
        static_cast<unsigned long long>(profile.host_total_bytes),
        static_cast<unsigned int>(profile.os), profile.n_cpu_threads,
        static_cast<double>(profile.mem_copy_delay_ms),
        static_cast<double>(profile.accel_copy_delay_ms),
        static_cast<double>(profile.disk_read_seq_gbps),
        static_cast<double>(profile.disk_read_rnd_gbps));
    print_float_vector("cpu_gflops", profile.cpu_gflops);
    print_float_vector("accel_gflops", profile.accel_gflops);
    std::printf("\n");
    if (!probe.error.empty()) {
        std::printf("%s error=%s\n", name, probe.error.c_str());
    }
}

void check_float_vectors_equal(const std::vector<float> & actual,
                               const std::vector<float> & expected) {
    CHECK(actual.size() == expected.size());
    if (!actual.empty()) {
        CHECK(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0);
    }
}

void check_speed_fields_equal(const potluck::device_profile & actual,
                              const potluck::device_profile & expected) {
    check_float_vectors_equal(actual.cpu_gflops, expected.cpu_gflops);
    check_float_vectors_equal(actual.accel_gflops, expected.accel_gflops);
    CHECK(std::memcmp(&actual.disk_read_seq_gbps, &expected.disk_read_seq_gbps,
                      sizeof(actual.disk_read_seq_gbps)) == 0);
    CHECK(std::memcmp(&actual.disk_read_rnd_gbps, &expected.disk_read_rnd_gbps,
                      sizeof(actual.disk_read_rnd_gbps)) == 0);
    CHECK(std::memcmp(&actual.mem_copy_delay_ms, &expected.mem_copy_delay_ms,
                      sizeof(actual.mem_copy_delay_ms)) == 0);
    CHECK(std::memcmp(&actual.accel_copy_delay_ms, &expected.accel_copy_delay_ms,
                      sizeof(actual.accel_copy_delay_ms)) == 0);
}

} // namespace

int main(int argc, char ** argv) {
    CHECK(argc >= 2);
    const std::string worker_path = argv[1];

    const auto first_start = std::chrono::steady_clock::now();
    const device_probe first = probe_local_worker(worker_path, {}, 64ull << 20);
    const auto first_end = std::chrono::steady_clock::now();
    const double first_seconds =
        std::chrono::duration<double>(first_end - first_start).count();
    CHECK(first.ok);
    print_probe("first_probe", first_seconds, first);

    constexpr size_t pressure_bytes = 3ull << 30;
    CHECK(pressure_bytes % sizeof(std::uint64_t) == 0);
    std::vector<std::uint64_t> pressure(pressure_bytes / sizeof(std::uint64_t));
    volatile std::uint64_t * pressure_data = pressure.data();
    std::mt19937_64 generator(0x4f4d505f50524f42ull);
    for (size_t index = 0; index < pressure.size(); ++index) {
        pressure_data[index] = generator();
    }

    const auto second_start = std::chrono::steady_clock::now();
    const device_probe second = probe_local_worker(worker_path, {}, 64ull << 20);
    const auto second_end = std::chrono::steady_clock::now();
    const double second_seconds =
        std::chrono::duration<double>(second_end - second_start).count();
    CHECK(second.ok);
    print_probe("second_probe", second_seconds, second);

    check_speed_fields_equal(second.profile, first.profile);
    CHECK(second_seconds * 5.0 <= first_seconds);

    constexpr uint64_t gib = 1ull << 30;
    constexpr uint64_t host_total_threshold = 8ull << 30;
    if (first.profile.host_total_bytes < host_total_threshold) {
        std::printf(
            "host pressure check skipped: host_total_bytes=%llu is below 8 GiB\n",
            static_cast<unsigned long long>(first.profile.host_total_bytes));
    } else {
        CHECK(first.profile.host_free_bytes >= second.profile.host_free_bytes);
        CHECK(first.profile.host_free_bytes - second.profile.host_free_bytes >= gib);
    }

    return 0;
}
