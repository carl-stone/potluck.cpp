#include "halda.h"

#include <cstdio>
#include <cstdlib>
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

halda_model fixture_model() {
    halda_model model;
    model.n_layer = 12;
    model.n_embd = 8;
    model.n_ff = 16;
    model.n_head = 2;
    model.n_head_kv = 1;
    model.n_vocab = 32;
    model.n_ctx = 16;
    model.n_ubatch = 2;
    model.b = 100;
    model.bi = 3200;
    model.bo = 6400;
    model.kv_per_layer = 16;
    model.types = {0};
    model.layer_flops = {1.0};
    return model;
}

halda_device fixture_device(uint32_t rank, uint64_t host_free,
                            double disk_seq, double disk_rnd) {
    halda_device device;
    device.original_rank = rank;
    device.profile.os = halda_os_kind::linux_os;
    device.profile.accel = halda_accel_kind::none;
    device.profile.host_free_bytes = host_free;
    device.profile.host_total_bytes = host_free * 2;
    device.profile.cpu_gflops = {100.0};
    device.profile.disk_read_seq_gbps = disk_seq;
    device.profile.disk_read_rnd_gbps = disk_rnd;
    return device;
}

halda_options fixture_options() {
    halda_options options;
    options.model = fixture_model();
    options.devices.push_back(fixture_device(0, 64ull * 1024ull * 1024ull, 1.0, 1.0));
    options.devices.push_back(fixture_device(1, 64ull * 1024ull * 1024ull, 1.0, 1.0));
    options.devices[1].profile.cpu_gflops = {10.0};
    options.k_override = 1;
    options.master_priority = 1.0;
    return options;
}
void test_cpu_only_device() {
    halda_options options = fixture_options();
    options.devices.resize(1);
    options.devices[0].profile.free_bytes = 0;
    options.devices[0].profile.total_bytes = 0;
    std::string error;
    halda_solution solution;
    CHECK(solve_halda(options, solution, error));
    CHECK(error.empty());
    CHECK(solution.original_w[0] > 0);
}

void test_exact_single_survivor() {
    halda_options options = fixture_options();
    std::string error;
    halda_solution solution;
    CHECK(solve_halda(options, solution, error));
    CHECK(error.empty());
    CHECK(solution.k == 1);
    CHECK(solution.W == 12);
    CHECK(solution.active_original_ranks == std::vector<uint32_t>{0});
    CHECK(solution.w == std::vector<uint32_t>{12});
    CHECK(solution.n == std::vector<uint32_t>{0});
    CHECK(solution.set_labels == std::vector<std::string>{"M4"});
    CHECK(solution.original_w == std::vector<uint32_t>({12, 0}));
    CHECK(solution.original_n == std::vector<uint32_t>({0, 0}));
    CHECK(solution.removed_original_ranks == std::vector<uint32_t>{1});
}

void test_memory_pressure_and_slow_disk() {
    halda_options options = fixture_options();
    options.devices[1].profile.host_free_bytes = 1100;
    options.devices[1].profile.disk_read_seq_gbps = 0.05;
    std::string error;
    halda_solution solution;
    CHECK(solve_halda(options, solution, error));
    CHECK(solution.k == 1);
    CHECK(solution.active_original_ranks == std::vector<uint32_t>{0});
    CHECK(solution.original_set_labels[1] == "removed");

    options.devices[1].profile.host_free_bytes = 64ull * 1024ull * 1024ull;
    options.devices[1].profile.disk_read_seq_gbps = 0.05;
    options.master_priority = 0.01;
    solution = {};
    CHECK(solve_halda(options, solution, error));
    CHECK(solution.k == 1);
    CHECK(solution.original_set_labels[1] == "M4");
    CHECK(solution.original_w[1] > 0);
}

void test_route_and_determinism() {
    halda_options options = fixture_options();
    std::string error;
    halda_solution first;
    halda_solution second;
    CHECK(solve_halda(options, first, error));
    CHECK(solve_halda(options, second, error));
    CHECK(first.k == second.k);
    CHECK(first.active_original_ranks == second.active_original_ranks);
    CHECK(first.w == second.w);
    CHECK(first.n == second.n);
    CHECK(first.set_labels == second.set_labels);

    std::vector<halda_route_window> route;
    CHECK(build_halda_route(first, route, error));
    CHECK(route.size() == first.k * first.w.size());
    CHECK(route.front().start == 0);
    CHECK(route.back().end == first.n_layer);
    for (size_t i = 1; i < route.size(); ++i) {
        CHECK(route[i - 1].end == route[i].start);
    }
}
halda_options tight_window_options(uint32_t n_ubatch) {
    halda_options options = fixture_options();
    options.devices[0].name = "tight-worker";
    options.devices[1].name = "roomy-worker";
    options.devices[0].profile.host_free_bytes = 8300;
    options.devices[0].profile.host_total_bytes = 16600;
    options.devices[1].profile.host_free_bytes = 64ull * 1024ull * 1024ull;
    options.devices[1].profile.host_total_bytes = 128ull * 1024ull * 1024ull;
    options.model.n_ubatch = n_ubatch;
    options.fixed_w = {2, 4};
    options.k_override = -1;
    return options;
}

void test_tie_prefers_low_rank() {
    halda_options options = fixture_options();
    options.devices[0].profile.host_free_bytes = 1800;
    options.devices[0].profile.host_total_bytes = 3600;
    options.devices[1].profile.host_free_bytes = 1800;
    options.devices[1].profile.host_total_bytes = 3600;
    options.devices[1].profile.cpu_gflops = {100.0};
    options.head_participates = false;
    options.k_override = -1;
    std::string error;
    halda_solution first;
    halda_solution second;
    CHECK(solve_halda(options, first, error));
    CHECK(solve_halda(options, second, error));
    CHECK(first.active_original_ranks == std::vector<uint32_t>({0, 1}));
    CHECK(first.w == second.w);
    CHECK(first.w[0] > first.w[1]);
}

void test_compute_buffers_are_priced() {
    halda_options options = tight_window_options(2);
    std::string error;
    halda_solution solution;
    CHECK(solve_halda(options, solution, error));
    options.model.n_ubatch = 512;
    solution = {};
    CHECK(!solve_halda(options, solution, error));
    CHECK(error.find("tight-worker") != std::string::npos);
}

void test_resident_memory_follows_window() {
    halda_options options = tight_window_options(2);
    std::string error;
    halda_solution solution;
    CHECK(solve_halda(options, solution, error));
    options.fixed_w = {4, 2};
    solution = {};
    CHECK(!solve_halda(options, solution, error));
    CHECK(error.find("tight-worker") != std::string::npos);
}


void test_fixed_window_validation() {
    halda_options options = fixture_options();
    options.k_override = -1;
    options.fixed_w = {4, 2};
    std::string error;
    halda_solution solution;
    CHECK(solve_halda(options, solution, error));
    CHECK(solution.k == 2);
    CHECK(solution.w == std::vector<uint32_t>({4, 2}));
    CHECK(solution.original_w == std::vector<uint32_t>({4, 2}));

    options.fixed_w = {4, 0};
    solution = {};
    CHECK(!solve_halda(options, solution, error));
    CHECK(error == "HALDA fixed window widths must be positive");

    options.fixed_w = {4};
    solution = {};
    CHECK(!solve_halda(options, solution, error));
    CHECK(error == "HALDA fixed window widths must have one entry per device");

    options.fixed_w = {4, 3};
    solution = {};
    CHECK(!solve_halda(options, solution, error));
    CHECK(error == "HALDA fixed window widths must sum to more than one and divide the layer count");
}

void test_fixed_window_memory_diagnostic() {
    halda_options options = fixture_options();
    options.devices[0].name = "memory-poor-worker";
    options.devices[0].profile.host_free_bytes = 1000;
    options.devices[0].profile.host_total_bytes = 2000;
    options.fixed_w = {4, 2};
    options.k_override = -1;
    std::string error;
    halda_solution solution;
    CHECK(!solve_halda(options, solution, error));
    CHECK(error.find("memory-poor-worker") != std::string::npos);
    CHECK(error.find("requested") != std::string::npos);
    CHECK(error.find("available") != std::string::npos);
}


} // namespace

int main() {
    test_cpu_only_device();
    test_exact_single_survivor();
    test_memory_pressure_and_slow_disk();
    test_route_and_determinism();
    test_tie_prefers_low_rank();
    test_compute_buffers_are_priced();
    test_resident_memory_follows_window();
    test_fixed_window_validation();
    test_fixed_window_memory_diagnostic();
    return 0;
}
