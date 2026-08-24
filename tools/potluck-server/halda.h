#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class halda_os_kind : uint8_t {
    none = 0,
    macos = 1,
    linux_os = 2,
    None = none,
    MacOS = macos,
    Linux = linux_os,
};

enum class halda_accel_kind : uint8_t {
    none = 0,
    metal = 1,
    cuda = 2,
    None = none,
    Metal = metal,
    CUDA = cuda,
};

enum class halda_set : uint8_t {
    m1 = 1,
    m2 = 2,
    m3 = 3,
    m4 = 4,
    M1 = m1,
    M2 = m2,
    M3 = m3,
    M4 = m4,
};

using halda_os = halda_os_kind;
using halda_accel = halda_accel_kind;

struct halda_profile {
    halda_os_kind os = halda_os_kind::none;
    halda_accel_kind accel = halda_accel_kind::none;
    uint64_t free_bytes = 0;
    uint64_t total_bytes = 0;
    uint64_t host_free_bytes = 0;
    uint64_t host_total_bytes = 0;
    std::vector<double> cpu_gflops;
    std::vector<double> accel_gflops;
    double mem_copy_delay_ms = 0.0;
    double accel_copy_delay_ms = 0.0;
    double disk_read_seq_gbps = 0.0;
    double disk_read_rnd_gbps = 0.0;
    uint32_t n_cpu_threads = 0;
};

using halda_device_profile = halda_profile;
struct halda_model {
    uint32_t n_layer = 0;
    uint32_t n_embd = 0;
    uint32_t n_ff = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t n_vocab = 0;
    uint32_t n_ctx = 0;
    uint32_t n_ubatch = 0;
    uint64_t b = 0;
    uint64_t bi = 0;
    uint64_t bo = 0;
    uint64_t kv_per_layer = 0;
    std::vector<uint32_t> types;
    std::vector<double> layer_flops;
};

struct halda_device {
    uint32_t original_rank = 0;
    std::string name;
    halda_profile profile;
    bool forced_m4 = false;
    bool head = false;
};

struct halda_options {
    halda_model model;
    std::vector<halda_device> devices;
    std::vector<uint32_t> fixed_w;
    int32_t k_override = -1;
    double master_priority = 1.01;
    bool head_participates = true;
};

struct halda_solution {
    uint32_t n_layer = 0;
    uint32_t k = 0;
    uint32_t W = 0;
    double objective = 0.0;

    std::vector<uint32_t> active_original_ranks;
    std::vector<uint32_t> active_ranks;
    std::vector<uint32_t> original_ranks;
    std::vector<uint32_t> w;
    std::vector<uint32_t> n;
    std::vector<halda_set> sets;
    std::vector<std::string> set_labels;

    std::vector<uint32_t> original_w;
    std::vector<uint32_t> original_rank_w;
    std::vector<uint32_t> original_rank_n;
    std::vector<uint32_t> original_n;
    std::vector<halda_set> original_sets;
    std::vector<std::string> original_set_labels;
    std::vector<int32_t> removal_mapping;
    std::vector<int32_t> original_to_active;
    std::vector<uint32_t> removed_original_ranks;
};

struct halda_route_window {
    uint32_t owner = 0;
    uint32_t start = 0;
    uint32_t end = 0;
    int32_t n_gpu_layers = 0;
};
using halda_window = halda_route_window;
using halda_route = std::vector<halda_route_window>;

bool solve_halda(const halda_options & options, halda_solution & solution,
                 std::string & error);
halda_solution solve_halda(const halda_options & options, std::string & error);

bool build_halda_route(const halda_solution & solution,
                       const std::vector<halda_device> & devices,
                       std::vector<halda_route_window> & route,
                       std::string & error);
std::vector<halda_route_window> build_halda_route(
    const halda_solution & solution,
    const std::vector<halda_device> & devices,
    std::string & error);
bool build_halda_route(const halda_solution & solution,
                       std::vector<halda_route_window> & route,
                       std::string & error);
std::vector<halda_route_window> build_halda_route(const halda_solution & solution,
                                                  std::string & error);
