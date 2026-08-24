#include "halda.h"

#include "Highs.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>

namespace {

constexpr double kEpsilon = 1e-9;
constexpr long double kGigabyte = 1000000000.0L;
constexpr uint64_t kCudaContextReserve = 700ull * 1024ull * 1024ull;
constexpr uint64_t kMetalContextReserve = 300ull * 1024ull * 1024ull;
constexpr double kMinDiskGbps = 0.1;

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool checked_mul(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool finite_nonnegative(double value) {
    return std::isfinite(value) && value >= 0.0;
}

bool finite_positive(double value) {
    return std::isfinite(value) && value > 0.0;
}

const char * set_label(halda_set set) {
    switch (set) {
    case halda_set::m1: return "M1";
    case halda_set::m2: return "M2";
    case halda_set::m3: return "M3";
    case halda_set::m4: return "M4";
    }
    return "";
}

bool is_accel(const halda_device & device) {
    return device.profile.accel == halda_accel_kind::metal ||
           device.profile.accel == halda_accel_kind::cuda;
}

bool is_metal(const halda_device & device) {
    return device.profile.accel == halda_accel_kind::metal;
}

bool is_macos(const halda_device & device) {
    return device.profile.os == halda_os_kind::macos;
}

bool is_linux(const halda_device & device) {
    return device.profile.os == halda_os_kind::linux_os;
}

bool device_is_head(const halda_options & options, size_t index) {
    return options.head_participates && index == 0;
}

bool validate_model(const halda_model & model, std::string & error) {
    if (model.n_layer < 2) {
        error = "HALDA requires at least two model layers";
        return false;
    }
    if (model.n_embd == 0 || model.n_ff == 0 || model.n_head == 0 ||
        model.n_vocab == 0 || model.n_ctx == 0 || model.n_ubatch == 0) {
        error = "HALDA model dimensions must be nonzero";
        return false;
    }
    if (model.b == 0) {
        error = "HALDA model layer bytes must be nonzero";
        return false;
    }
    uint64_t b_prime = 0;
    if (!checked_add(model.b, model.kv_per_layer, b_prime) || b_prime == 0) {
        error = "HALDA model layer bytes overflow";
        return false;
    }
    if (model.layer_flops.empty()) {
        error = "HALDA model is missing per-type FLOPs";
        return false;
    }
    if (!model.types.empty() && model.types.size() != model.layer_flops.size()) {
        error = "HALDA model tensor types and FLOPs differ in size";
        return false;
    }
    for (double flops : model.layer_flops) {
        if (!finite_nonnegative(flops)) {
            error = "HALDA model FLOPs are invalid";
            return false;
        }
    }
    uint64_t denominator = 0;
    if (!checked_mul(static_cast<uint64_t>(model.n_layer), b_prime, denominator) ||
        denominator == 0) {
        error = "HALDA model capacity denominator overflow";
        return false;
    }
    return true;
}

bool validate_profile(const halda_model & model, const halda_device & device,
                      std::string & error) {
    const halda_profile & profile = device.profile;
    if (profile.os != halda_os_kind::macos && profile.os != halda_os_kind::linux_os) {
        error = "HALDA profile has an unsupported operating system";
        return false;
    }
    if (profile.accel != halda_accel_kind::none &&
        profile.accel != halda_accel_kind::metal &&
        profile.accel != halda_accel_kind::cuda) {
        error = "HALDA profile has an unsupported accelerator";
        return false;
    }
    if (profile.accel == halda_accel_kind::metal && profile.os != halda_os_kind::macos) {
        error = "HALDA Metal profile must be macOS";
        return false;
    }
    if (profile.accel == halda_accel_kind::cuda && profile.os != halda_os_kind::linux_os) {
        error = "HALDA CUDA profile must be Linux";
        return false;
    }
    if (profile.host_free_bytes == 0) {
        error = "HALDA profile has no free host memory";
        return false;
    }
    if (is_accel(device) && profile.free_bytes == 0) {
        error = "HALDA accelerator profile has no free memory";
        return false;
    }
    if (profile.cpu_gflops.size() != model.layer_flops.size()) {
        error = "HALDA profile CPU FLOPs do not match the model";
        return false;
    }
    if (is_accel(device) && profile.accel_gflops.size() != model.layer_flops.size()) {
        error = "HALDA profile accelerator FLOPs do not match the model";
        return false;
    }
    for (double value : profile.cpu_gflops) {
        if (!finite_positive(value)) {
            error = "HALDA profile CPU FLOPs are invalid";
            return false;
        }
    }
    for (double value : profile.accel_gflops) {
        if (!finite_positive(value)) {
            error = "HALDA profile accelerator FLOPs are invalid";
            return false;
        }
    }
    if (!finite_nonnegative(profile.mem_copy_delay_ms) ||
        !finite_nonnegative(profile.accel_copy_delay_ms)) {
        error = "HALDA profile copy delays are invalid";
        return false;
    }
    if (!finite_nonnegative(profile.disk_read_seq_gbps) ||
        !finite_nonnegative(profile.disk_read_rnd_gbps)) {
        error = "HALDA profile disk bandwidth is invalid";
        return false;
    }
    const double selected_disk = profile.os == halda_os_kind::linux_os
        ? profile.disk_read_seq_gbps : profile.disk_read_rnd_gbps;
    if (!finite_positive(selected_disk)) {
        error = "HALDA selected disk bandwidth is invalid";
        return false;
    }
    return true;
}

bool validate_options(const halda_options & options, std::string & error) {
    if (!validate_model(options.model, error)) {
        return false;
    }
    if (options.devices.empty()) {
        error = "HALDA requires at least one device";
        return false;
    }
    if (options.devices.size() > options.model.n_layer) {
        error = "HALDA has more devices than model layers";
        return false;
    }
    if (!finite_positive(options.master_priority)) {
        error = "HALDA master priority is invalid";
        return false;
    }
    if (options.k_override == 0 || options.k_override < -1 ||
        (options.k_override > static_cast<int32_t>(options.model.n_layer / 2))) {
        error = "HALDA cycle override is invalid";
        return false;
    }
    if (options.k_override > 0 && options.model.n_layer % options.k_override != 0) {
        error = "HALDA cycle override must divide the layer count";
        return false;
    }
    if (!options.fixed_w.empty()) {
        if (options.fixed_w.size() > 32) {
            error = "HALDA fixed window widths accept at most 32 entries";
            return false;
        }
        if (options.fixed_w.size() != options.devices.size()) {
            error = "HALDA fixed window widths must have one entry per device";
            return false;
        }
        uint64_t fixed_width = 0;
        for (const uint32_t width : options.fixed_w) {
            if (width == 0) {
                error = "HALDA fixed window widths must be positive";
                return false;
            }
            if (!checked_add(fixed_width, width, fixed_width)) {
                error = "HALDA fixed window width sum overflow";
                return false;
            }
        }
        if (fixed_width <= 1 || options.model.n_layer % fixed_width != 0) {
            error = "HALDA fixed window widths must sum to more than one and divide the layer count";
            return false;
        }
        const uint32_t fixed_k = static_cast<uint32_t>(
            options.model.n_layer / fixed_width);
        if (options.k_override > 0 &&
            static_cast<uint32_t>(options.k_override) != fixed_k) {
            error = "HALDA cycle override conflicts with fixed window widths";
            return false;
        }
    }
    for (size_t i = 0; i < options.devices.size(); ++i) {
        if (!validate_profile(options.model, options.devices[i], error)) {
            std::ostringstream message;
            message << "device " << i << ": " << error;
            error = message.str();
            return false;
        }
        for (size_t j = 0; j < i; ++j) {
            if (options.devices[i].original_rank == options.devices[j].original_rank) {
                error = "HALDA device original ranks must be unique";
                return false;
            }
        }
    }
    return true;
}

bool compute_buffers(const halda_options & options,
                     const std::vector<halda_device> & devices,
                     const std::vector<uint32_t> & n,
                     std::vector<uint64_t> & c_cpu,
                     std::vector<uint64_t> & c_gpu,
                     std::string & error) {
    const halda_model & model = options.model;
    uint64_t act_terms = 0;
    uint64_t term = 0;
    if (!checked_mul(2, model.n_embd, term) ||
        !checked_add(term, static_cast<uint64_t>(model.n_ff) * 2, act_terms) ||
        !checked_add(act_terms, static_cast<uint64_t>(model.n_embd) * 2, act_terms) ||
        !checked_mul(act_terms, model.n_ubatch, act_terms)) {
        error = "HALDA activation buffer arithmetic overflow";
        return false;
    }
    uint64_t attention = 0;
    uint64_t heads = 0;
    if (!checked_add(1, model.n_head, heads) ||
        !checked_mul(model.n_ctx, model.n_ubatch, attention) ||
        !checked_mul(attention, heads, attention)) {
        error = "HALDA attention buffer arithmetic overflow";
        return false;
    }
    uint64_t positions = 0;
    if (!checked_mul(3, model.n_ubatch, positions) ||
        !checked_add(act_terms, attention, act_terms) ||
        !checked_add(act_terms, positions, act_terms) ||
        !checked_mul(act_terms, 4, act_terms)) {
        error = "HALDA activation buffer arithmetic overflow";
        return false;
    }
    uint64_t head_values = 0;
    uint64_t head_bytes = 0;
    if (!checked_add(model.n_embd, model.n_vocab, head_values) ||
        !checked_mul(head_values, model.n_ubatch, head_bytes) ||
        !checked_mul(head_bytes, 4, head_bytes)) {
        error = "HALDA head buffer arithmetic overflow";
        return false;
    }

    c_cpu.assign(devices.size(), 0);
    c_gpu.assign(devices.size(), 0);
    for (size_t i = 0; i < devices.size(); ++i) {
        c_cpu[i] = act_terms;
        if (device_is_head(options, i) && !checked_add(c_cpu[i], head_bytes, c_cpu[i])) {
            error = "HALDA CPU buffer arithmetic overflow";
            return false;
        }
        if (n[i] > 0) {
            const uint64_t reserve = is_metal(devices[i])
                ? kMetalContextReserve : kCudaContextReserve;
            if (!checked_add(act_terms, reserve, c_gpu[i])) {
                error = "HALDA accelerator buffer arithmetic overflow";
                return false;
            }
        }
    }
    return true;
}

uint64_t memory_budget(const halda_device & device) {
    if (is_macos(device) && is_metal(device)) {
        return device.profile.free_bytes;
    }
    return device.profile.host_free_bytes;
}

bool exceeds_budget(uint64_t first, uint64_t second, uint64_t third,
                    uint64_t fourth, uint64_t budget) {
    uint64_t sum = 0;
    if (!checked_add(first, second, sum) || !checked_add(sum, third, sum) ||
        !checked_add(sum, fourth, sum)) {
        return true;
    }
    return sum > budget;
}

struct sets_state {
    std::vector<halda_set> labels;
    std::vector<size_t> m1;
    std::vector<size_t> m2;
    std::vector<size_t> m3;
    std::vector<size_t> m4;
};

bool classify_sets(const halda_options & options,
                   const std::vector<halda_device> & devices,
                   const std::vector<uint32_t> & w,
                   const std::vector<uint32_t> & n,
                   const std::vector<uint64_t> & c_cpu,
                   const std::vector<uint64_t> & c_gpu,
                   const std::vector<bool> & forced_m4,
                   sets_state & state,
                   std::string & error) {
    const halda_model & model = options.model;
    const uint64_t width = std::accumulate(w.begin(), w.end(), uint64_t{0});
    if (width <= 1 || model.n_layer % width != 0) {
        error = "HALDA allocation does not divide the model layers";
        return false;
    }
    const uint64_t k = model.n_layer / width;
    uint64_t b_prime = 0;
    if (!checked_add(model.b, model.kv_per_layer, b_prime)) {
        error = "HALDA layer byte arithmetic overflow";
        return false;
    }
    uint64_t io_bytes = 0;
    if (!checked_add(model.bi / model.n_vocab, model.bo, io_bytes)) {
        error = "HALDA input/output byte arithmetic overflow";
        return false;
    }
    state.labels.assign(devices.size(), halda_set::m4);
    state.m1.clear();
    state.m2.clear();
    state.m3.clear();
    state.m4.clear();
    for (size_t m = 0; m < devices.size(); ++m) {
        uint64_t layers = 0;
        uint64_t gpu_layers = 0;
        if (!checked_mul(w[m], k, layers) || !checked_mul(n[m], k, gpu_layers) ||
            gpu_layers > layers) {
            error = "HALDA layer arithmetic overflow";
            return false;
        }
        uint64_t fixed = device_is_head(options, m) ? io_bytes : 0;
        const uint64_t budget = memory_budget(devices[m]);
        uint64_t weight_bytes = 0;
        uint64_t kv_bytes = 0;
        if (!checked_mul(layers, model.b, weight_bytes) ||
            !checked_mul(layers, model.kv_per_layer, kv_bytes)) {
            error = "HALDA layer memory arithmetic overflow";
            return false;
        }
        const bool condition1 = exceeds_budget(weight_bytes, fixed, kv_bytes,
                                               c_cpu[m], budget);
        uint64_t condition3_weights = 0;
        uint64_t cpu_gpu = 0;
        if (!checked_mul(layers - gpu_layers, b_prime, condition3_weights) ||
            !checked_add(c_cpu[m], c_gpu[m], cpu_gpu)) {
            error = "HALDA layer memory arithmetic overflow";
            return false;
        }
        const bool condition2 = exceeds_budget(weight_bytes, fixed, kv_bytes,
                                               cpu_gpu, budget);
        const bool condition3 = exceeds_budget(condition3_weights, fixed, 0,
                                               c_cpu[m], budget);
        const double disk = is_linux(devices[m])
            ? devices[m].profile.disk_read_seq_gbps
            : devices[m].profile.disk_read_rnd_gbps;
        const bool slow_disk = disk < kMinDiskGbps;

        halda_set set = halda_set::m4;
        if (forced_m4[m] || slow_disk) {
            set = halda_set::m4;
        } else if (is_macos(devices[m]) && !is_metal(devices[m]) && condition1) {
            set = halda_set::m1;
        } else if (is_macos(devices[m]) && is_metal(devices[m]) && condition2) {
            set = halda_set::m2;
        } else if (is_linux(devices[m]) && condition3) {
            set = halda_set::m3;
        }
        state.labels[m] = set;
        switch (set) {
        case halda_set::m1: state.m1.push_back(m); break;
        case halda_set::m2: state.m2.push_back(m); break;
        case halda_set::m3: state.m3.push_back(m); break;
        case halda_set::m4: state.m4.push_back(m); break;
        }
    }
    return true;
}

void fixed_window_memory_error(const halda_options & options,
                               const std::vector<halda_device> & devices,
                               const sets_state & sets,
                               const std::vector<uint64_t> & c_cpu,
                               const std::vector<uint64_t> & c_gpu,
                               uint32_t k,
                               std::string & error) {
    uint64_t b_prime = 0;
    uint64_t io_bytes = 0;
    if (!checked_add(options.model.b, options.model.kv_per_layer, b_prime) ||
        !checked_add(options.model.bi / options.model.n_vocab, options.model.bo, io_bytes)) {
        error = "HALDA fixed window memory diagnostic overflow";
        return;
    }
    size_t selected = 0;
    uint64_t selected_requested = 0;
    uint64_t selected_available = 0;
    bool selected_over = false;
    for (size_t m = 0; m < devices.size(); ++m) {
        uint64_t layers = 0;
        uint64_t requested = 0;
        if (!checked_mul(options.fixed_w[m], k, layers) ||
            !checked_mul(layers, b_prime, requested) ||
            !checked_add(requested, c_cpu[m], requested) ||
            (device_is_head(options, m) && !checked_add(requested, io_bytes, requested))) {
            error = "HALDA fixed window memory diagnostic overflow";
            return;
        }
        const bool needs_gpu_buffer =
            sets.labels[m] == halda_set::m2 ||
            (sets.labels[m] == halda_set::m4 && is_macos(devices[m]) &&
             is_metal(devices[m]));
        if (needs_gpu_buffer && !checked_add(requested, c_gpu[m], requested)) {
            error = "HALDA fixed window memory diagnostic overflow";
            return;
        }
        const uint64_t available = memory_budget(devices[m]);
        const bool over = requested > available;
        const bool larger_deficit = over && selected_over &&
            requested - available > selected_requested - selected_available;
        if ((over && !selected_over) || larger_deficit ||
            (!selected_over && !over && requested > selected_requested)) {
            selected = m;
            selected_requested = requested;
            selected_available = available;
            selected_over = over;
        }
    }
    const halda_device & device = devices[selected];
    const std::string name = device.name.empty()
        ? "device " + std::to_string(device.original_rank) : device.name;
    std::ostringstream message;
    message << "HALDA fixed window is infeasible on " << name
            << ": requested " << selected_requested
            << " bytes, available " << selected_available << " bytes";
    error = message.str();
}

bool fixed_allocation_memory_error(
        const halda_options & options,
        const std::vector<halda_device> & devices,
        const std::vector<uint64_t> & c_cpu,
        const std::vector<uint64_t> & c_gpu,
        uint32_t k,
        const std::vector<uint32_t> & w,
        const std::vector<uint32_t> & n,
        std::string & error) {
    uint64_t b_prime = 0;
    uint64_t io_bytes = 0;
    if (!checked_add(options.model.b, options.model.kv_per_layer, b_prime) ||
        !checked_add(options.model.bi / options.model.n_vocab, options.model.bo, io_bytes)) {
        error = "HALDA fixed window memory diagnostic overflow";
        return true;
    }
    size_t selected = 0;
    uint64_t selected_requested = 0;
    uint64_t selected_available = 0;
    bool selected_over = false;
    auto consider = [&](size_t m, uint64_t requested, uint64_t available) {
        const bool over = requested > available;
        const bool larger_deficit = over && selected_over &&
            requested - available > selected_requested - selected_available;
        if ((over && !selected_over) || larger_deficit ||
            (!selected_over && !over && requested > selected_requested)) {
            selected = m;
            selected_requested = requested;
            selected_available = available;
            selected_over = over;
        }
    };
    for (size_t m = 0; m < devices.size(); ++m) {
        uint64_t layers = 0;
        uint64_t gpu_layers = 0;
        uint64_t weight_bytes = 0;
        uint64_t kv_bytes = 0;
        if (!checked_mul(k, w[m], layers) ||
            !checked_mul(k, n[m], gpu_layers) ||
            gpu_layers > layers ||
            !checked_mul(layers, options.model.b, weight_bytes) ||
            !checked_mul(layers, options.model.kv_per_layer, kv_bytes)) {
            error = "HALDA fixed window memory diagnostic overflow";
            return true;
        }
        uint64_t fixed = device_is_head(options, m) ? io_bytes : 0;
        if (is_metal(devices[m])) {
            uint64_t requested = 0;
            if (!checked_add(weight_bytes, kv_bytes, requested) ||
                !checked_add(requested, c_cpu[m], requested) ||
                !checked_add(requested, c_gpu[m], requested) ||
                !checked_add(requested, fixed, requested)) {
                error = "HALDA fixed window memory diagnostic overflow";
                return true;
            }
            consider(m, requested, devices[m].profile.free_bytes);
        } else if (is_accel(devices[m])) {
            uint64_t cpu_layers = layers - gpu_layers;
            uint64_t cpu_layer_bytes = 0;
            uint64_t host_requested = 0;
            uint64_t gpu_requested = 0;
            if (!checked_mul(cpu_layers, b_prime, cpu_layer_bytes) ||
                !checked_add(cpu_layer_bytes, c_cpu[m], host_requested) ||
                !checked_add(host_requested, fixed, host_requested) ||
                !checked_mul(gpu_layers, b_prime, gpu_requested) ||
                !checked_add(gpu_requested, c_gpu[m], gpu_requested)) {
                error = "HALDA fixed window memory diagnostic overflow";
                return true;
            }
            consider(m, host_requested, devices[m].profile.host_free_bytes);
            consider(m, gpu_requested, devices[m].profile.free_bytes);
        } else {
            uint64_t requested = 0;
            if (!checked_add(weight_bytes, kv_bytes, requested) ||
                !checked_add(requested, c_cpu[m], requested) ||
                !checked_add(requested, fixed, requested)) {
                error = "HALDA fixed window memory diagnostic overflow";
                return true;
            }
            consider(m, requested, devices[m].profile.host_free_bytes);
        }
    }
    if (!selected_over) {
        return false;
    }
    const halda_device & device = devices[selected];
    const std::string name = device.name.empty()
        ? "device " + std::to_string(device.original_rank) : device.name;
    std::ostringstream message;
    message << "HALDA fixed window is infeasible on " << name
            << ": requested " << selected_requested
            << " bytes, available " << selected_available << " bytes";
    error = message.str();
    return true;
}

long double disk_ms(long double bytes, double gbps) {
    return bytes / (static_cast<long double>(gbps) * kGigabyte) * 1000.0L;
}

bool flop_time(const std::vector<double> & flops, const std::vector<double> & speed,
               double & result) {
    long double total = 0.0L;
    for (size_t i = 0; i < flops.size(); ++i) {
        total += static_cast<long double>(flops[i]) /
                 (static_cast<long double>(speed[i]) * kGigabyte + kEpsilon);
    }
    total *= 1000.0L;
    result = static_cast<double>(total);
    return std::isfinite(result);
}

struct coefficients {
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> z;
    std::vector<double> z_gpu;
    double kappa = 0.0;
};

bool make_coefficients(const halda_options & options,
                       const std::vector<halda_device> & devices,
                       const std::vector<uint64_t> & c_cpu,
                       const std::vector<uint64_t> & c_gpu,
                       const sets_state & sets,
                       coefficients & result,
                       std::string & error) {
    const halda_model & model = options.model;
    const size_t count = devices.size();
    uint64_t b_prime = 0;
    uint64_t denominator_bytes = 0;
    if (!checked_add(model.b, model.kv_per_layer, b_prime) ||
        !checked_mul(model.n_layer, b_prime, denominator_bytes) ||
        denominator_bytes == 0) {
        error = "HALDA capacity denominator overflow";
        return false;
    }
    uint64_t io_bytes = 0;
    if (!checked_add(model.bi / model.n_vocab, model.bo, io_bytes)) {
        error = "HALDA input/output byte arithmetic overflow";
        return false;
    }
    const long double denominator = static_cast<long double>(denominator_bytes);
    result.a.assign(count, 0.0);
    result.b.assign(count, 0.0);
    result.z.assign(count, 0.0);
    result.z_gpu.assign(count, 0.0);
    std::vector<double> t_cpu(count, 0.0);
    std::vector<double> t_accel(count, 0.0);
    for (size_t m = 0; m < count; ++m) {
        if (!flop_time(model.layer_flops, devices[m].profile.cpu_gflops, t_cpu[m])) {
            error = "HALDA CPU coefficient is invalid";
            return false;
        }
        if (is_accel(devices[m]) &&
            !flop_time(model.layer_flops, devices[m].profile.accel_gflops,
                       t_accel[m])) {
            error = "HALDA accelerator coefficient is invalid";
            return false;
        }
        const double alpha = t_cpu[m] + devices[m].profile.mem_copy_delay_ms;
        const double beta = t_accel[m] - t_cpu[m] +
            devices[m].profile.accel_copy_delay_ms -
            devices[m].profile.mem_copy_delay_ms;
        const double disk = is_linux(devices[m])
            ? devices[m].profile.disk_read_seq_gbps
            : devices[m].profile.disk_read_rnd_gbps;
        const long double read_b = disk_ms(static_cast<long double>(model.b), disk);
        const long double read_b_prime = disk_ms(static_cast<long double>(b_prime), disk);
        if (sets.labels[m] == halda_set::m1) {
            result.a[m] = alpha + static_cast<double>(read_b_prime);
        } else if (sets.labels[m] == halda_set::m2) {
            result.a[m] = alpha + static_cast<double>(read_b);
            result.b[m] = beta;
        } else if (sets.labels[m] == halda_set::m3) {
            result.a[m] = alpha + static_cast<double>(read_b_prime);
            if (is_accel(devices[m])) {
                result.b[m] = beta - static_cast<double>(read_b_prime);
            }
        } else {
            result.a[m] = alpha;
            if (is_accel(devices[m])) {
                result.b[m] = beta;
            }
        }
        if (is_accel(devices[m])) {
            const long double available = static_cast<long double>(devices[m].profile.free_bytes) -
                                           static_cast<long double>(c_gpu[m]);
            result.z_gpu[m] = std::max(static_cast<double>(available / denominator), 0.0);
        }
        const long double fixed = device_is_head(options, m)
            ? static_cast<long double>(io_bytes) : 0.0L;
        long double capacity = 0.0L;
        if (sets.labels[m] == halda_set::m1 || sets.labels[m] == halda_set::m3) {
            capacity = static_cast<long double>(devices[m].profile.host_free_bytes) -
                       static_cast<long double>(c_cpu[m]) - fixed;
        } else if (sets.labels[m] == halda_set::m2 ||
                   (sets.labels[m] == halda_set::m4 && is_macos(devices[m]) && is_metal(devices[m]))) {
            capacity = static_cast<long double>(devices[m].profile.free_bytes) -
                       static_cast<long double>(c_cpu[m]) -
                       static_cast<long double>(c_gpu[m]) - fixed;
        } else {
            capacity = static_cast<long double>(devices[m].profile.host_free_bytes) -
                       static_cast<long double>(c_cpu[m]) - fixed;
        }
        result.z[m] = sets.labels[m] == halda_set::m4
            ? static_cast<double>(-capacity / denominator)
            : static_cast<double>(capacity / denominator);
        if (!finite_nonnegative(result.a[m]) || !std::isfinite(result.b[m]) ||
            !std::isfinite(result.z[m]) || !std::isfinite(result.z_gpu[m])) {
            error = "HALDA coefficient is invalid";
            return false;
        }
    }

    result.kappa = 0.0;
    if (options.head_participates) {
        const double disk0 = is_linux(devices[0])
            ? devices[0].profile.disk_read_seq_gbps
            : devices[0].profile.disk_read_rnd_gbps;
        result.kappa = t_cpu[0] + static_cast<double>(disk_ms(
            static_cast<long double>(model.bi / model.n_vocab), disk0));
        if (sets.labels[0] != halda_set::m4) {
            result.kappa += static_cast<double>(disk_ms(static_cast<long double>(model.bo), disk0));
        }
    }
    for (size_t m : sets.m1) {
        const double disk = is_linux(devices[m])
            ? devices[m].profile.disk_read_seq_gbps
            : devices[m].profile.disk_read_rnd_gbps;
        result.kappa += static_cast<double>(disk_ms(
            static_cast<long double>(c_cpu[m]) -
                static_cast<long double>(devices[m].profile.host_free_bytes), disk));
    }
    for (size_t m : sets.m3) {
        const double disk = is_linux(devices[m])
            ? devices[m].profile.disk_read_seq_gbps
            : devices[m].profile.disk_read_rnd_gbps;
        result.kappa += static_cast<double>(disk_ms(
            static_cast<long double>(c_cpu[m]) -
                static_cast<long double>(devices[m].profile.host_free_bytes), disk));
    }
    if (!std::isfinite(result.kappa)) {
        error = "HALDA fixed objective is invalid";
        return false;
    }
    return true;
}

struct mip_solution {
    bool valid = false;
    uint32_t k = 0;
    uint32_t W = 0;
    double objective = 0.0;
    std::vector<uint32_t> w;
    std::vector<uint32_t> n;
};

bool all_m4(const sets_state & sets) {
    return sets.m1.empty() && sets.m2.empty() && sets.m3.empty();
}

bool solve_for_k(const halda_options & options,
                 const std::vector<halda_device> & devices,
                 const sets_state & sets,
                 const coefficients & coeff,
                 uint32_t k,
                 mip_solution & result) {
    if (k == 0 || options.model.n_layer % k != 0) {
        return false;
    }
    const size_t count = devices.size();
    const bool fixed_width = !options.fixed_w.empty();
    uint64_t requested_width = 0;
    if (fixed_width) {
        if (options.fixed_w.size() != count) {
            return false;
        }
        requested_width = std::accumulate(
            options.fixed_w.begin(), options.fixed_w.end(), uint64_t{0});
        if (requested_width <= 1 ||
            options.model.n_layer % requested_width != 0 ||
            k != options.model.n_layer / requested_width) {
            return false;
        }
    }
    const uint32_t W = options.model.n_layer / k;
    const HighsInt columns = static_cast<HighsInt>(count * 2);
    const HighsInt rows = static_cast<HighsInt>(1 + count * 3);
    HighsModel model;
    model.lp_.num_col_ = columns;
    model.lp_.num_row_ = rows;
    model.lp_.sense_ = ObjSense::kMinimize;
    model.lp_.offset_ = coeff.kappa;
    model.lp_.col_cost_.assign(columns, 0.0);
    for (size_t m = 0; m < count; ++m) {
        model.lp_.col_cost_[m] = coeff.a[m] * k;
        model.lp_.col_cost_[m + count] = coeff.b[m] * k;
    }
    if (options.head_participates && !model.lp_.col_cost_.empty()) {
        model.lp_.col_cost_[0] /= options.master_priority;
    }
    model.lp_.col_lower_.assign(columns, 0.0);
    model.lp_.col_upper_.assign(columns, static_cast<double>(options.model.n_layer));
    if (fixed_width) {
        for (size_t m = 0; m < count; ++m) {
            model.lp_.col_lower_[m] = static_cast<double>(options.fixed_w[m]);
            model.lp_.col_upper_[m] = static_cast<double>(options.fixed_w[m]);
        }
    } else {
        std::fill(model.lp_.col_lower_.begin(), model.lp_.col_lower_.begin() + count, 1.0);
    }
    model.lp_.row_lower_.assign(rows, -1.0e30);
    model.lp_.row_upper_.assign(rows, 1.0e30);
    model.lp_.row_lower_[0] = static_cast<double>(W);
    model.lp_.row_upper_[0] = static_cast<double>(W);
    for (size_t m = 0; m < count; ++m) {
        model.lp_.row_upper_[1 + m] = 0.0;
        model.lp_.row_upper_[1 + count + m] = -static_cast<double>(W) * coeff.z[m];
        model.lp_.row_upper_[1 + count * 2 + m] = std::max(
            static_cast<double>(W) * coeff.z_gpu[m], 0.0);
    }
    model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    model.lp_.a_matrix_.num_col_ = columns;
    model.lp_.a_matrix_.num_row_ = rows;
    model.lp_.a_matrix_.start_.assign(columns + 1, 0);
    for (size_t col = 0; col < count * 2; ++col) {
        model.lp_.a_matrix_.start_[col] =
            static_cast<HighsInt>(model.lp_.a_matrix_.index_.size());
        if (col < count) {
            const size_t m = col;
            model.lp_.a_matrix_.index_.push_back(0);
            model.lp_.a_matrix_.value_.push_back(1.0);
            model.lp_.a_matrix_.index_.push_back(static_cast<HighsInt>(1 + m));
            model.lp_.a_matrix_.value_.push_back(-1.0);
            if (sets.labels[m] == halda_set::m1 || sets.labels[m] == halda_set::m2 ||
                sets.labels[m] == halda_set::m3) {
                model.lp_.a_matrix_.index_.push_back(static_cast<HighsInt>(1 + count + m));
                model.lp_.a_matrix_.value_.push_back(-1.0);
            } else {
                model.lp_.a_matrix_.index_.push_back(static_cast<HighsInt>(1 + count + m));
                model.lp_.a_matrix_.value_.push_back(1.0);
            }
        } else {
            const size_t m = col - count;
            model.lp_.a_matrix_.index_.push_back(static_cast<HighsInt>(1 + m));
            model.lp_.a_matrix_.value_.push_back(1.0);
            if (sets.labels[m] == halda_set::m3 && is_accel(devices[m])) {
                model.lp_.a_matrix_.index_.push_back(static_cast<HighsInt>(1 + count + m));
                model.lp_.a_matrix_.value_.push_back(1.0);
            } else if (sets.labels[m] == halda_set::m4 &&
                       !is_macos(devices[m]) && is_accel(devices[m])) {
                model.lp_.a_matrix_.index_.push_back(static_cast<HighsInt>(1 + count + m));
                model.lp_.a_matrix_.value_.push_back(-1.0);
            }
            model.lp_.a_matrix_.index_.push_back(static_cast<HighsInt>(1 + count * 2 + m));
            model.lp_.a_matrix_.value_.push_back(1.0);
        }
    }
    model.lp_.a_matrix_.start_[columns] =
        static_cast<HighsInt>(model.lp_.a_matrix_.index_.size());
    model.lp_.integrality_.assign(columns, HighsVarType::kInteger);

    try {
        Highs highs;
        highs.setOptionValue("log_to_console", false);
        highs.setOptionValue("threads", 1);
        highs.setOptionValue("mip_rel_gap", 0.0);
        highs.setOptionValue("random_seed", 0);
        if (highs.passModel(model) != HighsStatus::kOk ||
            highs.run() != HighsStatus::kOk) {
            return false;
        }
        const bool acceptable_status = highs.getModelStatus() == HighsModelStatus::kOptimal ||
                                       (all_m4(sets) &&
                                        highs.getModelStatus() != HighsModelStatus::kInfeasible);
        if (!acceptable_status) {
            return false;
        }
        const HighsSolution & solution = highs.getSolution();
        if (!solution.value_valid || solution.col_value.size() != count * 2) {
            return false;
        }
        result.w.assign(count, 0);
        result.n.assign(count, 0);
        for (size_t m = 0; m < count * 2; ++m) {
            const double value = solution.col_value[m];
            if (!std::isfinite(value)) {
                return false;
            }
            const double rounded = std::round(value);
            if (std::abs(value - rounded) > 1e-6 || rounded < 0.0 ||
                rounded > static_cast<double>(options.model.n_layer)) {
                return false;
            }
            if (m < count) {
                result.w[m] = static_cast<uint32_t>(rounded);
            } else {
                result.n[m - count] = static_cast<uint32_t>(rounded);
            }
        }
        uint64_t sum = 0;
        for (size_t m = 0; m < count; ++m) {
            if (result.w[m] == 0 || result.n[m] > result.w[m] ||
                !checked_add(sum, result.w[m], sum)) {
                return false;
            }
        }
        if (sum != W) {
            return false;
        }
        result.valid = true;
        result.k = k;
        result.W = W;
        result.objective = highs.getInfo().objective_function_value;
        return std::isfinite(result.objective);
    } catch (...) {
        return false;
    }
}

bool seed_allocation(const halda_options & options,
                     const std::vector<halda_device> & devices,
                     std::vector<uint32_t> & w,
                     std::vector<uint32_t> & n,
                     std::string & error) {
    if (!options.fixed_w.empty()) {
        if (options.fixed_w.size() != devices.size()) {
            error = "HALDA fixed window widths do not match the active device set";
            return false;
        }
        w = options.fixed_w;
        n.assign(devices.size(), 0);
        for (size_t m = 0; m < devices.size(); ++m) {
            if (is_accel(devices[m])) {
                n[m] = w[m];
            }
        }
        return true;
    }
    uint64_t total = 0;
    std::vector<uint64_t> budget(devices.size(), 0);
    for (size_t m = 0; m < devices.size(); ++m) {
        budget[m] = memory_budget(devices[m]);
        if (!checked_add(total, budget[m], total)) {
            error = "HALDA memory budget arithmetic overflow";
            return false;
        }
    }
    if (total == 0) {
        error = "HALDA devices have no usable memory";
        return false;
    }
    w.assign(devices.size(), 0);
    for (size_t m = 0; m < devices.size(); ++m) {
        const double value = static_cast<double>(budget[m]) /
                             static_cast<double>(total) * options.model.n_layer;
        if (!std::isfinite(value) || value < 0.0 ||
            value > static_cast<double>(options.model.n_layer)) {
            error = "HALDA memory seed is invalid";
            return false;
        }
        w[m] = static_cast<uint32_t>(std::round(value));
    }
    for (size_t m = 0; m < w.size(); ++m) {
        if (w[m] == 0) {
            w[m] = 1;
            size_t largest = 0;
            for (size_t i = 1; i < w.size(); ++i) {
                if (w[i] > w[largest]) {
                    largest = i;
                }
            }
            if (w[largest] <= 1) {
                error = "HALDA cannot give every device a window";
                return false;
            }
            --w[largest];
        }
    }
    uint64_t sum = std::accumulate(w.begin(), w.end(), uint64_t{0});
    const int64_t diff = static_cast<int64_t>(options.model.n_layer) -
                         static_cast<int64_t>(sum);
    size_t adjust = 0;
    if (diff > 0) {
        adjust = static_cast<size_t>(std::distance(
            budget.begin(), std::max_element(budget.begin(), budget.end())));
    } else if (diff < 0) {
        adjust = static_cast<size_t>(std::distance(
            budget.begin(), std::min_element(budget.begin(), budget.end())));
    }
    if (diff != 0) {
        const int64_t adjusted = static_cast<int64_t>(w[adjust]) + diff;
        if (adjusted < 1 || adjusted > options.model.n_layer) {
            error = "HALDA memory seed cannot satisfy window bounds";
            return false;
        }
        w[adjust] = static_cast<uint32_t>(adjusted);
    }
    n.assign(devices.size(), 0);
    for (size_t m = 0; m < devices.size(); ++m) {
        if (is_accel(devices[m])) {
            n[m] = w[m];
        }
    }
    return true;
}

struct fixed_result {
    uint32_t k = 0;
    uint32_t W = 0;
    double objective = 0.0;
    std::vector<uint32_t> w;
    std::vector<uint32_t> n;
    sets_state sets;
};

bool solve_active(const halda_options & options,
                  const std::vector<halda_device> & devices,
                  fixed_result & result,
                  std::string & error) {
    std::vector<uint32_t> w;
    std::vector<uint32_t> n;
    if (!seed_allocation(options, devices, w, n, error)) {
        return false;
    }
    std::vector<bool> forced_m4(devices.size(), false);
    for (size_t m = 0; m < devices.size(); ++m) {
        forced_m4[m] = devices[m].forced_m4;
    }
    sets_state previous_sets;
    fixed_result last;
    bool have_last = false;
    for (size_t iteration = 0; iteration < 128; ++iteration) {
        const uint64_t width = std::accumulate(w.begin(), w.end(), uint64_t{0});
        if (width <= 1 || width > options.model.n_layer ||
            options.model.n_layer % width != 0) {
            error = "HALDA requires L = k * W with W greater than one";
            return false;
        }
        std::vector<uint64_t> c_cpu;
        std::vector<uint64_t> c_gpu;
        if (!compute_buffers(options, devices, n, c_cpu, c_gpu, error)) {
            return false;
        }
        sets_state sets;
        if (!classify_sets(options, devices, w, n, c_cpu, c_gpu, forced_m4,
                           sets, error)) {
            return false;
        }
        if (!previous_sets.labels.empty() && previous_sets.labels == sets.labels) {
            if (!have_last) {
                error = "HALDA set classification stabilized without a solution";
                return false;
            }
            result = last;
            return true;
        }
        previous_sets = sets;

        coefficients coeff;
        if (!make_coefficients(options, devices, c_cpu, c_gpu, sets, coeff, error)) {
            return false;
        }
        std::vector<uint32_t> candidates;
        if (!options.fixed_w.empty()) {
            const uint64_t fixed_width = std::accumulate(
                options.fixed_w.begin(), options.fixed_w.end(), uint64_t{0});
            if (fixed_width <= 1 ||
                options.model.n_layer % fixed_width != 0) {
                error = "HALDA fixed window width constraint is infeasible";
                return false;
            }
            candidates.push_back(static_cast<uint32_t>(
                options.model.n_layer / fixed_width));
        } else if (options.k_override > 0) {
            candidates.push_back(static_cast<uint32_t>(options.k_override));
        } else {
            for (uint32_t k = 1; k <= options.model.n_layer / 2; ++k) {
                if (options.model.n_layer % k == 0) {
                    candidates.push_back(k);
                }
            }
        }
        mip_solution best;
        for (uint32_t k : candidates) {
            mip_solution candidate;
            if (!solve_for_k(options, devices, sets, coeff, k, candidate)) {
                continue;
            }
            const double tolerance = kEpsilon * std::max(
                1.0, std::max(std::abs(candidate.objective), std::abs(best.objective)));
            // Prefer lower original ranks on objective ties for deterministic Potluck placement.
            if (!best.valid || candidate.objective < best.objective - tolerance ||
                (std::abs(candidate.objective - best.objective) <= tolerance &&
                 (candidate.w > best.w ||
                  (candidate.w == best.w && candidate.k > best.k)))) {
                best = std::move(candidate);
            }
        }
        if (!best.valid) {
            if (!have_last) {
                if (options.fixed_w.empty()) {
                    error = "HALDA found no feasible integer placement";
                } else {
                    fixed_window_memory_error(
                        options, devices, sets, c_cpu, c_gpu,
                        candidates.empty() ? 0 : candidates.front(), error);
                }
                return false;
            }
            result = last;
            return true;
        }
        if (!options.fixed_w.empty() &&
            fixed_allocation_memory_error(
                options, devices, c_cpu, c_gpu, best.k,
                best.w, best.n, error)) {
            return false;
        }

        bool weak = false;
        bool free_gpu = false;
        bool gpu_overload = false;
        bool cpu_overload = false;
        for (size_t m = 0; m < devices.size(); ++m) {
            weak = weak || (best.w[m] == 1 && best.n[m] == 0);
            if (is_accel(devices[m])) {
                const double available = std::floor(
                    static_cast<double>(best.W) * coeff.z_gpu[m]);
                free_gpu = free_gpu || static_cast<double>(best.n[m]) < available;
                gpu_overload = gpu_overload || best.w[m] > best.n[m];
            } else if (sets.labels[m] != halda_set::m4) {
                cpu_overload = true;
            }
        }
        if (!weak && free_gpu && (gpu_overload || cpu_overload)) {
            size_t worst = devices.size();
            double worst_speed = std::numeric_limits<double>::max();
            for (size_t m = 0; m < devices.size(); ++m) {
                if (sets.labels[m] == halda_set::m4) {
                    continue;
                }
                const double disk = is_linux(devices[m])
                    ? devices[m].profile.disk_read_seq_gbps
                    : devices[m].profile.disk_read_rnd_gbps;
                if (disk < worst_speed) {
                    worst_speed = disk;
                    worst = m;
                }
            }
            if (worst != devices.size()) {
                forced_m4[worst] = true;
                continue;
            }
        }

        fixed_result adopted;
        adopted.k = best.k;
        adopted.W = best.W;
        adopted.objective = best.objective;
        adopted.w = best.w;
        adopted.n = best.n;
        adopted.sets = sets;
        const bool unchanged = have_last && last.k == adopted.k &&
                               last.w == adopted.w && last.n == adopted.n;
        last = adopted;
        have_last = true;
        w = best.w;
        n = best.n;
        if (unchanged) {
            result = last;
            return true;
        }
    }
    error = "HALDA fixed point did not converge";
    return false;
}

void fill_solution(const halda_options & options,
                   const std::vector<halda_device> & original_devices,
                   const std::vector<size_t> & active_indices,
                   const fixed_result & fixed,
                   halda_solution & solution) {
    solution = {};
    solution.n_layer = options.model.n_layer;
    solution.k = fixed.k;
    solution.W = fixed.W;
    solution.objective = fixed.objective;
    solution.original_ranks.reserve(original_devices.size());
    solution.original_w.assign(original_devices.size(), 0);
    solution.original_rank_w.assign(original_devices.size(), 0);
    solution.original_n.assign(original_devices.size(), 0);
    solution.original_rank_n.assign(original_devices.size(), 0);
    solution.original_sets.assign(original_devices.size(), halda_set::m4);
    solution.original_set_labels.assign(original_devices.size(), "removed");
    solution.removal_mapping.assign(original_devices.size(), -1);
    solution.original_to_active.assign(original_devices.size(), -1);
    for (const halda_device & device : original_devices) {
        solution.original_ranks.push_back(device.original_rank);
    }
    for (size_t active = 0; active < active_indices.size(); ++active) {
        const size_t original = active_indices[active];
        const uint32_t rank = original_devices[original].original_rank;
        const halda_set set = fixed.sets.labels[active];
        solution.active_ranks.push_back(rank);
        solution.active_original_ranks.push_back(rank);
        solution.w.push_back(fixed.w[active]);
        solution.n.push_back(fixed.n[active]);
        solution.sets.push_back(set);
        solution.set_labels.emplace_back(set_label(set));
        solution.original_w[original] = fixed.w[active];
        solution.original_rank_w[original] = fixed.w[active];
        solution.original_rank_n[original] = fixed.n[active];
        solution.original_n[original] = fixed.n[active];
        solution.original_sets[original] = set;
        solution.original_set_labels[original] = set_label(set);
        solution.removal_mapping[original] = static_cast<int32_t>(active);
        solution.original_to_active[original] = static_cast<int32_t>(active);
    }
    for (size_t original = 0; original < original_devices.size(); ++original) {
        if (solution.removal_mapping[original] < 0) {
            solution.removed_original_ranks.push_back(original_devices[original].original_rank);
        }
    }
}

} // namespace

bool solve_halda(const halda_options & options, halda_solution & solution,
                 std::string & error) {
    solution = {};
    error.clear();
    if (!validate_options(options, error)) {
        return false;
    }
    const std::vector<halda_device> original_devices = options.devices;
    std::vector<size_t> active_indices(original_devices.size());
    std::iota(active_indices.begin(), active_indices.end(), 0);
    if (!options.fixed_w.empty()) {
        fixed_result fixed;
        if (!solve_active(options, original_devices, fixed, error)) {
            return false;
        }
        fill_solution(options, original_devices, active_indices, fixed, solution);
        return true;
    }
    for (size_t removal_iteration = 0; removal_iteration <= original_devices.size();
         ++removal_iteration) {
        halda_options active_options = options;
        active_options.devices.clear();
        for (size_t original : active_indices) {
            active_options.devices.push_back(original_devices[original]);
        }
        fixed_result fixed;
        if (!solve_active(active_options, active_options.devices, fixed, error)) {
            return false;
        }
        std::vector<size_t> next_indices;
        next_indices.reserve(active_indices.size());
        for (size_t active = 0; active < active_indices.size(); ++active) {
            const bool keep = fixed.w[active] != 1 ||
                              original_devices[active_indices[active]].original_rank == 0;
            if (keep) {
                next_indices.push_back(active_indices[active]);
            }
        }
        if (next_indices.size() == active_indices.size()) {
            fill_solution(options, original_devices, active_indices, fixed, solution);
            return true;
        }
        if (next_indices.empty()) {
            error = "HALDA removed every device";
            return false;
        }
        active_indices = std::move(next_indices);
    }
    error = "HALDA device removal did not converge";
    return false;
}

halda_solution solve_halda(const halda_options & options, std::string & error) {
    halda_solution solution;
    solve_halda(options, solution, error);
    return solution;
}

bool build_halda_route(const halda_solution & solution,
                       std::vector<halda_route_window> & route,
                       std::string & error) {
    route.clear();
    error.clear();
    if (solution.n_layer == 0 || solution.k == 0 || solution.W == 0 ||
        solution.active_original_ranks.empty() ||
        solution.active_original_ranks.size() != solution.w.size() ||
        solution.w.size() != solution.n.size()) {
        error = "HALDA route solution is incomplete";
        return false;
    }
    uint64_t width = 0;
    for (size_t i = 0; i < solution.w.size(); ++i) {
        if (solution.w[i] == 0 || solution.n[i] > solution.w[i] ||
            !checked_add(width, solution.w[i], width)) {
            error = "HALDA route contains an invalid window";
            return false;
        }
    }
    if (width != solution.W || static_cast<uint64_t>(solution.k) * width != solution.n_layer) {
        error = "HALDA route does not divide model layers";
        return false;
    }
    if (solution.w.size() != 0 &&
        solution.k > std::numeric_limits<uint64_t>::max() / solution.w.size()) {
        error = "HALDA route is too large";
        return false;
    }
    const uint64_t count = static_cast<uint64_t>(solution.k) * solution.w.size();
    if (count > std::numeric_limits<size_t>::max()) {
        error = "HALDA route is too large";
        return false;
    }
    route.reserve(static_cast<size_t>(count));
    uint64_t next = 0;
    for (uint32_t round = 0; round < solution.k; ++round) {
        for (size_t i = 0; i < solution.w.size(); ++i) {
            uint64_t end = 0;
            if (!checked_add(next, solution.w[i], end) ||
                end > solution.n_layer ||
                solution.n[i] > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
                error = "HALDA route layer coverage overflow";
                route.clear();
                return false;
            }
            route.push_back({solution.active_original_ranks[i],
                             static_cast<uint32_t>(next),
                             static_cast<uint32_t>(end),
                             static_cast<int32_t>(solution.n[i])});
            next = end;
        }
    }
    if (next != solution.n_layer) {
        error = "HALDA route does not cover every model layer";
        route.clear();
        return false;
    }
    return true;
}

std::vector<halda_route_window> build_halda_route(const halda_solution & solution,
                                                  std::string & error) {
    std::vector<halda_route_window> route;
    build_halda_route(solution, route, error);
    return route;
}

bool build_halda_route(const halda_solution & solution,
                       const std::vector<halda_device> & devices,
                       std::vector<halda_route_window> & route,
                       std::string & error) {
    for (uint32_t rank : solution.active_original_ranks) {
        const auto found = std::find_if(
            devices.begin(), devices.end(),
            [rank](const halda_device & device) { return device.original_rank == rank; });
        if (found == devices.end()) {
            route.clear();
            error = "HALDA route owner is missing from the device set";
            return false;
        }
    }
    return build_halda_route(solution, route, error);
}

std::vector<halda_route_window> build_halda_route(
    const halda_solution & solution,
    const std::vector<halda_device> & devices,
    std::string & error) {
    std::vector<halda_route_window> route;
    build_halda_route(solution, devices, route, error);
    return route;
}
