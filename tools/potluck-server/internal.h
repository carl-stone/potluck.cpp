#pragma once

#include "llama.h"
#include "gguf.h"
#include "chat.h"
#include "nlohmann/json.hpp"
#include <cpp-httplib/httplib.h>
#include "potluck-discovery.h"
#include "potluck-transport.h"
#include "ggml.h"
#include "potluck_runtime.h"
#include "halda.h"
#include "speculative.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <algorithm>
#include <stdexcept>
using json = nlohmann::ordered_json;

#include <sys/types.h>

struct ring_worker {
    potluck::node_addr address;
    std::string bind_endpoint;
    std::string connect_endpoint;
    std::string next_endpoint;
};

struct bootstrap_node {
    std::string ssh_target;
    uint16_t ssh_port = 22;
    std::string ring_host;
    uint16_t ring_port = 40001;
    std::string known_hosts_file;
};

struct device_probe {
    std::string host;
    bootstrap_node bootstrap;
    potluck::device_profile profile;
    bool ok = false;
    std::string error;
    uint64_t placement_usable_limit = std::numeric_limits<uint64_t>::max();

    uint64_t usable_bytes(uint64_t limit) const {
        constexpr uint64_t mib = 1024ull * 1024ull;
        const uint64_t accel_reserve = std::max<uint64_t>(512ull * mib, profile.total_bytes / 8);
        const uint64_t host_reserve = std::max<uint64_t>(2ull * 1024ull * mib,
                                                         profile.host_total_bytes / 8);
        const uint64_t accel = profile.free_bytes > accel_reserve
            ? profile.free_bytes - accel_reserve : 0;
        const uint64_t host = profile.host_free_bytes > host_reserve
            ? profile.host_free_bytes - host_reserve : 0;
        return std::min(std::max(accel, host), limit);
    }

    uint64_t usable_bytes() const {
        return usable_bytes(placement_usable_limit);
    }
};

struct halda_model_metadata {
    uint32_t n_layer = 0;
    uint32_t n_embd = 0;
    uint32_t n_ff = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t n_vocab = 0;
    uint32_t n_ctx = 0;
    uint32_t head_dim = 0;
    uint64_t b = 0;
    uint64_t bi = 0;
    uint64_t bo = 0;
    uint64_t kv_per_layer = 0;
    uint64_t b_prime = 0;
    std::vector<ggml_type> ordered_types;
    std::vector<uint64_t> flops_per_type;
};

inline constexpr uint32_t potluck_probe_protocol_version = 3;
inline constexpr const char * potluck_probe_build_id = "potluck";

struct head_participation_plan {
    uint64_t budget = 0;
    bool participates = false;
};

inline head_participation_plan plan_head_participation(const device_probe & head,
                                                       uint64_t reserve,
                                                       uint64_t layer_cost) {
    const uint64_t budget = head.profile.host_free_bytes > reserve
        ? head.profile.host_free_bytes - reserve : 0;
    constexpr uint64_t min_budget = 2ull * 1024ull * 1024ull * 1024ull;
    return {
        budget,
        budget >= min_budget && head.usable_bytes(budget) >= layer_cost,
    };
}


enum class worker_kind {
    local,
    remote,
};

struct planned_worker {
    worker_kind kind = worker_kind::local;
    device_probe device;
    std::string model;
    ring_worker ring;
    pid_t local_pid = -1;
    pid_t remote_ssh_pid = -1;
    bool remote_launch_attempted = false;
};

struct ServerRing {
    std::vector<ring_worker> workers;
    std::vector<potluck::ring_window> windows;
    potluck::ring_receiver result;
    std::string result_endpoint;
    std::string result_server_public_key;
    std::vector<std::string> worker_server_keys;
    std::vector<potluck::ring_sender> controls;
    potluck::ring_sender ingress;
    std::deque<potluck::message> pending_results;
    uint64_t batch_sequence = 0;
    uint64_t heartbeat_sequence = 0;
    uint32_t heartbeat_misses = 0;
    std::chrono::steady_clock::time_point last_heartbeat{};
    std::string error;
};

struct ring_session {
    ServerRing ring;
    std::vector<planned_worker> workers;
    bool healthy = false;
    std::string health_reason;
    bool head_participates = false;
    bool has_head_profile = false;
    uint64_t head_budget = 0;
    uint64_t head_reserve = 0;
    double head_cpu_load = 0.0;
    potluck::device_profile head_profile;
    std::chrono::steady_clock::time_point last_heartbeat{};
    std::atomic<bool> stopping { false };
    mutable std::mutex mutex;
};


struct model_digest_cache {
    std::mutex mutex;
    std::filesystem::path path;
    std::filesystem::file_time_type mtime;
    uintmax_t size = 0;
    std::string digest;
    bool valid = false;

    static std::filesystem::path persistent_path(
        const std::filesystem::path & canonical_path);
    std::string get(const std::filesystem::path & model_path);
};

struct ring_startup_options {
    std::string hosts_spec;
    std::string model_path;
    std::string hf_repo;
    std::string hf_file;
    std::string hf_token;
    bool hf_offline = false;
    std::string model_name;
    std::string worker_path;
    std::string host;
    std::string head_share;
    std::string local_platform;
    std::filesystem::path adjacent_root;
    std::filesystem::path stage_dir;
    std::vector<bootstrap_node> * bootstrap_nodes = nullptr;
    model_digest_cache * digest_cache = nullptr;
    bool workers_option = false;
    bool has_staged_payload = false;
    uint32_t worker_local = 0;
    uint32_t n_layer = 0;
    uint32_t head_dim = 0;
    uint32_t n_head_kv = 0;
    uint32_t n_ctx = 0;
    uint32_t n_seq_max = 1;
    uint32_t n_ubatch = 0;
    uint32_t speculative_n_rs_seq = 0;
    uint64_t speculative_head_reserve = 0;
    uint32_t seed = 0;
    float temp = 0.0f;
    float top_p = 0.0f;
    potluck::prefetch_mode prefetch = potluck::prefetch_mode::advise;
    std::vector<uint32_t> layer_window;
    double gpu_mem_gib = 0.0;
    int32_t k_override = -1;
    double master_priority = 1.01;
};
inline uint64_t potluck_speculative_head_reserve(
        uint32_t n_head_kv,
        uint32_t head_dim,
        uint32_t n_ctx,
        uint32_t n_layer,
        uint32_t n_seq_max,
        const std::vector<std::string> & spec_types,
        const std::string & spec_draft_model) {
    const std::vector<common_speculative_type> parsed_spec_types =
        common_speculative_types_from_names(spec_types);
    const bool speculative_context = !spec_types.empty()
        ? std::any_of(parsed_spec_types.begin(), parsed_spec_types.end(),
                      [](common_speculative_type type) {
                          return type != COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE &&
                                 type != COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K &&
                                 type != COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V &&
                                 type != COMMON_SPECULATIVE_TYPE_NGRAM_MOD &&
                                 type != COMMON_SPECULATIVE_TYPE_NGRAM_CACHE;
                      })
        : !spec_draft_model.empty();
    if (!speculative_context) {
        return 0;
    }
    uint64_t draft_context_bytes = 1;
    const uint64_t factors[] = {
        static_cast<uint64_t>(n_head_kv),
        static_cast<uint64_t>(head_dim),
        4,
        static_cast<uint64_t>(n_ctx),
        static_cast<uint64_t>(n_layer),
        static_cast<uint64_t>(n_seq_max),
    };
    for (const uint64_t factor : factors) {
        if (factor == 0 ||
            draft_context_bytes > std::numeric_limits<uint64_t>::max() / factor) {
            return std::numeric_limits<uint64_t>::max();
        }
        draft_context_bytes *= factor;
    }
    return draft_context_bytes;
}
inline uint32_t potluck_speculative_n_rs_seq(
        const std::vector<std::string> & spec_types,
        const std::string & spec_draft_model,
        uint32_t n_draft) {
    std::vector<std::string> effective_types = spec_types;
    if (effective_types.empty() && !spec_draft_model.empty()) {
        effective_types.push_back("draft-simple");
    }
    common_params_speculative speculative;
    speculative.types = common_speculative_types_from_names(effective_types);
    speculative.draft.n_max = static_cast<int32_t>(n_draft == 0 ? 3 : n_draft);
    return speculative.need_n_rs_seq();
}

std::string shell_quote(const std::string & value);
void validate_ssh_target(const std::string & target);
std::string ssh_options(const bootstrap_node & bootstrap);
std::vector<bootstrap_node> discover_bootstrap_nodes();

device_probe probe_local_worker(
    const std::string & worker_path,
    const std::vector<ggml_type> & probe_types = {},
    uint64_t probe_bytes = 0);
device_probe probe_remote_worker(
    const bootstrap_node & bootstrap,
    const std::vector<ggml_type> & probe_types = {},
    uint64_t probe_bytes = 0);
std::vector<device_probe> probe_remote_candidates(
    const std::vector<bootstrap_node> & candidates,
    const std::vector<ggml_type> & probe_types = {},
    uint64_t probe_bytes = 0);
device_probe probe_local_pressure(const std::string & worker_path);
device_probe probe_remote_pressure(const bootstrap_node & bootstrap);
std::vector<device_probe> probe_remote_pressure_candidates(
    const std::vector<bootstrap_node> & candidates);
bool merge_pressure_profile(potluck::device_profile & target,
                            const potluck::device_profile & pressure,
                            std::string & error);

bool extract_halda_model_metadata(const std::filesystem::path & model_path,
                                  uint32_t n_ctx,
                                  halda_model_metadata & metadata,
                                  std::string & error);
bool solve_ring_placement(const std::vector<device_probe> & candidates,
                          const halda_model_metadata & metadata,
                          uint32_t n_ubatch, bool head_participates,
                          const std::vector<uint32_t> & fixed_w,
                          int32_t k_override, double master_priority,
                          double gpu_mem_gib,
                          std::vector<device_probe> & active_devices,
                          std::vector<potluck::ring_window> & windows,
                          halda_solution * solution, std::string & error);

std::string first_command_line(const std::string & command);
bool refresh_remote_binaries(const bootstrap_node & bootstrap,
                             const std::filesystem::path & stage_dir,
                             const std::string & local_platform);
bool ensure_remote_artifact(const bootstrap_node & bootstrap,
                            const std::filesystem::path & local_path,
                            const std::filesystem::path & remote_path,
                            const std::string & digest, std::string & error);
bool ensure_remote_hf_artifact(const bootstrap_node & bootstrap,
                               const std::filesystem::path & remote_path,
                               const std::string & hf_repo,
                               const std::string & hf_file,
                               const std::string & hf_token,
                               const std::string & digest, bool offline,
                               std::string & error);
std::string pinned_model_digest(const std::filesystem::path & model_path,
                                const std::filesystem::path & repo_root);

std::vector<potluck::device_profile> collect_device_profiles(ServerRing & ring, uint32_t n_workers);
const char * accel_kind_name(potluck::accel_kind kind);

std::filesystem::path canonical_model_path(const std::filesystem::path & model_path);

uint16_t free_port();
std::string ipv4_address_for_host(const std::string & host);
std::string local_address_for_peer(const std::string & host, uint16_t port);
std::string tcp_endpoint(const std::string & host, uint16_t port);
std::string endpoint_host(const std::string & endpoint, const std::string & host);

pid_t launch_remote_worker(const bootstrap_node & bootstrap, const std::string & model,
                           const ring_worker & worker, const std::string & result_endpoint,
                           uint32_t index,
                           const potluck::curve_bootstrap_credentials & credentials);
pid_t launch_local_worker(const std::string & worker_path, const std::string & model,
                          const ring_worker & worker, const std::string & result_endpoint,
                          uint32_t index,
                          const potluck::curve_bootstrap_credentials & credentials);
void stop_planned_workers(const std::vector<planned_worker> & planned);
void configure_ring(ServerRing & ring, uint32_t n_layer, uint32_t n_ctx,
                    uint32_t n_seq_max, uint32_t n_ubatch, uint32_t n_rs_seq,
                    uint32_t seed, float temp, float top_p, potluck::prefetch_mode prefetch,
                    const potluck::curve_client_credentials & controller_credentials);
bool reset_ring_workers(ServerRing & ring, std::string & error);

bool bring_up_ring(ring_session & target, ring_startup_options & options,
                   std::string & error);
bool rebuild_ring(ring_session & session, ring_startup_options & options,
                  bool restore_on_failure, std::string & error);
bool heartbeat_ring(ring_session & session, std::string & error);
bool ring_workers_alive(ring_session & session, std::string & error);

enum class topology_refresh_result {
    unchanged,
    rebuilt,
    failed,
};

topology_refresh_result refresh_ring_if_needed(ring_session & session,
                                               ring_startup_options & options,
                                               std::string & error);

class request_cancelled final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::vector<int32_t> drive_ring_cycle(ServerRing & ring,
                                 const std::vector<int32_t> & positions,
                                 const std::vector<int32_t> & sequences,
                                 const std::vector<int32_t> & tokens,
                                 int32_t clear_seq, int32_t trim_seq, int32_t trim_to,
                                 uint32_t n_logits,
                                 std::function<bool()> should_cancel = {},
                                 bool * batch_started = nullptr,
                                 std::function<bool(std::string &)> heartbeat = {},
                                 potluck::batch_logprobs * result_logprobs = nullptr,
                                 const std::vector<int32_t> & draft_tokens = {},
                                 uint32_t accepted_count = 0,
                                 uint32_t * result_accepted_count = nullptr);

std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & text);
std::string token_piece(const llama_vocab * vocab, llama_token token);
std::string render_tokens(const llama_vocab * vocab, const std::vector<llama_token> & tokens);
double peak_rss_mb();
void configure_slot(ServerRing & ring, const potluck::slot_config & config);
enum class slot_state {
    free,
    queued,
    prefill,
    decode,
    done,
    cancelled,
};

inline const char * slot_state_name(slot_state state) {
    switch (state) {
        case slot_state::free: return "free";
        case slot_state::queued: return "queued";
        case slot_state::prefill: return "prefill";
        case slot_state::decode: return "decode";
        case slot_state::done: return "done";
        case slot_state::cancelled: return "cancelled";
    }
    return "free";
}
struct slot_speculative_config {
    std::string draft_model;
    std::vector<std::string> types;
    uint32_t n_draft = 0;
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 2048;
    uint32_t n_ubatch = 512;
};


struct scheduled_slot {
    uint32_t index = 0;
    int32_t seq = 0;
    std::string conversation;
    slot_state state = slot_state::free;
    std::vector<llama_token> prompt;
    size_t prefill_offset = 0;
    uint32_t n_predict = 0;
    uint32_t n_decoded = 0;
    uint32_t next_position = 0;
    llama_token last = 0;
    potluck::slot_config sampling;
    std::vector<std::string> stops;
    bool configured = false;
    bool ever_used = false;
    bool needs_clear = false;
    bool needs_trim = false;
    int32_t trim_to = -1;
    bool speculative_started = false;
    bool speculative_prompt_tail_pending = false;
    llama_token speculative_prompt_tail = 0;
    int32_t speculative_prompt_tail_position = -1;
    common_speculative_init_result_ptr speculative_init;
    common_speculative_ptr speculative;
    bool stream = false;
    bool chat = false;
    std::string id;
    uint64_t created = 0;
    std::string generated_text;
    size_t stop_offset = std::string::npos;
    std::deque<std::string> pieces;
    std::deque<std::vector<potluck::token_logprob>> piece_logprobs;
    std::vector<llama_token> generated;
    std::vector<std::vector<potluck::token_logprob>> generated_logprobs;
    std::string error;
    bool finished = false;
    bool cancelled = false;
    bool release_when_finished = false;
    bool callback_done = false;
    std::mutex mutex;
    std::condition_variable cv;
};

struct batch_item {
    std::shared_ptr<scheduled_slot> slot;
    int32_t position = 0;
    int32_t token = 0;
};

class slot_scheduler {
public:
    slot_scheduler(ServerRing & ring, const llama_vocab * vocab, uint32_t n_slots,
                   uint32_t prefill_batch, std::function<bool(std::string &)> rebuild = {},
                   std::function<bool(std::string &)> heartbeat = {},
                   std::function<topology_refresh_result(std::string &)> refresh = {},
                   const llama_model * target_model = nullptr,
                   slot_speculative_config speculative = {})
        : ring_(ring), vocab_(vocab), target_model_(target_model),
          speculative_config_(std::move(speculative)),
          prefill_batch_(std::max<uint32_t>(1, prefill_batch)),
          rebuild_(std::move(rebuild)),
          heartbeat_(std::move(heartbeat)),
          refresh_(std::move(refresh)) {
        slots_.reserve(std::max<uint32_t>(1, n_slots));
        for (uint32_t i = 0; i < std::max<uint32_t>(1, n_slots); ++i) {
            auto slot = std::make_shared<scheduled_slot>();
            slot->index = i;
            slot->seq = static_cast<int32_t>(i);
            slots_.push_back(std::move(slot));
        }
    }

    void start() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!thread_.joinable()) {
            next_topology_check_ = std::chrono::steady_clock::now() +
                                   std::chrono::seconds(30);
            thread_ = std::thread([this] { run(); });
        }
    }

    void request_stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            for (const auto & slot : slots_) {
                std::lock_guard<std::mutex> slot_lock(slot->mutex);
                slot->cancelled = true;
                slot->state = slot_state::cancelled;
                slot->finished = true;
                if (slot->error.empty()) {
                    slot->error = "server stopped";
                }
                slot->cv.notify_all();
            }
        }
        work_cv_.notify_all();
    }

    void stop() {
        request_stop();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    ~slot_scheduler() {
        stop();
        if (speculative_target_context_ != nullptr) {
            llama_free(speculative_target_context_);
            speculative_target_context_ = nullptr;
        }
        if (speculative_configured()) {
            const double accept_rate = speculative_drafted_ == 0
                ? 0.0
                : static_cast<double>(speculative_accepted_) /
                      static_cast<double>(speculative_drafted_);
            std::printf("potluck-server: speculative drafted=%llu accepted=%llu accept-rate=%.3f\n",
                        static_cast<unsigned long long>(speculative_drafted_),
                        static_cast<unsigned long long>(speculative_accepted_),
                        accept_rate);
            std::fflush(stdout);
        }
    }
    std::vector<std::shared_ptr<scheduled_slot>> acquire_many(
            const std::vector<llama_token> & prompt,
            uint32_t n_predict,
            const potluck::slot_config & sampling,
            const std::vector<std::string> & stops,
            bool stream, bool chat,
            const std::string & id, uint64_t created,
            size_t count, const std::string & conversation = {},
            const std::function<bool()> & disconnected = {}) {
        std::vector<std::string> notes;
        std::vector<std::shared_ptr<scheduled_slot>> acquired;
        std::unique_lock<std::mutex> lock(mutex_);
        if (count == 0 || count > slots_.size() || stopping_ || recovery_exhausted_ ||
            (!conversation.empty() && count != 1) ||
            (disconnected && disconnected())) {
            lock.unlock();
            print_notes(notes);
            return {};
        }

        std::shared_ptr<scheduled_slot> bound_slot;
        if (!conversation.empty()) {
            auto binding = conversations_.find(conversation);
            if (binding != conversations_.end()) {
                if (binding->second.slot_index < slots_.size()) {
                    bound_slot = slots_[binding->second.slot_index];
                } else {
                    conversations_.erase(binding);
                }
            }
            if (bound_slot != nullptr) {
                bool free = false;
                {
                    std::lock_guard<std::mutex> slot_lock(bound_slot->mutex);
                    free = bound_slot->state == slot_state::free && !bound_slot->needs_clear;
                }
                if (!free) {
                    preempt_locked(bound_slot, notes);
                }
            }
            const bool ready = work_cv_.wait_for(lock, std::chrono::seconds(30), [&] {
                if (stopping_ || recovery_exhausted_ ||
                    (disconnected && disconnected())) {
                    return true;
                }
                if (rebuilding_) {
                    return false;
                }
                const auto current = conversations_.find(conversation);
                if (current != conversations_.end() &&
                    current->second.slot_index < slots_.size()) {
                    const auto candidate = slots_[current->second.slot_index];
                    std::lock_guard<std::mutex> slot_lock(candidate->mutex);
                    return candidate->state == slot_state::free && !candidate->needs_clear;
                }
                return free_slots_locked(count).size() >= count;
            });
            if (!ready || stopping_ || recovery_exhausted_ ||
                (disconnected && disconnected())) {
                lock.unlock();
                print_notes(notes);
                return {};
            }
            binding = conversations_.find(conversation);
            if (binding != conversations_.end() && binding->second.slot_index < slots_.size()) {
                bound_slot = slots_[binding->second.slot_index];
                std::lock_guard<std::mutex> slot_lock(bound_slot->mutex);
                if (bound_slot->state == slot_state::free && !bound_slot->needs_clear) {
                    acquired.push_back(bound_slot);
                }
            } else {
                acquired = free_slots_locked(count);
            }
        } else {
            const bool ready = work_cv_.wait_for(lock, std::chrono::seconds(30), [&] {
                return stopping_ || recovery_exhausted_ ||
                       (disconnected && disconnected()) ||
                       (!rebuilding_ && free_slots_locked(count).size() >= count);
            });
            if (!ready || stopping_ || recovery_exhausted_ ||
                (disconnected && disconnected())) {
                lock.unlock();
                print_notes(notes);
                return {};
            }
            acquired = free_slots_locked(count);
        }
        if (acquired.size() != count) {
            lock.unlock();
            print_notes(notes);
            return {};
        }

        for (const auto & slot : acquired) {
            std::string owner;
            {
                std::lock_guard<std::mutex> slot_lock(slot->mutex);
                owner = slot->conversation;
            }
            if (!owner.empty() && owner != conversation) {
                drop_binding_locked(slot, notes);
            }
            std::lock_guard<std::mutex> slot_lock(slot->mutex);
            slot->prompt = prompt;
            slot->prefill_offset = 0;
            slot->n_predict = n_predict;
            slot->n_decoded = 0;
            slot->next_position = 0;
            slot->last = 0;
            slot->sampling = sampling;
            slot->sampling.seq = slot->seq;
            slot->sampling.seed += static_cast<uint32_t>(acquired.size());
            slot->stops = stops;
            slot->configured = false;
            slot->needs_clear = slot->ever_used;
            slot->stream = stream;
            slot->chat = chat;
            slot->id = id;
            slot->created = created;
            slot->generated_text.clear();
            slot->stop_offset = std::string::npos;
            slot->pieces.clear();
            slot->piece_logprobs.clear();
            slot->generated.clear();
            slot->generated_logprobs.clear();
            slot->error.clear();
            slot->finished = false;
            slot->cancelled = false;
            slot->release_when_finished = false;
            slot->callback_done = false;
            slot->state = slot_state::queued;
            if (!conversation.empty()) {
                slot->conversation = conversation;
                auto & binding = conversations_[conversation];
                if (binding.slot_index == slot->index && binding.turns != 0) {
                    ++binding.turns;
                } else {
                    binding.slot_index = slot->index;
                    binding.turns = 1;
                }
                binding.last_use = ++conversation_clock_;
                notes.push_back(
                    "potluck-server: conversation " + conversation +
                    " slot " + std::to_string(slot->index) +
                    " seq " + std::to_string(slot->seq) +
                    " turn " + std::to_string(binding.turns) +
                    " prompt " + std::to_string(prompt.size()));
            } else {
                slot->conversation.clear();
            }
        }
        work_cv_.notify_all();
        lock.unlock();
        print_notes(notes);
        return acquired;
    }
    bool rebuilding() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return rebuilding_ || recovery_exhausted_;
    }
    bool recovery_exhausted() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return recovery_exhausted_;
    }
    std::string recovery_error() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return recovery_error_;
    }


    void wait_done(const std::shared_ptr<scheduled_slot> & slot) {
        std::unique_lock<std::mutex> lock(slot->mutex);
        slot->cv.wait(lock, [&] { return slot->finished; });
    }
    bool wait_done(const std::shared_ptr<scheduled_slot> & slot,
                   std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(slot->mutex);
        return slot->cv.wait_for(lock, timeout, [&] { return slot->finished; });
    }
    bool wait_ready(const std::shared_ptr<scheduled_slot> & slot,
                    std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(slot->mutex);
        return slot->cv.wait_for(lock, timeout, [&] {
            return slot->cancelled || !slot->pieces.empty() || slot->finished;
        });
    }

    bool take_piece(const std::shared_ptr<scheduled_slot> & slot, std::string & piece,
                    std::vector<potluck::token_logprob> * logprobs = nullptr) {
        std::unique_lock<std::mutex> lock(slot->mutex);
        slot->cv.wait(lock, [&] {
            return slot->cancelled || !slot->pieces.empty() || slot->finished;
        });
        if (!slot->pieces.empty()) {
            piece = std::move(slot->pieces.front());
            slot->pieces.pop_front();
            if (logprobs != nullptr) {
                if (!slot->piece_logprobs.empty()) {
                    *logprobs = std::move(slot->piece_logprobs.front());
                    slot->piece_logprobs.pop_front();
                } else {
                    logprobs->clear();
                }
            } else if (!slot->piece_logprobs.empty()) {
                slot->piece_logprobs.pop_front();
            }
            return true;
        }
        return false;
    }
    void cancel(const std::shared_ptr<scheduled_slot> & slot) {
        std::lock_guard<std::mutex> lock(slot->mutex);
        if (slot->state == slot_state::free) {
            return;
        }
        slot->cancelled = true;
        slot->release_when_finished = true;
        slot->state = slot_state::cancelled;
        slot->cv.notify_all();
        work_cv_.notify_all();
    }
    void acknowledge_cancel(const std::shared_ptr<scheduled_slot> & slot) {
        std::lock_guard<std::mutex> lock(slot->mutex);
        if (slot->state == slot_state::free) {
            return;
        }
        slot->callback_done = true;
        slot->cv.notify_all();
        work_cv_.notify_all();
    }
    bool release(const std::shared_ptr<scheduled_slot> & slot) {
        std::lock_guard<std::mutex> scheduler_lock(mutex_);
        {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (slot->state == slot_state::free) {
                return true;
            }
            if (slot->cancelled && slot->release_when_finished) {
                slot->callback_done = true;
                slot->cv.notify_all();
                work_cv_.notify_all();
                return false;
            }
            slot->prompt.clear();
            slot->pieces.clear();
            slot->piece_logprobs.clear();
            slot->generated_logprobs.clear();
            slot->generated.clear();
            slot->speculative.reset();
            slot->speculative_init.reset();
            slot->speculative_started = false;
            slot->speculative_prompt_tail_pending = false;
            slot->speculative_prompt_tail = 0;
            slot->speculative_prompt_tail_position = -1;
            slot->needs_trim = false;
            slot->trim_to = -1;
            slot->n_decoded = 0;
            slot->needs_clear = slot->ever_used;
            slot->cancelled = false;
            slot->release_when_finished = false;
            slot->callback_done = false;
            slot->state = slot_state::free;
            if (!slot->conversation.empty()) {
                const auto binding = conversations_.find(slot->conversation);
                if (binding != conversations_.end() &&
                    binding->second.slot_index == slot->index) {
                    binding->second.last_use = ++conversation_clock_;
                }
            }
        }
        work_cv_.notify_all();
        return true;
    }

    json health() const {
        json output = json::array();
        for (const auto & slot : slots_) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            output.push_back(json{
                { "index", slot->index },
                { "state", slot_state_name(slot->state) },
                { "n_decoded", slot->n_decoded },
                { "conversation", slot->conversation },
            });
        }
        return output;
    }
    size_t slot_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slots_.size();
    }

    bool is_stopping() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopping_;
    }

private:
    bool speculative_configured() const {
        return speculative_config_.n_draft != 0 ||
               !speculative_config_.draft_model.empty() ||
               !speculative_config_.types.empty();
    }
    void print_notes(const std::vector<std::string> & notes) const {
        if (notes.empty()) {
            return;
        }
        for (const std::string & note : notes) {
            std::printf("%s\n", note.c_str());
        }
        std::fflush(stdout);
    }
    void preempt_locked(const std::shared_ptr<scheduled_slot> & slot,
                        std::vector<std::string> & notes) {
        std::lock_guard<std::mutex> lock(slot->mutex);
        if (slot->state == slot_state::free) {
            return;
        }
        if (!slot->cancelled && !slot->conversation.empty()) {
            notes.push_back(
                "potluck-server: conversation " + slot->conversation +
                " preempted slot " + std::to_string(slot->index));
        }
        slot->cancelled = true;
        slot->release_when_finished = true;
        slot->state = slot_state::cancelled;
        slot->cv.notify_all();
        work_cv_.notify_all();
    }
    void drop_binding_locked(const std::shared_ptr<scheduled_slot> & slot,
                             std::vector<std::string> & notes) {
        std::lock_guard<std::mutex> lock(slot->mutex);
        if (slot->conversation.empty()) {
            return;
        }
        const std::string conversation = slot->conversation;
        const auto binding = conversations_.find(conversation);
        if (binding != conversations_.end() &&
            binding->second.slot_index == slot->index) {
            conversations_.erase(binding);
        }
        slot->conversation.clear();
        notes.push_back(
            "potluck-server: conversation " + conversation +
            " dropped from slot " + std::to_string(slot->index));
    }
    std::vector<std::shared_ptr<scheduled_slot>> free_slots_locked(size_t count) const {
        std::vector<std::shared_ptr<scheduled_slot>> result;
        result.reserve(count);
        std::vector<std::pair<uint64_t, std::shared_ptr<scheduled_slot>>> bound;
        for (const auto & slot : slots_) {
            if (slot->state != slot_state::free || slot->needs_clear) {
                continue;
            }
            if (slot->conversation.empty()) {
                result.push_back(slot);
                if (result.size() == count) {
                    return result;
                }
                continue;
            }
            uint64_t last_use = 0;
            const auto binding = conversations_.find(slot->conversation);
            if (binding != conversations_.end() &&
                binding->second.slot_index == slot->index) {
                last_use = binding->second.last_use;
            }
            bound.emplace_back(last_use, slot);
        }
        std::sort(bound.begin(), bound.end(),
                  [](const auto & left, const auto & right) {
                      if (left.first != right.first) {
                          return left.first < right.first;
                      }
                      return left.second->index < right.second->index;
                  });
        for (const auto & entry : bound) {
            result.push_back(entry.second);
            if (result.size() == count) {
                break;
            }
        }
        return result;
    }

    void ensure_speculator(const std::shared_ptr<scheduled_slot> & slot);
    void process_speculative_batch(const std::shared_ptr<scheduled_slot> & slot,
                                   const std::vector<int32_t> & positions,
                                   const std::vector<int32_t> & tokens,
                                   bool prompt_complete,
                                   uint32_t accepted_count,
                                   int32_t trim_to);
    void prime_speculative_prompt_tail(const std::shared_ptr<scheduled_slot> & slot);
    void reap_cancelled() {
        std::vector<std::shared_ptr<scheduled_slot>> ready;
        for (const auto & slot : slots_) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (slot->release_when_finished && slot->callback_done && slot->finished) {
                slot->release_when_finished = false;
                ready.push_back(slot);
            }
        }
        for (const auto & slot : ready) {
            release(slot);
        }
    }
    void finish(const std::shared_ptr<scheduled_slot> & slot, const std::string & error = {}) {
        std::lock_guard<std::mutex> lock(slot->mutex);
        if (!error.empty() && !slot->cancelled) {
            slot->error = error;
        }
        slot->finished = true;
        if (!slot->cancelled) {
            slot->state = slot_state::done;
        }
        slot->cv.notify_all();
    }

    void emit(const std::shared_ptr<scheduled_slot> & slot, llama_token token,
              uint32_t position,
              const std::vector<potluck::token_logprob> & logprobs = {});


    void run_round(const std::vector<std::shared_ptr<scheduled_slot>> & active) {
        std::shared_ptr<scheduled_slot> clear_slot;
        std::vector<std::shared_ptr<scheduled_slot>> cancelled;
        for (const auto & slot : active) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (slot->state == slot_state::cancelled) {
                cancelled.push_back(slot);
                if ((slot->ever_used || slot->needs_clear) && !clear_slot) {
                    clear_slot = slot;
                }
            }
        }
        if (!clear_slot) {
            for (const auto & slot : active) {
                std::lock_guard<std::mutex> lock(slot->mutex);
                if (slot->state != slot_state::cancelled && slot->needs_clear) {
                    clear_slot = slot;
                    break;
                }
            }
        }

        std::vector<std::shared_ptr<scheduled_slot>> selected;
        for (const auto & slot : active) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (slot->state == slot_state::free || slot->state == slot_state::done ||
                slot->state == slot_state::cancelled) {
                continue;
            }
            
            selected.push_back(slot);
        }
        if (selected.empty()) {
            if (clear_slot) {
                const std::vector<int32_t> empty;
                try {
                    (void) drive_ring_cycle(ring_, empty, empty, empty, clear_slot->seq,
                                            -1, -1, 0,
                                            [this] { return is_stopping(); },
                                            nullptr, heartbeat_);
                } catch (const request_cancelled &) {
                    return;
                }
                std::lock_guard<std::mutex> lock(clear_slot->mutex);
                clear_slot->needs_clear = false;
                clear_slot->ever_used = false;
            }
            for (const auto & slot : cancelled) {
                std::lock_guard<std::mutex> lock(slot->mutex);
                if (!slot->ever_used && !slot->needs_clear) {
                    slot->finished = true;
                    slot->cv.notify_all();
                }
            }
            return;
        }

        std::vector<std::pair<std::shared_ptr<scheduled_slot>, potluck::slot_config>> configurations;
        for (const auto & slot : selected) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (!slot->configured) {
                configurations.emplace_back(slot, slot->sampling);
            }
        }
        for (const auto & configuration : configurations) {
            configure_slot(ring_, configuration.second);
            std::lock_guard<std::mutex> lock(configuration.first->mutex);
            configuration.first->configured = true;
        }
        if (speculative_configured()) {
            for (const auto & slot : selected) {
                ensure_speculator(slot);
            }
        }

        std::vector<batch_item> prefill_body;
        std::vector<batch_item> prefill_logits;
        std::vector<batch_item> decode;
        std::shared_ptr<scheduled_slot> speculative_slot;
        std::vector<int32_t> speculative_draft;
        std::vector<batch_item> speculative_items;
        std::vector<std::pair<std::shared_ptr<scheduled_slot>, size_t>> prefill_offsets;
        size_t prefill_count = 0;
        for (const auto & slot : selected) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (slot->state != slot_state::prefill) {
                continue;
            }
            const size_t remaining = slot->prompt.size() - slot->prefill_offset;
            const size_t count = std::min(remaining, prefill_batch_ - prefill_count);
            if (count != 0) {
                prefill_offsets.emplace_back(slot, slot->prefill_offset);
            }
            for (size_t i = 0; i < count; ++i) {
                batch_item item;
                item.slot = slot;
                item.position = static_cast<int32_t>(slot->prefill_offset + i);
                item.token = static_cast<int32_t>(slot->prompt[slot->prefill_offset + i]);
                if (i + 1 == count && slot->prefill_offset + count == slot->prompt.size()) {
                    prefill_logits.push_back(item);
                } else {
                    prefill_body.push_back(item);
                }
            }
            slot->prefill_offset += count;
            prefill_count += count;
            if (prefill_count == prefill_batch_) {
                break;
            }
        }
        const size_t available = prefill_batch_ - prefill_count;
        for (const auto & slot : selected) {
            if (decode.size() == available) {
                break;
            }
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (slot->state != slot_state::decode) {
                continue;
            }
            decode.push_back({
                slot, static_cast<int32_t>(slot->next_position),
                static_cast<int32_t>(slot->last)
            });
        }
        if (speculative_configured()) {
            for (const auto & item : decode) {
                prime_speculative_prompt_tail(item.slot);
            }
        }
        if (speculative_configured() && !decode.empty()) {
            const uint32_t configured_draft = speculative_config_.n_draft == 0
                ? 3 : speculative_config_.n_draft;
            for (size_t index = 0; index < decode.size(); ++index) {
                const size_t normal_count = decode.size() - 1;
                if (prefill_count + normal_count + configured_draft + 1 > prefill_batch_) {
                    continue;
                }
                const auto & candidate = decode[index];
                std::vector<llama_token> history;
                llama_token last = 0;
                uint32_t position = 0;
                {
                    std::lock_guard<std::mutex> lock(candidate.slot->mutex);
                    history = candidate.slot->prompt;
                    if (!candidate.slot->generated.empty()) {
                        history.insert(history.end(), candidate.slot->generated.begin(),
                                       candidate.slot->generated.end() - 1);
                    }
                    last = candidate.slot->last;
                    position = candidate.slot->next_position;
                }
                std::vector<llama_token> draft;
                auto & draft_params = common_speculative_get_draft_params(
                    candidate.slot->speculative.get(), 0);
                draft_params = {
                    true,
                    static_cast<int32_t>(configured_draft),
                    static_cast<llama_pos>(position),
                    last,
                    &history,
                    &draft,
                };
                common_speculative_draft(candidate.slot->speculative.get());
                if (candidate.slot->speculative_init != nullptr &&
                    candidate.slot->speculative_init->context() != nullptr &&
                    !llama_memory_seq_rm(
                        llama_get_memory(candidate.slot->speculative_init->context()),
                        0, static_cast<int32_t>(position), -1)) {
                    throw std::runtime_error(
                        "cannot reset speculative draft context for verification");
                }
                if (draft.size() > configured_draft) {
                    draft.resize(configured_draft);
                }
                if (draft.empty()) {
                    continue;
                }
                speculative_slot = candidate.slot;
                decode.erase(decode.begin() + static_cast<std::ptrdiff_t>(index));
                speculative_draft.reserve(draft.size());
                for (const llama_token token : draft) {
                    speculative_draft.push_back(static_cast<int32_t>(token));
                }
                speculative_items.push_back(candidate);
                for (size_t draft_index = 0; draft_index < draft.size(); ++draft_index) {
                    speculative_items.push_back({
                        candidate.slot,
                        static_cast<int32_t>(position + draft_index + 1),
                        static_cast<int32_t>(draft[draft_index]),
                    });
                }
                break;
            }
        }

        std::shared_ptr<scheduled_slot> trim_slot;
        int32_t trim_seq = -1;
        int32_t trim_to = -1;
        for (const auto & slot : selected) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (slot->needs_trim) {
                trim_slot = slot;
                trim_seq = slot->seq;
                trim_to = slot->trim_to;
                break;
            }
        }

        std::vector<batch_item> items;
        items.reserve(prefill_body.size() + decode.size() + prefill_logits.size() +
                      speculative_items.size());
        items.insert(items.end(), prefill_body.begin(), prefill_body.end());
        items.insert(items.end(), decode.begin(), decode.end());
        items.insert(items.end(), prefill_logits.begin(), prefill_logits.end());
        items.insert(items.end(), speculative_items.begin(), speculative_items.end());
        if (items.empty()) {
            return;
        }
        std::vector<int32_t> positions;
        std::vector<int32_t> sequences;
        std::vector<int32_t> tokens;
        positions.reserve(items.size());
        sequences.reserve(items.size());
        tokens.reserve(items.size());
        for (const batch_item & item : items) {
            positions.push_back(item.position);
            sequences.push_back(item.slot->seq);
            tokens.push_back(item.token);
        }
        const uint32_t n_logits = static_cast<uint32_t>(
            decode.size() + prefill_logits.size() + speculative_items.size());
        const int32_t clear_seq = clear_slot ? clear_slot->seq : -1;
        const auto batch_cancelled = [&]() {
            return is_stopping();
        };
        bool batch_started = false;
        const auto fail_selected = [&]() {
            for (const auto & slot : selected) {
                std::lock_guard<std::mutex> lock(slot->mutex);
                slot->cancelled = true;
                slot->state = slot_state::cancelled;
                slot->finished = true;
                slot->error = "request cancelled";
                slot->ever_used = true;
                slot->needs_clear = true;
                slot->cv.notify_all();
            }
            work_cv_.notify_all();
        };
        std::vector<int32_t> result;
        potluck::batch_logprobs result_logprobs;
        uint32_t accepted_count_result = 0;
        try {
            result = drive_ring_cycle(
                ring_, positions, sequences, tokens, clear_seq, trim_seq, trim_to, n_logits,
                batch_cancelled, &batch_started, heartbeat_, &result_logprobs,
                speculative_draft, 0, &accepted_count_result);
        } catch (const request_cancelled &) {
            if (batch_started) {
                fail_selected();
            } else {
                for (const auto & offset : prefill_offsets) {
                    std::lock_guard<std::mutex> lock(offset.first->mutex);
                    offset.first->prefill_offset = offset.second;
                }
                for (const auto & slot : selected) {
                    std::lock_guard<std::mutex> lock(slot->mutex);
                    if (slot->cancelled) {
                        slot->finished = true;
                        slot->error = "request cancelled";
                        slot->cv.notify_all();
                    }
                }
            }
            return;
        } catch (...) {
            for (const auto & slot : selected) {
                std::lock_guard<std::mutex> lock(slot->mutex);
                if (batch_started) {
                    slot->ever_used = true;
                }
                slot->needs_clear = true;
            }
            throw;
        }
        if (batch_cancelled()) {
            fail_selected();
            return;
        }
        if (clear_slot) {
            std::lock_guard<std::mutex> lock(clear_slot->mutex);
            clear_slot->needs_clear = false;
            if (clear_slot->cancelled) {
                clear_slot->ever_used = false;
            }
        }
        for (const auto & slot : selected) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            slot->ever_used = true;
            if (slot->cancelled) {
                slot->needs_clear = true;
            }
        }
        if (trim_slot != nullptr) {
            std::lock_guard<std::mutex> lock(trim_slot->mutex);
            trim_slot->needs_trim = false;
            trim_slot->trim_to = -1;
        }
        for (const auto & slot : selected) {
            std::vector<int32_t> slot_positions;
            std::vector<int32_t> slot_tokens;
            for (const batch_item & item : items) {
                if (item.slot == slot) {
                    slot_positions.push_back(item.position);
                    slot_tokens.push_back(item.token);
                }
            }
            if (slot_positions.empty()) {
                continue;
            }
            bool cancelled_slot = false;
            bool prompt_complete = false;
            {
                std::lock_guard<std::mutex> lock(slot->mutex);
                cancelled_slot = slot->cancelled;
                prompt_complete = slot->state == slot_state::prefill &&
                    slot->prefill_offset == slot->prompt.size();
            }
            if (cancelled_slot) {
                continue;
            }
            if (slot->speculative != nullptr) {
                const bool is_speculative = slot == speculative_slot;
                const int32_t next_trim = is_speculative
                    ? static_cast<int32_t>(
                        speculative_items.front().position + accepted_count_result + 1)
                    : -1;
                process_speculative_batch(
                    slot, slot_positions, slot_tokens, prompt_complete,
                    is_speculative ? accepted_count_result : 0, next_trim);
                if (is_speculative) {
                    std::lock_guard<std::mutex> lock(slot->mutex);
                    slot->needs_trim = true;
                    slot->trim_to = next_trim;
                    speculative_drafted_ += speculative_draft.size();
                    speculative_accepted_ += accepted_count_result;
                }
            }
        }
        const size_t result_start = items.size() - n_logits;
        const auto logprobs_at = [&](size_t index)
            -> const std::vector<potluck::token_logprob> & {
            static const std::vector<potluck::token_logprob> empty;
            return result_logprobs.size() == n_logits ? result_logprobs[index] : empty;
        };
        for (size_t i = 0; i < decode.size(); ++i) {
            emit(decode[i].slot, static_cast<llama_token>(result[result_start + i]),
                 static_cast<uint32_t>(decode[i].position), logprobs_at(i));
        }
        for (size_t i = 0; i < prefill_logits.size(); ++i) {
            emit(prefill_logits[i].slot,
                 static_cast<llama_token>(result[result_start + decode.size() + i]),
                 static_cast<uint32_t>(prefill_logits[i].position),
                 logprobs_at(decode.size() + i));
        }
        if (speculative_slot != nullptr) {
            const size_t spec_start = decode.size() + prefill_logits.size();
            const size_t spec_count = std::min(
                speculative_items.size(),
                static_cast<size_t>(accepted_count_result) + 1);
            for (size_t i = 0; i < spec_count; ++i) {
                emit(speculative_slot,
                     static_cast<llama_token>(result[result_start + spec_start + i]),
                     static_cast<uint32_t>(speculative_items[i].position),
                     logprobs_at(spec_start + i));
            }
        }
        for (const auto & slot : selected) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (!slot->cancelled) {
                continue;
            }
            slot->finished = true;
            slot->state = slot_state::cancelled;
            slot->error = "request cancelled";
            slot->cv.notify_all();
        }
        for (const auto & slot : cancelled) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (!slot->ever_used && !slot->needs_clear) {
                slot->finished = true;
                slot->cv.notify_all();
            }
        }
        work_cv_.notify_all();
    }

    std::chrono::milliseconds retry_delay_locked() {
        const uint32_t exponent = std::min<uint32_t>(
            rebuild_attempts_ == 0 ? 0 : rebuild_attempts_ - 1, 7);
        const uint64_t base = std::min<uint64_t>(30000ull, 250ull << exponent);
        jitter_state_ ^= jitter_state_ << 7;
        jitter_state_ ^= jitter_state_ >> 9;
        jitter_state_ ^= jitter_state_ << 8;
        const int jitter_percent = static_cast<int>(jitter_state_ % 41) - 20;
        const uint64_t adjusted = base * static_cast<uint64_t>(100 + jitter_percent) / 100;
        return std::chrono::milliseconds(std::max<uint64_t>(1, adjusted));
    }

    bool schedule_rebuild_retry_locked(const std::string & detail,
                                       std::chrono::milliseconds & delay) {
        ++rebuild_attempts_;
        rebuilding_ = true;
        recovery_error_ = detail.empty() ? "ring rebuild failed" : detail;
        constexpr uint32_t max_rebuild_attempts = 6;
        if (rebuild_attempts_ >= max_rebuild_attempts) {
            recovery_exhausted_ = true;
            recovery_error_ = "ring recovery exhausted after " +
                              std::to_string(rebuild_attempts_) + " attempts: " +
                              recovery_error_;
            return false;
        }
        delay = retry_delay_locked();
        next_rebuild_ = std::chrono::steady_clock::now() + delay;
        return true;
    }

    void fail_unfinished_slots(const std::string & error) {
        std::vector<std::shared_ptr<scheduled_slot>> pending;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending = slots_;
        }
        for (const auto & slot : pending) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            if (slot->state == slot_state::free || slot->finished) {
                continue;
            }
            const bool cancelled = slot->cancelled;
            slot->state = cancelled ? slot_state::cancelled : slot_state::done;
            slot->finished = true;
            if (!cancelled) {
                slot->cancelled = false;
                slot->error = error.empty() ? "ring recovery exhausted" : error;
            }
            slot->cv.notify_all();
        }
    }

    void attempt_rebuild() {
        std::string detail;
        bool rebuilt = false;
        try {
            rebuilt = rebuild_ && rebuild_(detail);
        } catch (const std::exception & exception) {
            detail = exception.what();
        }
        std::chrono::milliseconds delay(0);
        bool retry = false;
        bool exhausted = false;
        std::string terminal_error;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (rebuilt) {
                rebuilding_ = false;
                recovery_exhausted_ = false;
                rebuild_attempts_ = 0;
                recovery_error_.clear();
                next_rebuild_ = {};
                next_topology_check_ = std::chrono::steady_clock::now() +
                                       std::chrono::seconds(30);
            } else {
                retry = schedule_rebuild_retry_locked(detail, delay);
                exhausted = recovery_exhausted_;
                if (exhausted) {
                    terminal_error = recovery_error_;
                }
            }
        }
        if (rebuilt) {
            std::printf("potluck-server: ring rebuild succeeded: %s\n",
                        detail.empty() ? "ring is ready" : detail.c_str());
            std::fflush(stdout);
        } else if (exhausted) {
            fail_unfinished_slots(terminal_error);
            std::fprintf(stderr, "potluck-server: ring rebuild terminal: %s\n",
                         terminal_error.empty() ? "ring recovery exhausted" :
                                                   terminal_error.c_str());
        } else if (retry) {
            std::fprintf(stderr, "potluck-server: ring rebuild failed; retrying in %lld ms: %s\n",
                         static_cast<long long>(delay.count()),
                         detail.empty() ? "no detail" : detail.c_str());
        }
        work_cv_.notify_all();
    }
    void run() {
        for (;;) {
            std::vector<std::shared_ptr<scheduled_slot>> active;
            bool retry_rebuild = false;
            bool refresh_topology = false;
            bool waiting_for_rebuild = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_cv_.wait_for(lock, std::chrono::seconds(1), [&] {
                    if (stopping_) {
                        return true;
                    }
                    const auto now = std::chrono::steady_clock::now();
                    if (rebuilding_ && !recovery_exhausted_ && rebuild_ &&
                        next_rebuild_.time_since_epoch().count() != 0 &&
                        now >= next_rebuild_) {
                        return true;
                    }
                    if (!rebuilding_ && refresh_ && now >= next_topology_check_) {
                        return true;
                    }
                    for (const auto & slot : slots_) {
                        std::lock_guard<std::mutex> slot_lock(slot->mutex);
                        if (slot->state == slot_state::queued ||
                            slot->state == slot_state::prefill ||
                            slot->state == slot_state::decode ||
                            (slot->state == slot_state::cancelled &&
                             (!slot->finished ||
                              (slot->release_when_finished && slot->callback_done))) ||
                            (slot->state == slot_state::free && slot->needs_clear)) {
                            return true;
                        }
                    }
                    return false;
                });
                if (stopping_) {
                    return;
                }
                const auto now = std::chrono::steady_clock::now();
                bool slots_idle = true;
                for (const auto & slot : slots_) {
                    std::lock_guard<std::mutex> slot_lock(slot->mutex);
                    if (slot->state == slot_state::queued ||
                        slot->state == slot_state::prefill ||
                        slot->state == slot_state::decode ||
                        (slot->state == slot_state::cancelled &&
                         (!slot->finished ||
                          (slot->release_when_finished && slot->callback_done))) ||
                        (slot->state == slot_state::free && slot->needs_clear)) {
                        slots_idle = false;
                        break;
                    }
                }
                if (rebuilding_ && !recovery_exhausted_ && rebuild_ &&
                    next_rebuild_.time_since_epoch().count() != 0 &&
                    now >= next_rebuild_) {
                    retry_rebuild = true;
                    next_rebuild_ = {};
                } else if (!rebuilding_ && slots_idle && refresh_ && now >= next_topology_check_) {
                    rebuilding_ = true;
                    refresh_topology = true;
                    next_topology_check_ = now + std::chrono::seconds(30);
                } else if (!rebuilding_) {
                    for (const auto & slot : slots_) {
                        std::lock_guard<std::mutex> slot_lock(slot->mutex);
                        if (slot->state == slot_state::queued) {
                            if (slot->n_predict == 0) {
                                slot->finished = true;
                                slot->state = slot_state::done;
                                slot->cv.notify_all();
                            } else {
                                slot->state = slot_state::prefill;
                                active.push_back(slot);
                            }
                        } else if (slot->state == slot_state::prefill ||
                                   slot->state == slot_state::decode ||
                                   (slot->state == slot_state::cancelled && !slot->finished) ||
                                   (slot->state == slot_state::free && slot->needs_clear)) {
                            active.push_back(slot);
                        }
                    }
                } else if (recovery_exhausted_) {
                    waiting_for_rebuild = false;
                } else {
                    waiting_for_rebuild = true;
                }
            }
            if (retry_rebuild) {
                attempt_rebuild();
                continue;
            }
            if (waiting_for_rebuild) {
                continue;
            }
            if (refresh_topology) {
                std::string detail;
                topology_refresh_result result = topology_refresh_result::failed;
                try {
                    result = refresh_(detail);
                } catch (const std::exception & exception) {
                    detail = "topology refresh deferred: " + std::string(exception.what());
                    result = topology_refresh_result::unchanged;
                }
                if (result == topology_refresh_result::unchanged ||
                    result == topology_refresh_result::rebuilt) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    rebuilding_ = false;
                    recovery_exhausted_ = false;
                    rebuild_attempts_ = 0;
                    recovery_error_.clear();
                    next_rebuild_ = {};
                    next_topology_check_ = std::chrono::steady_clock::now() +
                                           std::chrono::seconds(30);
                    if (!detail.empty()) {
                        std::printf("potluck-server: topology refresh %s: %s\n",
                                    result == topology_refresh_result::rebuilt
                                        ? "rebuilt" : "unchanged", detail.c_str());
                    }
                } else {
                    std::chrono::milliseconds delay(0);
                    bool retry = false;
                    bool exhausted = false;
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        retry = schedule_rebuild_retry_locked(detail, delay);
                        exhausted = recovery_exhausted_;
                    }
                    if (exhausted) {
                        std::fprintf(stderr,
                                     "potluck-server: topology refresh recovery terminal: %s\n",
                                     recovery_error().c_str());
                    } else if (retry) {
                        std::fprintf(stderr,
                                     "potluck-server: topology refresh rebuild failed; retrying in %lld ms: %s\n",
                                     static_cast<long long>(delay.count()),
                                     detail.empty() ? "no detail" : detail.c_str());
                    }
                }
                work_cv_.notify_all();
                continue;
            }
            if (active.empty()) {
                reap_cancelled();
                if (heartbeat_) {
                    std::string heartbeat_error;
                    if (!heartbeat_(heartbeat_error)) {
                        bool start_rebuild = false;
                        {
                            std::lock_guard<std::mutex> lock(mutex_);
                            if (!rebuilding_) {
                                rebuilding_ = true;
                                recovery_exhausted_ = false;
                                recovery_error_.clear();
                                rebuild_attempts_ = 0;
                                next_rebuild_ = std::chrono::steady_clock::now();
                                start_rebuild = true;
                            }
                        }
                        if (start_rebuild) {
                            std::fprintf(stderr,
                                         "potluck-server: heartbeat lost: %s\n",
                                         heartbeat_error.empty()
                                             ? "no detail" : heartbeat_error.c_str());
                            attempt_rebuild();
                            continue;
                        }
                    }
                }
                work_cv_.notify_all();
                continue;
            }
            try {
                run_round(active);
            } catch (const std::exception & exception) {
                if (is_stopping()) {
                    return;
                }
                bool start_rebuild = false;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (!rebuilding_) {
                        rebuilding_ = true;
                        recovery_exhausted_ = false;
                        recovery_error_.clear();
                        rebuild_attempts_ = 0;
                        next_rebuild_ = std::chrono::steady_clock::now();
                        start_rebuild = true;
                    }
                }
                for (const auto & slot : active) {
                    finish(slot, "request interrupted by ring rebuild; retry");
                }
                reap_cancelled();
                if (start_rebuild) {
                    std::fprintf(stderr, "potluck-server: request transport failed: %s\n",
                                 exception.what());
                    work_cv_.notify_all();
                    attempt_rebuild();
                }
            }
            reap_cancelled();
        }
    }

    ServerRing & ring_;
    const llama_vocab * vocab_;
    const llama_model * target_model_;
    slot_speculative_config speculative_config_;
    llama_context * speculative_target_context_ = nullptr;
    uint64_t speculative_drafted_ = 0;
    uint64_t speculative_accepted_ = 0;
    uint32_t prefill_batch_;
    std::function<bool(std::string &)> rebuild_;
    std::function<bool(std::string &)> heartbeat_;
    std::function<topology_refresh_result(std::string &)> refresh_;
    std::chrono::steady_clock::time_point next_rebuild_{};
    std::chrono::steady_clock::time_point next_topology_check_{};
    uint32_t rebuild_attempts_ = 0;
    bool rebuilding_ = false;
    bool recovery_exhausted_ = false;
    std::string recovery_error_;
    struct conversation_binding {
        uint32_t slot_index = 0;
        uint64_t turns = 0;
        uint64_t last_use = 0;
    };
    std::map<std::string, conversation_binding> conversations_;
    uint64_t conversation_clock_ = 0;
    uint64_t jitter_state_ = 0x9e3779b97f4a7c15ull;
    mutable std::mutex mutex_;
    std::condition_variable work_cv_;
    std::vector<std::shared_ptr<scheduled_slot>> slots_;
    std::thread thread_;
    bool stopping_ = false;

};

void setup_http_routes(httplib::Server & server, ring_session & session,
                       slot_scheduler & scheduler, const llama_vocab * vocab,
                       const std::string & model_name,
                       common_chat_templates_ptr & chat_templates,
                       uint32_t n_predict_default, float temp, float top_p, uint32_t seed,
                       const std::string & api_key, const std::string & cors_origin);
