#pragma once

#include "llama.h"
#include "chat.h"
#include "nlohmann/json.hpp"
#include "potluck-discovery.h"
#include "potluck-transport.h"
#include "potluck_runtime.h"
#include <cpp-httplib/httplib.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <limits>
#include <set>
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
    potluck::accel_profile profile;
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
};

struct ServerRing {
    std::vector<ring_worker> workers;
    std::vector<potluck::ring_window> windows;
    potluck::ring_receiver result;
    std::string result_endpoint;
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
    potluck::accel_profile head_profile;
    std::chrono::steady_clock::time_point last_heartbeat{};
    std::atomic<bool> stopping { false };
    mutable std::mutex mutex;
};

struct serve_stats {
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    uint64_t head_payload_bytes = 0;
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
    uint32_t n_seq_max = 0;
    uint32_t n_ubatch = 0;
    uint32_t seed = 0;
    float temp = 0.0f;
    float top_p = 0.0f;
};

std::string shell_quote(const std::string & value);
void validate_ssh_target(const std::string & target);
std::string ssh_options(const bootstrap_node & bootstrap);
std::vector<bootstrap_node> discover_bootstrap_nodes();

device_probe probe_local_worker(const std::string & worker_path);
device_probe probe_remote_worker(const bootstrap_node & bootstrap);
std::vector<device_probe> probe_remote_candidates(
    const std::vector<bootstrap_node> & candidates);

uint64_t route_layer_cost(uint32_t n_layer, uint64_t model_bytes, uint32_t n_head_kv,
                          uint32_t head_dim, uint32_t n_ctx);
uint64_t route_needed_bytes(uint32_t n_layer, uint64_t model_bytes, uint32_t n_head_kv,
                            uint32_t head_dim, uint32_t n_ctx);
std::vector<device_probe> admit_devices(std::vector<device_probe> candidates,
                                         uint32_t n_layer, uint64_t model_bytes,
                                         uint32_t n_head_kv, uint32_t head_dim,
                                         uint32_t n_ctx, bool allow_shortfall = false);
std::vector<potluck::ring_window> build_ring_route(const std::vector<device_probe> & devices,
                                                   uint32_t n_layer, uint64_t model_bytes,
                                                   uint32_t n_head_kv, uint32_t head_dim,
                                                   uint32_t n_ctx);

std::string first_command_line(const std::string & command);
bool refresh_remote_binaries(const bootstrap_node & bootstrap,
                             const std::filesystem::path & stage_dir,
                             const std::string & local_platform);
bool ensure_remote_model(const bootstrap_node & bootstrap, const std::string & model_path,
                         const std::string & digest, std::string & error);
std::string pinned_model_digest(const std::filesystem::path & model_path,
                                const std::filesystem::path & repo_root);

std::vector<potluck::accel_profile> collect_accel_profiles(ServerRing & ring, uint32_t n_workers);
void assign_gpu_layers(std::vector<potluck::ring_window> & windows,
                       const std::vector<potluck::accel_profile> & profiles,
                       uint32_t n_workers, uint64_t model_bytes, uint32_t n_layer,
                       uint32_t n_head_kv, uint32_t head_dim, uint32_t n_ctx);
const char * accel_kind_name(potluck::accel_kind kind);

std::filesystem::path canonical_model_path(const std::filesystem::path & model_path);

uint16_t free_port();
std::string ipv4_address_for_host(const std::string & host);
std::string local_address_for_peer(const std::string & host, uint16_t port);
std::string tcp_endpoint(const std::string & host, uint16_t port);
std::string endpoint_host(const std::string & endpoint, const std::string & host);

pid_t launch_remote_worker(const bootstrap_node & bootstrap, const std::string & model,
                           const ring_worker & worker, const std::string & result_endpoint,
                           uint32_t index);
pid_t launch_local_worker(const std::string & worker_path, const std::string & model,
                          const ring_worker & worker, const std::string & result_endpoint,
                          uint32_t index);
bool stop_remote_workers(const bootstrap_node & bootstrap);
void terminate_child_process(pid_t pid);
void stop_planned_workers(const std::vector<planned_worker> & planned);

void configure_ring(ServerRing & ring, uint32_t n_layer, uint32_t n_ctx,
                    uint32_t n_seq_max, uint32_t n_ubatch, uint32_t seed,
                    float temp, float top_p);
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

std::vector<int32_t> drive_batch(ServerRing & ring,
                                 const std::vector<int32_t> & positions,
                                 const std::vector<int32_t> & sequences,
                                 const std::vector<int32_t> & tokens,
                                 int32_t clear_seq, int32_t trim_seq, int32_t trim_to,
                                 uint32_t n_logits,
                                 serve_stats * stats = nullptr,
                                 std::function<bool()> should_cancel = {},
                                 bool * batch_started = nullptr,
                                 std::function<bool(std::string &)> heartbeat = {},
                                 potluck::batch_logprobs * result_logprobs = nullptr);

std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & text);
std::string token_piece(const llama_vocab * vocab, llama_token token);
std::string render_tokens(const llama_vocab * vocab, const std::vector<llama_token> & tokens);
double peak_rss_mb();
void configure_slot(ServerRing & ring, const potluck::slot_config & config);
std::vector<llama_token> serve(ServerRing & ring, const llama_vocab * vocab,
                               const std::vector<llama_token> & prompt,
                               uint32_t n_predict,
                               const std::function<void(const std::string &)> & emit,
                               serve_stats * stats = nullptr,
                               int32_t sequence_id = 0,
                               potluck::slot_config sampling = {},
                               uint32_t prefill_batch = 512);
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

struct scheduled_slot {
    uint32_t index = 0;
    int32_t seq = 0;
    slot_state state = slot_state::free;
    std::vector<llama_token> prompt;
    size_t prefill_offset = 0;
    uint32_t n_predict = 0;
    uint32_t n_decoded = 0;
    uint32_t next_position = 0;
    llama_token last = 0;
    potluck::slot_config sampling;
    bool configured = false;
    bool ever_used = false;
    bool needs_clear = false;
    bool stream = false;
    bool chat = false;
    std::string id;
    uint64_t created = 0;
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
                   std::function<topology_refresh_result(std::string &)> refresh = {})
        : ring_(ring), vocab_(vocab),
          prefill_batch_(std::max<uint32_t>(1, prefill_batch)),
          rebuild_(std::move(rebuild)),
          heartbeat_(std::move(heartbeat)),
          refresh_(std::move(refresh)),
          next_topology_check_(std::chrono::steady_clock::now() + std::chrono::seconds(5)) {
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
    }

    std::shared_ptr<scheduled_slot> acquire(const std::vector<llama_token> & prompt,
                                            uint32_t n_predict,
                                            const potluck::slot_config & sampling,
                                            bool stream, bool chat,
                                            const std::string & id, uint64_t created) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (stopping_ || rebuilding_) {
            return {};
        }
        if (!work_cv_.wait_for(lock, std::chrono::seconds(30), [&] {
                if (stopping_ || rebuilding_) {
                    return true;
                }
                for (const auto & slot : slots_) {
                    std::lock_guard<std::mutex> slot_lock(slot->mutex);
                    if (slot->state == slot_state::free) {
                        return true;
                    }
                }
                return false;
            })) {
            return {};
        }
        if (stopping_ || rebuilding_) {
            return {};
        }
        for (const auto & slot : slots_) {
            std::lock_guard<std::mutex> slot_lock(slot->mutex);
            if (slot->state != slot_state::free) {
                continue;
            }
            slot->prompt = prompt;
            slot->prefill_offset = 0;
            slot->n_predict = n_predict;
            slot->n_decoded = 0;

            slot->next_position = 0;
            slot->last = 0;
            slot->sampling = sampling;
            slot->sampling.seq = slot->seq;
            slot->configured = false;
            slot->needs_clear = slot->ever_used;
            slot->stream = stream;
            slot->chat = chat;
            slot->id = id;
            slot->created = created;
            slot->pieces.clear();
            slot->piece_logprobs.clear();
            slot->generated_logprobs.clear();
            slot->error.clear();
            slot->finished = false;
            slot->cancelled = false;
            slot->release_when_finished = false;
            slot->callback_done = false;
            slot->state = slot_state::queued;
            work_cv_.notify_all();
            return slot;
        }
        return {};
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
    void wait_first(const std::shared_ptr<scheduled_slot> & slot) {
        std::unique_lock<std::mutex> lock(slot->mutex);
        slot->cv.wait(lock, [&] {
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
        slot->cancelled = true;
        slot->release_when_finished = true;
        slot->state = slot_state::cancelled;
        slot->cv.notify_all();
        work_cv_.notify_all();
    }
    void acknowledge_cancel(const std::shared_ptr<scheduled_slot> & slot) {
        std::lock_guard<std::mutex> lock(slot->mutex);
        slot->callback_done = true;
        slot->cv.notify_all();
        work_cv_.notify_all();
    }
    void release(const std::shared_ptr<scheduled_slot> & slot) {
        {
            std::lock_guard<std::mutex> lock(slot->mutex);
            slot->prompt.clear();
            slot->pieces.clear();
            slot->piece_logprobs.clear();
            slot->generated_logprobs.clear();
            slot->generated.clear();
            slot->n_decoded = 0;
            slot->cancelled = false;
            slot->release_when_finished = false;
            slot->callback_done = false;
            slot->state = slot_state::free;
        }
        work_cv_.notify_all();
    }

    json health() const {
        json output = json::array();
        for (const auto & slot : slots_) {
            std::lock_guard<std::mutex> lock(slot->mutex);
            output.push_back(json{
                { "index", slot->index },
                { "state", slot_state_name(slot->state) },
                { "n_decoded", slot->n_decoded },
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
              const std::vector<potluck::token_logprob> & logprobs = {}) {
        std::lock_guard<std::mutex> lock(slot->mutex);
        if (slot->cancelled) {
            return;
        }
        if (!llama_vocab_is_eog(vocab_, token)) {
            slot->generated.push_back(token);
            slot->generated_logprobs.push_back(logprobs);
            slot->pieces.push_back(token_piece(vocab_, token));
            slot->piece_logprobs.push_back(logprobs);
            ++slot->n_decoded;
        }
        slot->last = token;
        slot->next_position = position + 1;
        if (llama_vocab_is_eog(vocab_, token) || slot->n_decoded >= slot->n_predict) {
            slot->finished = true;
            slot->state = slot_state::done;
        } else {
            slot->state = slot_state::decode;
        }
        slot->cv.notify_all();
    }


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
                    (void) drive_batch(ring_, empty, empty, empty, clear_slot->seq,
                                       -1, -1, 0, nullptr,
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

        std::vector<batch_item> prefill_body;
        std::vector<batch_item> prefill_logits;
        std::vector<batch_item> decode;
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

        std::vector<batch_item> items;
        items.reserve(prefill_body.size() + decode.size() + prefill_logits.size());
        items.insert(items.end(), prefill_body.begin(), prefill_body.end());
        items.insert(items.end(), decode.begin(), decode.end());
        items.insert(items.end(), prefill_logits.begin(), prefill_logits.end());
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
        const uint32_t n_logits = static_cast<uint32_t>(decode.size() + prefill_logits.size());
        const int32_t clear_seq = clear_slot ? clear_slot->seq : -1;
        const auto batch_cancelled = [&]() {
            if (is_stopping()) {
                return true;
            }
            for (const auto & slot : selected) {
                std::lock_guard<std::mutex> lock(slot->mutex);
                if (slot->cancelled) {
                    return true;
                }
            }
            return false;
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
        try {
            result = drive_batch(
                ring_, positions, sequences, tokens, clear_seq, -1, -1, n_logits,
                nullptr, batch_cancelled, &batch_started, heartbeat_, &result_logprobs);
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
            slot->cancelled = false;
            slot->state = slot_state::done;
            slot->finished = true;
            slot->error = error.empty() ? "ring recovery exhausted" : error;
            slot->release_when_finished = false;
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
                                       std::chrono::seconds(5);
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
                            (slot->state == slot_state::cancelled && !slot->finished)) {
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
                        (slot->state == slot_state::cancelled && !slot->finished)) {
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
                    next_topology_check_ = now + std::chrono::seconds(5);
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
                                   (slot->state == slot_state::cancelled && !slot->finished)) {
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
                                           std::chrono::seconds(5);
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
                       uint32_t n_predict_default, float temp, float top_p, uint32_t seed);
