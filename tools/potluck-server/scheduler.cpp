// potluck-server continuous batching and conversation slots.

#include "internal.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/resource.h>

namespace {

size_t first_stop_offset(const std::string & text, const std::vector<std::string> & stops) {
    size_t offset = std::string::npos;
    for (const std::string & stop : stops) {
        if (stop.empty()) {
            continue;
        }
        const size_t candidate = text.find(stop);
        if (candidate != std::string::npos) {
            offset = std::min(offset, candidate);
        }
    }
    return offset;
}

}
void slot_scheduler::ensure_speculator(const std::shared_ptr<scheduled_slot> & slot) {
    if (!speculative_configured()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(slot->mutex);
        if (slot->speculative != nullptr) {
            return;
        }
    }

    std::vector<std::string> type_names = speculative_config_.types;
    if (type_names.empty()) {
        if (speculative_config_.draft_model.empty()) {
            throw std::runtime_error("speculative decoding requires a draft type or model");
        }
        type_names.push_back("draft-simple");
    }

    const uint32_t configured_draft = speculative_config_.n_draft == 0
        ? 3 : speculative_config_.n_draft;
    common_params params;
    params.n_ctx = static_cast<int32_t>(std::max<uint32_t>(2, speculative_config_.n_ctx));
    params.n_batch = static_cast<int32_t>(std::max<uint32_t>(
        std::max<uint32_t>(32, speculative_config_.n_batch),
        configured_draft + 1));
    params.n_ubatch = static_cast<int32_t>(std::max<uint32_t>(
        32, std::min(speculative_config_.n_ubatch,
                      static_cast<uint32_t>(params.n_batch))));
    params.n_sequences = 1;
    params.n_outputs_max = static_cast<int32_t>(configured_draft + 1);
    params.n_outputs_max_per_seq = params.n_outputs_max;
    params.speculative.types = common_speculative_types_from_names(type_names);
    params.speculative.draft.n_max = static_cast<int32_t>(configured_draft);
    for (const common_speculative_type type : params.speculative.types) {
        switch (type) {
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
            // ngram-simple needs a draft window at least as long as its lookup pattern.
            params.speculative.ngram_simple.size_n =
                static_cast<uint16_t>(std::min<uint32_t>(
                    params.speculative.ngram_simple.size_n, configured_draft));
            params.speculative.ngram_simple.size_m =
                static_cast<uint16_t>(std::min<uint32_t>(configured_draft, UINT16_MAX));
            break;
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            params.speculative.ngram_map_k.size_m =
                static_cast<uint16_t>(std::min<uint32_t>(configured_draft, UINT16_MAX));
            break;
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
            params.speculative.ngram_map_k4v.size_m =
                static_cast<uint16_t>(std::min<uint32_t>(configured_draft, UINT16_MAX));
            break;
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
            params.speculative.ngram_mod.n_max =
                static_cast<int32_t>(std::min<uint32_t>(configured_draft, INT32_MAX));
            params.speculative.ngram_mod.n_min = 0;
            break;
        default:
            break;
        }
    }

    const bool needs_draft_model = std::find(
        params.speculative.types.begin(), params.speculative.types.end(),
        COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE) != params.speculative.types.end();
    common_speculative_init_result_ptr draft_init;
    if (needs_draft_model) {
        if (speculative_config_.draft_model.empty()) {
            throw std::runtime_error("draft-simple requires --spec-draft-model");
        }
        if (target_model_ == nullptr) {
            throw std::runtime_error(
                "draft-simple requires the target model metadata on the head");
        }
        params.model.path = speculative_config_.draft_model;
        params.speculative.draft.mparams.path = speculative_config_.draft_model;
        if (speculative_target_context_ == nullptr) {
            llama_context_params target_params = llama_context_default_params();
            target_params.n_ctx = 2;
            target_params.n_batch = 2;
            target_params.n_ubatch = 2;
            target_params.n_seq_max = 1;
            target_params.n_outputs_max = 1;
            target_params.n_threads = 1;
            target_params.n_threads_batch = 1;
            target_params.no_perf = true;
            speculative_target_context_ = llama_init_from_model(
                const_cast<llama_model *>(target_model_), target_params);
            if (speculative_target_context_ == nullptr) {
                throw std::runtime_error(
                    "cannot create target metadata context for speculation");
            }
        }
        params.speculative.draft.ctx_tgt = speculative_target_context_;
        draft_init = common_speculative_init_from_params(
            params, const_cast<llama_model *>(target_model_),
            speculative_target_context_);
        if (draft_init == nullptr || draft_init->model() == nullptr ||
            draft_init->context() == nullptr) {
            throw std::runtime_error("cannot initialize speculative draft model");
        }
        params.speculative.draft.ctx_dft = draft_init->context();
    }

    common_speculative_ptr spec(
        common_speculative_init(params.speculative, 1));
    if (spec == nullptr) {
        throw std::runtime_error("cannot initialize speculative decoder");
    }
    {
        std::lock_guard<std::mutex> lock(slot->mutex);
        if (slot->speculative == nullptr) {
            slot->speculative_init = std::move(draft_init);
            slot->speculative = std::move(spec);
            slot->speculative_started = false;
        }
    }
}
void slot_scheduler::prime_speculative_prompt_tail(
        const std::shared_ptr<scheduled_slot> & slot) {
    if (slot->speculative == nullptr) {
        return;
    }
    llama_token token = 0;
    int32_t position = -1;
    {
        std::lock_guard<std::mutex> lock(slot->mutex);
        if (!slot->speculative_prompt_tail_pending) {
            return;
        }
        token = slot->speculative_prompt_tail;
        position = slot->speculative_prompt_tail_position;
        if (position < 0 || slot->next_position != static_cast<uint32_t>(position + 1)) {
            throw std::runtime_error("speculative prompt tail position is not next");
        }
    }
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, token, static_cast<llama_pos>(position), { 0 }, false);
    const bool processed = common_speculative_process(slot->speculative.get(), batch);
    llama_batch_free(batch);
    if (!processed) {
        throw std::runtime_error("speculative draft context failed to process prompt tail");
    }
    std::lock_guard<std::mutex> lock(slot->mutex);
    slot->speculative_prompt_tail_pending = false;
}


void slot_scheduler::process_speculative_batch(
        const std::shared_ptr<scheduled_slot> & slot,
        const std::vector<int32_t> & positions,
        const std::vector<int32_t> & tokens,
        bool prompt_complete,
        uint32_t accepted_count,
        int32_t trim_to) {
    if (slot->speculative == nullptr) {
        return;
    }
    if (positions.empty() || positions.size() != tokens.size()) {
        throw std::runtime_error("invalid speculative batch dimensions");
    }
    common_speculative * spec = slot->speculative.get();
    std::vector<int32_t> process_positions = positions;
    std::vector<int32_t> process_tokens = tokens;
    std::vector<llama_token> begin_prompt;
    bool begin_generation = false;
    llama_token prompt_tail = 0;
    int32_t prompt_tail_position = -1;
    {
        std::lock_guard<std::mutex> lock(slot->mutex);
        begin_generation = prompt_complete && !slot->speculative_started;
        if (begin_generation) {
            if (slot->prompt.empty() ||
                process_positions.back() != static_cast<int32_t>(slot->prompt.size() - 1) ||
                process_tokens.back() != static_cast<int32_t>(slot->prompt.back())) {
                throw std::runtime_error("speculative prompt tail does not match slot state");
            }
            begin_prompt.assign(slot->prompt.begin(), slot->prompt.end() - 1);
            prompt_tail = static_cast<llama_token>(process_tokens.back());
            prompt_tail_position = process_positions.back();
            process_positions.pop_back();
            process_tokens.pop_back();
        } else if (slot->speculative_prompt_tail_pending) {
            prompt_tail = slot->speculative_prompt_tail;
            prompt_tail_position = slot->speculative_prompt_tail_position;
            if (process_positions.front() != prompt_tail_position + 1 ||
                slot->next_position != static_cast<uint32_t>(process_positions.front())) {
                throw std::runtime_error("speculative draft context position is not contiguous");
            }
            process_positions.insert(process_positions.begin(), prompt_tail_position);
            process_tokens.insert(process_tokens.begin(), static_cast<int32_t>(prompt_tail));
            slot->speculative_prompt_tail_pending = false;
        }
    }
    if (!process_positions.empty()) {
        llama_batch batch = llama_batch_init(process_positions.size(), 0, 1);
        for (size_t i = 0; i < process_positions.size(); ++i) {
            common_batch_add(batch, static_cast<llama_token>(process_tokens[i]),
                             static_cast<llama_pos>(process_positions[i]), { 0 }, false);
        }
        const bool processed = common_speculative_process(spec, batch);
        llama_batch_free(batch);
        if (!processed) {
            throw std::runtime_error("speculative draft context failed to process ring batch");
        }
    }
    if (begin_generation) {
        common_speculative_begin(spec, 0, begin_prompt);
        std::lock_guard<std::mutex> lock(slot->mutex);
        slot->speculative_started = true;
        slot->speculative_prompt_tail_pending = true;
        slot->speculative_prompt_tail = prompt_tail;
        slot->speculative_prompt_tail_position = prompt_tail_position;
    }
    if (trim_to >= 0) {
        if (accepted_count > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("speculative acceptance count exceeds sampler limit");
        }
        common_speculative_accept(spec, 0, static_cast<uint16_t>(accepted_count));
        if (slot->speculative_init != nullptr &&
            slot->speculative_init->context() != nullptr &&
            !llama_memory_seq_rm(
                llama_get_memory(slot->speculative_init->context()), 0, trim_to, -1)) {
            throw std::runtime_error("cannot trim speculative draft sequence");
        }
    }
}


void slot_scheduler::emit(const std::shared_ptr<scheduled_slot> & slot, llama_token token,
                          uint32_t position,
                          const std::vector<potluck::token_logprob> & logprobs) {
    std::lock_guard<std::mutex> lock(slot->mutex);
    if (slot->cancelled || slot->finished) {
        return;
    }
    if (!llama_vocab_is_eog(vocab_, token)) {
        const std::string piece = token_piece(vocab_, token);
        slot->generated.push_back(token);
        slot->generated_logprobs.push_back(logprobs);
        slot->pieces.push_back(piece);
        slot->piece_logprobs.push_back(logprobs);
        slot->generated_text += piece;
        ++slot->n_decoded;
        if (slot->stop_offset == std::string::npos) {
            slot->stop_offset = first_stop_offset(slot->generated_text, slot->stops);
        }
    }
    slot->last = token;
    slot->next_position = position + 1;
    if (llama_vocab_is_eog(vocab_, token) ||
        slot->stop_offset != std::string::npos ||
        slot->n_decoded >= slot->n_predict) {
        slot->finished = true;
        slot->state = slot_state::done;
    } else {
        slot->state = slot_state::decode;
    }
    slot->cv.notify_all();
}


std::vector<int32_t> drive_ring_cycle(ServerRing & ring,
                                      const std::vector<int32_t> & positions,
                                      const std::vector<int32_t> & sequences,
                                      const std::vector<int32_t> & tokens,
                                      int32_t clear_seq, int32_t trim_seq, int32_t trim_to,
                                      uint32_t n_logits,
                                      std::function<bool()> should_cancel,
                                      bool * batch_started,
                                      std::function<bool(std::string &)> heartbeat,
                                      potluck::batch_logprobs * result_logprobs,
                                      const std::vector<int32_t> & draft_tokens,
                                      uint32_t accepted_count,
                                      uint32_t * result_accepted_count) {
    if (result_logprobs != nullptr) {
        result_logprobs->clear();
    }
    if (result_accepted_count != nullptr) {
        *result_accepted_count = 0;
    }
    if (batch_started != nullptr) {
        *batch_started = false;
    }
    const bool clear_only = positions.empty() && sequences.empty() && tokens.empty() &&
        clear_seq != -1;
    if ((!clear_only && (positions.empty() || positions.size() != sequences.size() ||
                         positions.size() != tokens.size())) ||
        (clear_only && (trim_seq >= 0 || trim_to >= 0 || n_logits != 0 ||
                        !draft_tokens.empty() || accepted_count != 0)) ||
        accepted_count > draft_tokens.size()) {
        throw std::runtime_error("invalid batch dimensions");
    }
    potluck::message input;
    input.type = potluck::message_type::batch_decode;
    input.flags = 0;
    input.sequence = ++ring.batch_sequence;
    if (!potluck::encode_batch_payload(positions, sequences, tokens, draft_tokens,
                                       accepted_count, nullptr, 0, clear_seq, trim_seq,
                                       trim_to, n_logits, input.payload)) {
        throw std::runtime_error("cannot encode ring batch");
    }
    const auto cancelled = [&]() {
        return should_cancel && should_cancel();
    };
    if (cancelled()) {
        throw request_cancelled("request cancelled");
    }
    constexpr int poll_timeout_ms = 250;
    if (!ring.ingress.set_send_timeout(poll_timeout_ms, ring.error)) {
        throw std::runtime_error("cannot set ring ingress timeout: " + ring.error);
    }
    if (!ring.ingress.send(input, ring.error)) {
        if (cancelled()) {
            throw request_cancelled("request cancelled");
        }
        throw std::runtime_error("cannot inject ring batch: " + ring.error);
    }
    if (batch_started != nullptr) {
        *batch_started = true;
    }
    if (!ring.result.set_receive_timeout(poll_timeout_ms, ring.error)) {
        throw std::runtime_error("cannot set ring result timeout: " + ring.error);
    }
    potluck::message output;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(600);
    bool cancel_requested = false;
    std::chrono::steady_clock::time_point cancel_deadline;
    for (;;) {
        if (!cancel_requested && cancelled()) {
            cancel_requested = true;
            cancel_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        }
        bool received = false;
        if (!ring.pending_results.empty()) {
            output = std::move(ring.pending_results.front());
            ring.pending_results.pop_front();
            received = true;
        } else {
            received = ring.result.receive(output, ring.error);
        }
        if (received) {
            if (output.type == potluck::message_type::heartbeat_ack) {
                continue;
            }
            if (output.type == potluck::message_type::error) {
                const std::string detail(output.payload.begin(), output.payload.end());
                throw std::runtime_error(
                    "worker " + std::to_string(output.rank) + " failed: " +
                    (detail.empty() ? std::string("no detail") : detail));
            }
            if (output.sequence < input.sequence) {
                continue;
            }
            if (output.sequence != input.sequence) {
                throw std::runtime_error("ring result sequence is ahead of the active batch");
            }
            if (cancel_requested) {
                throw request_cancelled("request cancelled");
            }
            break;
        }
        if (ring.error.find("timeout") == std::string::npos) {
            throw std::runtime_error("ring result receiver closed: " + ring.error);
        }
        if (heartbeat && !cancel_requested) {
            std::string heartbeat_error;
            if (!heartbeat(heartbeat_error)) {
                throw std::runtime_error("worker transport lost: " +
                                         (heartbeat_error.empty()
                                              ? std::string("heartbeat failed")
                                              : heartbeat_error));
            }
        }
        const auto now = std::chrono::steady_clock::now();
        if (cancel_requested && now >= cancel_deadline) {
            throw request_cancelled("request cancelled");
        }
        if (now >= deadline) {
            throw std::runtime_error("ring result receiver timed out");
        }
    }
    const bool has_logprobs =
        output.type == potluck::message_type::batch_result_logprobs;
    if (!has_logprobs && output.type != potluck::message_type::batch_result) {
        throw std::runtime_error("unexpected ring result message");
    }
    if (output.flags != ring.windows.size()) {
        throw std::runtime_error("ring result stopped before completing its route");
    }
    size_t base_payload_size = output.payload.size();
    if (has_logprobs) {
        if (output.shape.size() != 1 || output.shape[0] > output.payload.size()) {
            throw std::runtime_error("ring result has invalid logprob metadata");
        }
        base_payload_size = static_cast<size_t>(output.shape[0]);
    }
    std::vector<int32_t> result_positions, result_sequences, result_tokens;
    std::vector<int32_t> result_draft_tokens;
    std::vector<float> result_hidden;
    int32_t ignored_clear = -1, ignored_trim_seq = -1, ignored_trim = -1;
    uint32_t ignored_logits = 0;
    uint32_t ignored_accepted = 0;
    if (!potluck::decode_batch_payload(
            output.payload.data(), base_payload_size, 0,
            ignored_clear, ignored_trim_seq, ignored_trim, ignored_logits,
            result_positions, result_sequences, result_tokens, result_draft_tokens,
            ignored_accepted, result_hidden, ring.error)) {
        throw std::runtime_error("cannot decode ring result: " + ring.error);
    }
    if (has_logprobs) {
        potluck::batch_logprobs decoded;
        if (!potluck::decode_batch_logprobs(
                output.payload.data() + base_payload_size,
                output.payload.size() - base_payload_size, decoded, ring.error) ||
            decoded.size() != ignored_logits) {
            throw std::runtime_error("cannot decode ring result logprobs: " + ring.error);
        }
        if (result_logprobs != nullptr) {
            *result_logprobs = std::move(decoded);
        }
    }
    if (result_positions != positions || result_sequences != sequences ||
        result_tokens.size() != positions.size() ||
        result_draft_tokens != draft_tokens ||
        ignored_accepted > draft_tokens.size()) {
        throw std::runtime_error("ring result entries do not match the request");
    }
    if (result_accepted_count != nullptr) {
        *result_accepted_count = ignored_accepted;
    }
    return result_tokens;
}

std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & text) {
    if (text.empty()) {
        throw std::runtime_error("prompt is empty");
    }
    return common_tokenize(vocab, text, true, true);
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    return common_token_to_piece(vocab, token, true);
}

std::string render_tokens(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string text;
    for (const llama_token token : tokens) {
        text += token_piece(vocab, token);
    }
    return text;
}
double peak_rss_mb() {
    rusage usage = {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0.0;
    }
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
}

void configure_slot(ServerRing & ring, const potluck::slot_config & config) {
    potluck::message message;
    message.type = potluck::message_type::slot_config;
    message.flags = 0;
    message.sequence = static_cast<uint64_t>(std::max<int32_t>(0, config.seq));
    if (!potluck::encode_slot_config(config, message.payload)) {
        throw std::runtime_error("cannot encode slot configuration");
    }
    constexpr int poll_timeout_ms = 250;
    if (!ring.ingress.set_send_timeout(poll_timeout_ms, ring.error)) {
        throw std::runtime_error("cannot set ring ingress timeout: " + ring.error);
    }
    if (!ring.ingress.send(message, ring.error)) {
        throw std::runtime_error("cannot send slot configuration: " + ring.error);
    }
}
