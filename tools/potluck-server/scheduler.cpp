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

namespace {

}

std::vector<int32_t> drive_batch(ServerRing & ring,
                                 const std::vector<int32_t> & positions,
                                 const std::vector<int32_t> & sequences,
                                 const std::vector<int32_t> & tokens,
                                 int32_t clear_seq, int32_t trim_seq, int32_t trim_to,
                                 uint32_t n_logits,
                                 serve_stats * stats,
                                 std::function<bool()> should_cancel,
                                 bool * batch_started) {
    if (batch_started != nullptr) {
        *batch_started = false;
    }
    const bool clear_only = positions.empty() && sequences.empty() && tokens.empty() &&
        clear_seq != -1;
    if ((!clear_only && (positions.empty() || positions.size() != sequences.size() ||
                         positions.size() != tokens.size())) ||
        (clear_only && (trim_seq >= 0 || trim_to >= 0 || n_logits != 0))) {
        throw std::runtime_error("invalid batch dimensions");
    }
    potluck::message input;
    input.type = potluck::message_type::batch_decode;
    input.flags = 0;
    input.sequence = positions.empty() ? 0 : static_cast<uint64_t>(positions.back());
    if (!potluck::encode_batch_payload(positions, sequences, tokens, nullptr, 0,
                                       clear_seq, trim_seq, trim_to, n_logits, input.payload)) {
        throw std::runtime_error("cannot encode ring batch");
    }
    if (stats != nullptr) {
        stats->head_payload_bytes += input.payload.size();
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
        if (ring.result.receive(output, ring.error)) {
            if (cancel_requested) {
                throw request_cancelled("request cancelled");
            }
            break;
        }
        if (ring.error.find("timeout") == std::string::npos) {
            throw std::runtime_error("ring result receiver closed: " + ring.error);
        }
        const auto now = std::chrono::steady_clock::now();
        if (cancel_requested && now >= cancel_deadline) {
            throw std::runtime_error("ring result receiver timed out after cancellation");
        }
        if (now >= deadline) {
            throw std::runtime_error("ring result receiver timed out");
        }
    }
    if (stats != nullptr) {
        stats->head_payload_bytes += output.payload.size();
    }
    if (output.type != potluck::message_type::batch_result) {
        throw std::runtime_error("unexpected ring result message");
    }
    if (output.flags != ring.windows.size()) {
        throw std::runtime_error("ring result stopped before completing its route");
    }
    std::vector<int32_t> result_positions, result_sequences, result_tokens;
    std::vector<float> result_hidden;
    int32_t ignored_clear = -1, ignored_trim_seq = -1, ignored_trim = -1;
    uint32_t ignored_logits = 0;
    if (!potluck::decode_batch_payload(output.payload.data(), output.payload.size(), 0,
                                       ignored_clear, ignored_trim_seq, ignored_trim, ignored_logits,
                                       result_positions, result_sequences, result_tokens,
                                       result_hidden, ring.error)) {
        throw std::runtime_error("cannot decode ring result: " + ring.error);
    }
    if (result_positions != positions || result_sequences != sequences || result_tokens.size() != positions.size()) {
        throw std::runtime_error("ring result entries do not match the request");
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

std::vector<llama_token> serve(ServerRing & ring, const llama_vocab * vocab,
                               const std::vector<llama_token> & prompt,
                               uint32_t n_predict,
                               const std::function<void(const std::string &)> & emit,
                               serve_stats * stats,
                               int32_t sequence_id,
                               potluck::slot_config sampling,
                               uint32_t prefill_batch) {
    if (prompt.empty()) {
        throw std::runtime_error("prompt is empty");
    }
    sampling.seq = sequence_id;
    configure_slot(ring, sampling);
    const uint32_t batch_limit = std::max<uint32_t>(1, prefill_batch);
    const auto prefill_start = std::chrono::steady_clock::now();
    size_t offset = 0;
    while (offset < prompt.size()) {
        const size_t count = std::min<size_t>(batch_limit, prompt.size() - offset);
        std::vector<int32_t> positions(count);
        std::vector<int32_t> sequences(count, sequence_id);
        std::vector<int32_t> tokens(count);
        for (size_t i = 0; i < count; ++i) {
            positions[i] = static_cast<int32_t>(offset + i);
            tokens[i] = static_cast<int32_t>(prompt[offset + i]);
        }
        const bool final_chunk = offset + count == prompt.size();
        (void) drive_batch(ring, positions, sequences, tokens,
                           offset == 0 ? sequence_id : -1, -1, -1,
                           final_chunk ? 1 : 0, stats);
        offset += count;
    }
    if (stats != nullptr) {
        stats->prefill_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - prefill_start).count();
    }

    std::vector<llama_token> generated;
    generated.reserve(n_predict);
    llama_token previous = prompt.back();
    uint32_t position = static_cast<uint32_t>(prompt.size());
    for (uint32_t i = 0; i < n_predict; ++i) {
        const auto decode_start = std::chrono::steady_clock::now();
        const std::vector<int32_t> result = drive_batch(
            ring, { static_cast<int32_t>(position) }, { sequence_id },
            { static_cast<int32_t>(previous) }, -1, -1, -1, 1, stats);
        if (stats != nullptr) {
            stats->decode_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - decode_start).count();
        }
        if (result.empty()) {
            break;
        }
        const llama_token next = static_cast<llama_token>(result.front());
        if (llama_vocab_is_eog(vocab, next)) {
            break;
        }
        generated.push_back(next);
        if (emit) {
            emit(token_piece(vocab, next));
        }
        previous = next;
        ++position;
    }
    return generated;
}

