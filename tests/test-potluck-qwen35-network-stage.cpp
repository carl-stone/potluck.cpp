// Historical correctness check for the legacy raw-TCP stage path. It does not
// cover the direct-peer ZeroMQ ring required by ADR 0007.

#include "llama.h"
#include "potluck-transport.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// Runtime check that survives Release builds (assert() is compiled out under
// -DNDEBUG, which would silently skip the side-effecting llama_decode / output
// reads these tests rely on). On failure we print and abort so the test fails
// loudly instead of producing undefined behaviour.
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond,          \
                        __FILE__, __LINE__);                                    \
            std::abort();                                                       \
        }                                                                      \
    } while (0)

namespace {

void set_token_batch(llama_batch & batch, llama_token token) {
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
}

void set_hidden_batch(llama_batch & batch, const std::vector<uint8_t> & payload, uint32_t n_embd) {
    batch.n_tokens = 1;
    CHECK(payload.size() == sizeof(float) * n_embd);
    std::memcpy(batch.embd, payload.data(), payload.size());
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
}

int argmax(const float * values, uint32_t count) {
    return static_cast<int>(std::max_element(values, values + count) - values);
}

llama_context_params context_params(uint32_t start, uint32_t end, bool embeddings) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 128;
    params.n_batch = 1;
    params.n_ubatch = 1;
    params.embeddings = embeddings;
    params.potluck_layer_start = start;
    params.potluck_layer_end = end;
    return params;
}

} // namespace

int main(int argc, char ** argv) {
    const std::string model_path = argc > 1 ? argv[1] : "/tmp/potluck.cpp/models/Qwen3.5-0.8B-Q4_0.gguf";

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    CHECK(model != nullptr);

    const uint32_t n_embd = llama_model_n_embd(model);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const llama_token input = llama_vocab_bos(llama_model_get_vocab(model));
    const uint32_t split = llama_model_n_layer(model) / 2;

    potluck::tcp_listener listener = potluck::tcp_listener::bind_loopback(0);
    CHECK(listener.valid());

    int remote_token = -1;
    std::thread stage_thread([&] {
        std::string error;
        potluck::tcp_channel peer = listener.accept(error);
        CHECK(peer.valid());

        potluck::message incoming;
        CHECK(peer.receive(incoming, error));
        CHECK(incoming.type == potluck::message_type::hidden_state);
        CHECK(incoming.dtype == potluck::data_type::f32);
        CHECK(incoming.shape == std::vector<uint64_t>({1, n_embd}));

        llama_context * second = llama_init_from_model(model, context_params(split, 0, false));
        CHECK(second != nullptr);
        llama_batch batch = llama_batch_init(1, n_embd, 1);
        set_hidden_batch(batch, incoming.payload, n_embd);
        CHECK(llama_decode(second, batch) == 0);
        const float * logits = llama_get_logits_ith(second, 0);
        CHECK(logits != nullptr);
        remote_token = argmax(logits, n_vocab);

        potluck::message response;
        response.type = potluck::message_type::token;
        response.rank = 1;
        response.sequence = incoming.sequence;
        response.dtype = potluck::data_type::i32;
        response.shape = {1};
        response.payload.resize(sizeof(uint32_t));
        const uint32_t token = static_cast<uint32_t>(remote_token);
        std::memcpy(response.payload.data(), &token, sizeof(token));
        CHECK(peer.send(response, error));

        llama_batch_free(batch);
        llama_free(second);
    });

    std::string error;
    potluck::tcp_channel client = potluck::tcp_channel::connect_loopback(listener.port(), error);
    CHECK(client.valid());

    llama_context * first = llama_init_from_model(model, context_params(0, split, true));
    CHECK(first != nullptr);
    llama_batch first_batch = llama_batch_init(1, 0, 1);
    set_token_batch(first_batch, input);
    CHECK(llama_decode(first, first_batch) == 0);
    const float * hidden = llama_get_embeddings_ith(first, 0);
    CHECK(hidden != nullptr);

    potluck::message request;
    request.type = potluck::message_type::hidden_state;
    request.rank = 0;
    request.sequence = 7;
    request.dtype = potluck::data_type::f32;
    request.shape = {1, n_embd};
    request.payload.resize(sizeof(float) * n_embd);
    std::memcpy(request.payload.data(), hidden, request.payload.size());
    CHECK(client.send(request, error));

    potluck::message response;
    CHECK(client.receive(response, error));
    CHECK(response.type == potluck::message_type::token);
    CHECK(response.sequence == request.sequence);
    CHECK(response.payload.size() == sizeof(uint32_t));
    uint32_t network_token = 0;
    std::memcpy(&network_token, response.payload.data(), sizeof(network_token));

    llama_context * full = llama_init_from_model(model, context_params(0, 0, false));
    CHECK(full != nullptr);
    llama_batch full_batch = llama_batch_init(1, 0, 1);
    set_token_batch(full_batch, input);
    CHECK(llama_decode(full, full_batch) == 0);
    const float * full_logits = llama_get_logits_ith(full, 0);
    CHECK(full_logits != nullptr);
    const int full_token = argmax(full_logits, n_vocab);

    std::printf("network_token=%u stage_token=%d full_token=%d\n", network_token, remote_token, full_token);
    CHECK(static_cast<int>(network_token) == full_token);

    stage_thread.join();
    llama_batch_free(full_batch);
    llama_batch_free(first_batch);
    llama_free(full);
    llama_free(first);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
