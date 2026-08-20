#include "llama.h"
#include "potluck-transport.h"

#include <algorithm>
#include <cassert>
#include <chrono>
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

struct options {
    std::string role;
    std::string model;
    std::string host = "127.0.0.1";
    uint16_t port = 39271;
};

options parse_options(int argc, char ** argv) {
    CHECK(argc >= 3);
    options result;
    result.role = argv[1];
    result.model = argv[2];
    if (argc >= 4) {
        result.port = static_cast<uint16_t>(std::stoi(argv[3]));
    }
    if (argc >= 5) {
        result.host = argv[4];
    }
    CHECK(result.role == "head" || result.role == "worker");
    return result;
}

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

llama_model_params model_params(uint32_t start, uint32_t end) {
    llama_model_params params = llama_model_default_params();
    params.potluck_layer_start = start;
    params.potluck_layer_end = end;
    return params;
}

llama_model * load_model(const std::string & path, uint32_t start, uint32_t end) {
    llama_backend_init();
    llama_model_params params = model_params(start, end);
    llama_model * model = llama_model_load_from_file(path.c_str(), params);
    CHECK(model != nullptr);
    return model;
}

int run_worker(const options & opts) {
    constexpr uint32_t split = 12;
    // Bind before loading the split: the Metal kernel compile takes minutes on
    // some machines, and the head's connect-retry budget is only 60 s. This
    // mirrors the real potluck-worker, which binds its listener before model
    // init so the head can connect during setup and its commands queue up.
    potluck::tcp_listener listener = potluck::tcp_listener::bind_host(opts.host, opts.port);
    CHECK(listener.valid());
    std::printf("LISTENING %u\n", listener.port());
    std::fflush(stdout);

    llama_model * model = load_model(opts.model, split, 0);
    const uint32_t n_embd = llama_model_n_embd(model);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    CHECK(llama_model_n_layer(model) == 24);

    // Accept the first *well-formed* connection. Stray localhost probe daemons
    // (and other garbage traffic) may hit this port; discard those connections
    // and keep listening until the real head connects with a valid frame.
    std::string error;
    potluck::tcp_channel peer;
    for (;;) {
        peer = listener.accept(error);
        if (!peer.valid()) {
            std::fprintf(stderr, "worker accept failed: %s\n", error.c_str());
            return 1;
        }
        potluck::message incoming;
        if (peer.receive(incoming, error)) {
            if (incoming.type == potluck::message_type::hidden_state &&
                incoming.dtype == potluck::data_type::f32 &&
                incoming.shape == std::vector<uint64_t>({1, n_embd})) {
                break;  // real head connection
            }
            std::fprintf(stderr, "worker: discarding malformed connection (type=%d)\n",
                         static_cast<int>(incoming.type));
        } else {
            std::fprintf(stderr, "worker: discarding connection: %s\n", error.c_str());
        }
        peer = potluck::tcp_channel{};  // close the bogus socket and retry
    }

    potluck::message incoming;
    if (!peer.receive(incoming, error)) {
        std::fprintf(stderr, "worker receive failed: %s\n", error.c_str());
        return 1;
    }
    CHECK(incoming.type == potluck::message_type::hidden_state);
    CHECK(incoming.dtype == potluck::data_type::f32);
    CHECK(incoming.shape == std::vector<uint64_t>({1, n_embd}));

    llama_context * context = llama_init_from_model(model, context_params(split, 0, false));
    CHECK(context != nullptr);
    llama_batch batch = llama_batch_init(1, n_embd, 1);
    for (int step = 0; step < 2; ++step) {
        if (step > 0) {
            if (!peer.receive(incoming, error)) {
                std::fprintf(stderr, "worker receive step %d failed: %s\\n", step, error.c_str());
                return 1;
            }
        }
        CHECK(incoming.type == potluck::message_type::hidden_state);
        CHECK(incoming.dtype == potluck::data_type::f32);
        CHECK(incoming.sequence == static_cast<uint64_t>(7 + step));
        CHECK(incoming.shape == std::vector<uint64_t>({1, n_embd}));
        set_hidden_batch(batch, incoming.payload, n_embd);
        batch.pos[0] = step;
        CHECK(llama_decode(context, batch) == 0);
        const float * logits = llama_get_logits_ith(context, 0);
        CHECK(logits != nullptr);

        const uint32_t token = static_cast<uint32_t>(argmax(logits, n_vocab));
        potluck::message response;
        response.type = potluck::message_type::token;
        response.rank = 1;
        response.sequence = incoming.sequence;
        response.dtype = potluck::data_type::i32;
        response.shape = {1};
        response.payload.resize(sizeof(token));
        std::memcpy(response.payload.data(), &token, sizeof(token));
        CHECK(peer.send(response, error));
    }

    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

int run_head(const options & opts) {
    constexpr uint32_t split = 12;
    llama_model * model = load_model(opts.model, 0, split);
    const uint32_t n_embd = llama_model_n_embd(model);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    CHECK(llama_model_n_layer(model) == 24);
    const llama_token input = llama_vocab_bos(llama_model_get_vocab(model));

    std::string error;
    // The worker may still be loading its model when the head starts, so retry
    // the connection for a few seconds instead of failing on the first refused
    // attempt. This is a robustness fix for the two-process launch, not a change
    // to the distributed protocol.
    potluck::tcp_channel client;
    for (int attempt = 0; attempt < 600; ++attempt) {
        client = potluck::tcp_channel::connect_host(opts.host, opts.port, error);
        if (client.valid()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!client.valid()) {
        std::fprintf(stderr, "head connect failed: %s\\n", error.c_str());
        return 1;
    }

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
    std::vector<uint32_t> network_tokens;
    network_tokens.reserve(2);
    for (int step = 0; step < 2; ++step) {
        request.sequence = static_cast<uint64_t>(7 + step);
        if (step > 0) {
            const uint32_t previous = network_tokens.back();
            first_batch.token[0] = static_cast<llama_token>(previous);
            first_batch.pos[0] = step;
            CHECK(llama_decode(first, first_batch) == 0);
            hidden = llama_get_embeddings_ith(first, 0);
            CHECK(hidden != nullptr);
            std::memcpy(request.payload.data(), hidden, request.payload.size());
        }
        CHECK(client.send(request, error));

        potluck::message response;
        CHECK(client.receive(response, error));
        CHECK(response.type == potluck::message_type::token);
        CHECK(response.sequence == request.sequence);
        CHECK(response.payload.size() == sizeof(uint32_t));
        uint32_t network_token = 0;
        std::memcpy(&network_token, response.payload.data(), sizeof(network_token));
        network_tokens.push_back(network_token);
    }

    llama_free(first);
    llama_batch_free(first_batch);

    llama_model_free(model);
    model = load_model(opts.model, 0, 0);
    llama_context * full = llama_init_from_model(model, context_params(0, 0, false));
    CHECK(full != nullptr);
    llama_batch full_batch = llama_batch_init(1, 0, 1);
    set_token_batch(full_batch, input);
    std::vector<int> full_tokens;
    full_tokens.reserve(2);
    for (int step = 0; step < 2; ++step) {
        if (step > 0) {
            full_batch.token[0] = static_cast<llama_token>(full_tokens.back());
            full_batch.pos[0] = step;
        }
        CHECK(llama_decode(full, full_batch) == 0);
        const float * full_logits = llama_get_logits_ith(full, 0);
        CHECK(full_logits != nullptr);
        full_tokens.push_back(argmax(full_logits, n_vocab));
    }

    std::printf("network_tokens=%u,%u full_tokens=%d,%d\n",
        network_tokens[0], network_tokens[1], full_tokens[0], full_tokens[1]);
    CHECK(static_cast<int>(network_tokens[0]) == full_tokens[0]);
    CHECK(static_cast<int>(network_tokens[1]) == full_tokens[1]);

    llama_batch_free(full_batch);
    llama_free(full);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    const options opts = parse_options(argc, argv);
    return opts.role == "worker" ? run_worker(opts) : run_head(opts);
}
