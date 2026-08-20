// prima-stage: a single-machine prima-style distributed pipeline for Qwen3.5.
//
// One machine runs two stages that together cover every model layer:
//   head  : layers [0, split)          -> emits hidden state, no LM head
//   worker: layers [split, n_layer)    -> consumes hidden state, runs LM head
//
// The stages communicate over a POSIX TCP loopback channel using the versioned
// prima protocol. A handshake fixes the hidden-state contract once, then the
// head prefills the prompt, decodes autoregressively, and forwards each produced
// hidden state to the worker, which returns the greedy next token. Generation
// stops at EOS or the requested token budget.
//
// This is the finished single-machine slice of the prima runtime: prompt prefill,
// full multi-token generation, a bounded handshake, and robust error propagation.
// It deliberately does not add a scheduler, CUDA, Windows, or multi-machine ring.

#include "llama.h"
#include "prima-distributed-transport.h"
#include "prima_runtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

const uint16_t default_port = 39271;

struct options {
    std::string model;
    std::string host = "127.0.0.1";
    uint16_t port = default_port;
    uint32_t split = 0;
    uint32_t n_predict = 64;
    std::string prompt;
};

options parse_options(int argc, char ** argv) {
    options opts;
    opts.prompt = "The capital of France is";
    if (argc > 1) opts.model = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc) { opts.host = argv[++i]; }
        else if (arg == "--port" && i + 1 < argc) { opts.port = static_cast<uint16_t>(std::stoi(argv[++i])); }
        else if ((arg == "--split" || arg == "-s") && i + 1 < argc) { opts.split = static_cast<uint32_t>(std::stoi(argv[++i])); }
        else if ((arg == "--n-predict" || arg == "-n") && i + 1 < argc) { opts.n_predict = static_cast<uint32_t>(std::stoi(argv[++i])); }
        else if ((arg == "--prompt" || arg == "-p") && i + 1 < argc) { opts.prompt = argv[++i]; }
        else { std::fprintf(stderr, "ignoring unknown argument: %s\n", arg.c_str()); }
    }
    return opts;
}

// Worker stage: listens, performs a handshake, then serves hidden states until
// the head closes the connection or signals an error.
int run_worker(const options & opts) {
    prima::stage_model sm;
    std::string error;
    if (!prima::stage_load(sm, opts.model, opts.split, 0, /*embeddings=*/false, /*n_ctx=*/2048, error)) {
        std::fprintf(stderr, "worker: %s\n", error.c_str());
        return 1;
    }
    if (opts.split == 0 || opts.split >= sm.n_layer) {
        std::fprintf(stderr, "worker: invalid split %u for model with %u layers (need 0 < split < n_layer)\n",
                     opts.split, sm.n_layer);
        prima::stage_free(sm);
        return 1;
    }

    prima::tcp_listener listener = prima::tcp_listener::bind_host(opts.host, opts.port);
    if (!listener.valid()) {
        std::fprintf(stderr, "worker: failed to bind %s:%u\n", opts.host.c_str(), opts.port);
        prima::stage_free(sm);
        return 1;
    }
    std::printf("LISTENING %u\n", listener.port());
    std::fflush(stdout);

    std::string conn_error;
    prima::tcp_channel peer = listener.accept(conn_error);
    if (!peer.valid()) {
        std::fprintf(stderr, "worker: accept failed: %s\n", conn_error.c_str());
        prima::stage_free(sm);
        return 1;
    }

    // Handshake: receive the hello, echo it back.
    prima::message hello;
    if (!peer.receive(hello, error) || hello.type != prima::message_type::hello) {
        std::fprintf(stderr, "worker: handshake receive failed: %s\n", error.c_str());
        return 1;
    }
    if (!peer.send(hello, error)) {
        std::fprintf(stderr, "worker: handshake echo failed: %s\n", error.c_str());
        return 1;
    }

    while (true) {
        prima::message incoming;
        if (!peer.receive(incoming, error)) {
            break; // head closed or errored
        }
        if (incoming.type == prima::message_type::error) {
            std::fprintf(stderr, "worker: head reported error: %.*s\n",
                         static_cast<int>(incoming.payload.size()),
                         incoming.payload.empty() ? "" : reinterpret_cast<const char *>(incoming.payload.data()));
            break;
        }
        if (incoming.type != prima::message_type::hidden_state) {
            continue; // ignore unknown control messages
        }
        if (incoming.dtype != prima::data_type::f32 || incoming.shape != std::vector<uint64_t>({1, sm.n_embd})) {
            prima::message err;
            err.type = prima::message_type::error;
            const char * msg = "protocol: bad hidden-state contract";
            err.payload.assign(msg, msg + std::strlen(msg));
            peer.send(err, error);
            break;
        }

        if (prima::stage_decode_hidden(sm, reinterpret_cast<const float *>(incoming.payload.data()),
                                       static_cast<uint32_t>(incoming.sequence)) != 0) {
            std::fprintf(stderr, "worker: decode failed at seq %llu\n",
                         static_cast<unsigned long long>(incoming.sequence));
            break;
        }
        const float * logits = llama_get_logits_ith(sm.ctx, 0);
        if (logits == nullptr) {
            std::fprintf(stderr, "worker: no logits\n");
            break;
        }
        const uint32_t token = static_cast<uint32_t>(prima::argmax_token(logits, sm.n_vocab));

        prima::message response;
        response.type = prima::message_type::token;
        response.rank = 1;
        response.sequence = incoming.sequence;
        response.dtype = prima::data_type::i32;
        response.shape = {1};
        response.payload.resize(sizeof(token));
        std::memcpy(response.payload.data(), &token, sizeof(token));
        if (!peer.send(response, error)) {
            std::fprintf(stderr, "worker: send failed: %s\n", error.c_str());
            break;
        }
    }

    prima::stage_free(sm);
    return 0;
}

// Forwards the head's current hidden state to the worker for absolute position
// `seq`, and returns true on success.
bool forward_hidden(prima::tcp_channel & channel, prima::stage_model & head, uint64_t seq,
                    std::string & error, uint32_t n_embd) {
    const float * hidden = llama_get_embeddings_ith(head.ctx, 0);
    if (hidden == nullptr) {
        error = "head produced no hidden state";
        return false;
    }
    prima::message req;
    req.type = prima::message_type::hidden_state;
    req.rank = 0;
    req.sequence = seq;
    req.dtype = prima::data_type::f32;
    req.shape = {1, n_embd};
    req.payload.resize(sizeof(float) * n_embd);
    std::memcpy(req.payload.data(), hidden, req.payload.size());
    return channel.send(req, error);
}

// Head stage + coordinator. When a split is supplied it spawns the worker in a
// background thread and drives prefill + generation against it.
int run_head(const options & opts) {
    const uint32_t split = opts.split;
    const bool distributed = split > 0;

    std::thread worker_thread;
    if (distributed) {
        worker_thread = std::thread([&] { run_worker(opts); });
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    prima::stage_model head;
    std::string error;
    const uint32_t head_end = distributed ? split : 0;
    if (!prima::stage_load(head, opts.model, 0, head_end, /*embeddings=*/distributed, /*n_ctx=*/2048, error)) {
        std::fprintf(stderr, "head: %s\n", error.c_str());
        if (distributed) worker_thread.join();
        return 1;
    }
    if (distributed && (split == 0 || split >= head.n_layer)) {
        std::fprintf(stderr, "head: invalid split %u for model with %u layers\n", split, head.n_layer);
        prima::stage_free(head);
        worker_thread.join();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(head.model);
    const llama_token bos = llama_vocab_bos(vocab);

    // Tokenize the prompt (no automatic BOS; Qwen3.5 adds none, so we prepend it
    // explicitly so the full-model reference uses the identical token sequence).
    std::vector<llama_token> prompt_tokens(static_cast<size_t>(head.n_vocab));
    const int32_t n_prompt = llama_tokenize(vocab, opts.prompt.c_str(),
                                            static_cast<int32_t>(opts.prompt.size()),
                                            prompt_tokens.data(),
                                            static_cast<int32_t>(prompt_tokens.size()),
                                            /*add_special=*/false, /*parse_special=*/false);
    if (n_prompt < 0) {
        std::fprintf(stderr, "head: prompt tokenization failed\n");
        prima::stage_free(head);
        if (distributed) worker_thread.join();
        return 1;
    }
    prompt_tokens.resize(static_cast<size_t>(n_prompt));
    prompt_tokens.insert(prompt_tokens.begin(), bos);

    // Establish the link to the worker (distributed only).
    prima::tcp_channel channel;
    if (distributed) {
        channel = prima::tcp_channel::connect_host(opts.host, opts.port, error);
        if (!channel.valid()) {
            std::fprintf(stderr, "head: could not connect to worker at %s:%u: %s\n",
                         opts.host.c_str(), opts.port, error.c_str());
            prima::stage_free(head);
            worker_thread.join();
            return 1;
        }
        prima::message hello;
        hello.type = prima::message_type::hello;
        hello.rank = 0;
        hello.sequence = 0;
        hello.dtype = prima::data_type::f32;
        hello.shape = {1, head.n_embd};
        if (!channel.send(hello, error) || !channel.receive(hello, error) ||
            hello.type != prima::message_type::hello) {
            std::fprintf(stderr, "head: handshake failed: %s\n", error.c_str());
            prima::stage_free(head);
            worker_thread.join();
            return 1;
        }
    }

    // Prefill: run every prompt token through the head. Decoding token at
    // position i produces a hidden state that predicts token i+1, so forward
    // every produced hidden state (sequences 0..size-1) so the worker can
    // predict tokens 1..size.
    for (int i = 0; i < static_cast<int>(prompt_tokens.size()); ++i) {
        if (prima::stage_decode_token(head, prompt_tokens[i], static_cast<uint32_t>(i)) != 0) {
            std::fprintf(stderr, "head: prefill decode failed at %d\n", i);
            if (distributed) worker_thread.join();
            prima::stage_free(head);
            return 1;
        }
        if (distributed) {
            if (!forward_hidden(channel, head, static_cast<uint64_t>(i), error, head.n_embd)) {
                std::fprintf(stderr, "head: prefill forward failed: %s\n", error.c_str());
                if (distributed) worker_thread.join();
                prima::stage_free(head);
                return 1;
            }
        }
    }

    // Decode loop: from the final prefill hidden state, generate up to n_predict
    // tokens greedily, forwarding each to the worker (or reading local logits).
    // The next token to produce is at position == prompt_tokens.size().
    std::vector<llama_token> generated;
    generated.reserve(opts.n_predict);

    llama_token prev = prompt_tokens.back();
    uint32_t pos = static_cast<uint32_t>(prompt_tokens.size());
    for (uint32_t step = 0; step < opts.n_predict; ++step) {
        if (prima::stage_decode_token(head, prev, pos) != 0) {
            std::fprintf(stderr, "head: decode failed at %u\n", pos);
            if (distributed) worker_thread.join();
            prima::stage_free(head);
            return 1;
        }

        llama_token next;
        if (distributed) {
            if (!forward_hidden(channel, head, static_cast<uint64_t>(pos), error, head.n_embd)) {
                std::fprintf(stderr, "head: decode forward failed: %s\n", error.c_str());
                if (distributed) worker_thread.join();
                prima::stage_free(head);
                return 1;
            }
            prima::message resp;
            if (!channel.receive(resp, error) || resp.type != prima::message_type::token ||
                resp.payload.size() != sizeof(uint32_t)) {
                std::fprintf(stderr, "head: decode worker response failed: %s\n", error.c_str());
                if (distributed) worker_thread.join();
                prima::stage_free(head);
                return 1;
            }
            std::memcpy(&next, resp.payload.data(), sizeof(next));
        } else {
            const float * logits = llama_get_logits_ith(head.ctx, 0);
            if (logits == nullptr) {
                std::fprintf(stderr, "head: no logits at decode %u\n", pos);
                if (distributed) worker_thread.join();
                prima::stage_free(head);
                return 1;
            }
            next = static_cast<llama_token>(prima::argmax_token(logits, head.n_vocab));
        }

        generated.push_back(next);
        if (next == llama_vocab_eos(vocab)) {
            break;
        }
        prev = next;
        ++pos;
    }

    // Render the generated text.
    std::string text;
    {
        std::vector<char> buf(1024);
        int32_t n = llama_detokenize(vocab, generated.data(), static_cast<int32_t>(generated.size()),
                                      buf.data(), static_cast<int32_t>(buf.size()),
                                      /*remove_special=*/true, /*unparse_special=*/false);
        if (n < 0) {
            buf.resize(static_cast<size_t>(-n) + 1);
            n = llama_detokenize(vocab, generated.data(), static_cast<int32_t>(generated.size()),
                                 buf.data(), static_cast<int32_t>(buf.size()),
                                 /*remove_special=*/true, /*unparse_special=*/false);
        }
        if (n > 0) text.assign(buf.data(), static_cast<size_t>(n));
    }

    std::printf("\n=== generated (%zu tokens) ===\n%s\n", generated.size(), text.c_str());

    if (distributed) {
        channel = prima::tcp_channel{}; // close the link; worker exits
        worker_thread.join();
    }
    prima::stage_free(head);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    const options opts = parse_options(argc, argv);
    if (opts.model.empty()) {
        std::fprintf(stderr, "usage: %s <model.gguf> [--host H] [--port P] [--split N] [--n-predict N] [--prompt \"...\"]\n",
                     argc > 0 ? argv[0] : "prima-stage");
        return 2;
    }
    llama_backend_init();
    const int rc = run_head(opts);
    llama_backend_free();
    return rc;
}
