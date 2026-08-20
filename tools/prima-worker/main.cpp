// prima-worker: a generic stage in a prima-style layer pipeline.
//
// A worker consumes either tokens (stage index 0) or a hidden state (any other
// stage) from its upstream peer, runs its own layer window [start, end), and
// forwards the resulting hidden state to the next stage -- or, when it is the
// tail, reports the predicted token back to the coordinator. Each stage keeps
// its own persistent context so recurrent/attention state survives across
// positions, and it decodes every position (prefill and generation) so its
// local recurrent state stays in sync with the rest of the model.
//
// The whole chain is described by one node_config handed to stage 0 by the
// coordinator and forwarded unchanged down the chain; each worker reads its
// own sliver from `bounds[index]`.
//
// Usage: prima-worker <model.gguf> <host> <port> [-ngl N]

#include "llama.h"
#include "prima-distributed-transport.h"
#include "prima_runtime.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <thread>

namespace {

void fail(const char * what) {
    std::fprintf(stderr, "worker: %s\n", what);
    std::_Exit(1); // skip ggml-metal teardown, which asserts on failure paths
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        fail("usage: prima-worker <model.gguf> <host> <port> [-ngl N]");
    }
    const std::string model_path = argv[1];
    const std::string host = argv[2];
    const uint16_t port = static_cast<uint16_t>(std::stoi(argv[3]));
    int32_t n_gpu_layers = 0;
    bool bench_mode = false;
    bool prefetch_flag = false;
    for (int i = 4; i < argc; ++i) {
        if (std::string(argv[i]) == "-ngl" || std::string(argv[i]) == "--n-gpu-layers") {
            if (i + 1 >= argc) {
                fail("missing value for -ngl");
            }
            n_gpu_layers = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--bench") {
            bench_mode = true;
        } else if (std::string(argv[i]) == "--prefetch") {
            prefetch_flag = true;
        } else {
            fail(("unknown argument: " + std::string(argv[i])).c_str());
        }
    }

    llama_backend_init();

    // 1. Listen; the coordinator (stage 0) or the previous stage connects here.
    prima::tcp_listener listener = prima::tcp_listener::bind_host(host, port);
    if (!listener.valid()) {
        fail("cannot bind listener");
    }
    std::printf("WORKER %s:%u listening\n", host.c_str(), listener.port());
    std::fflush(stdout);

    std::string error;
    // 2. Receive the chain schedule. A coordinator may open (and reset) a
    //    probing connection before sending the real node_config; keep accepting
    //    until a valid one arrives instead of aborting on a spurious first
    //    connection.
    prima::tcp_channel upstream;
    prima::message msg;
    for (;;) {
        upstream = listener.accept(error);
        if (!upstream.valid()) {
            fail("accept failed");
        }
        bool got = upstream.receive(msg, error);
        if (got && msg.type == prima::message_type::node_config) {
            break;
        }
        // Spurious/reset connection (e.g. a coordinator connect_retry probe):
        // drop it and wait for the real one.
    }
    prima::node_config config;
    if (!prima::decode_config(msg.payload.data(), msg.payload.size(), config, error)) {
        fail("cannot decode node_config");
    }
    if (config.index >= config.n_workers || config.bounds.size() != config.n_workers + 1) {
        fail("invalid node_config");
    }

    // Piped-ring mode: the coordinator assigned this worker a set of windows
    // (config.ring), each a half-open [start,end) layer range. Each window is
    // hosted by its own llama context in this process; the worker serves
    // single-window decode requests from the coordinator over the upstream
    // channel (request/response). The ring rotates which window every device
    // computes on each pass, so a device owns several disjoint windows.
    if (!config.ring.empty()) {
        uint32_t n_layer = 0;
        for (const auto & w : config.ring) {
            n_layer = std::max(n_layer, w.second);
        }
        std::vector<prima::stage_model> stages;
        std::string err;
        for (const auto & w : config.ring) {
            prima::stage_model sm;
            const bool ceiling = w.second == n_layer;
            if (!prima::stage_load(sm, model_path, w.first, w.second, /*embeddings=*/false,
                                   config.n_ctx == 0 ? 2048 : config.n_ctx, err,
                                   /*tail=*/ceiling, n_gpu_layers)) {
                std::fprintf(stderr, "worker: ring window [%u,%u) load failed: %s\n",
                             w.first, w.second, err.c_str());
                return 1;
            }
            stages.push_back(std::move(sm));
        }
        std::printf("WORKER ring rank %u/%u loaded %zu windows\n",
                    config.index, config.n_workers, stages.size());
        std::fflush(stdout);

        // The ceiling window (the one ending at n_layer) runs the LM head and
        // samples; its RNG stays lock-step with the reference (one draw per
        // position), so sampled parity holds exactly like the static chain.
        llama_sampler * sampler = nullptr;
        {
            llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
            sampler = llama_sampler_chain_init(sparams);
            if (config.temp > 0.0f) {
                llama_sampler_chain_add(sampler, llama_sampler_init_temp(config.temp));
                if (config.top_p > 0.0f && config.top_p < 1.0f) {
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(config.top_p, 1));
                }
                llama_sampler_chain_add(sampler, llama_sampler_init_dist(config.seed));
            } else {
                llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
            }
        }

        prima::message ready;
        ready.type = prima::message_type::ready;
        ready.sequence = 1;
        if (!upstream.send(ready, error)) {
            fail("cannot send ring ready");
        }

        for (;;) {
            if (!upstream.receive(msg, error)) {
                std::fprintf(stderr, "worker: ring upstream closed\n");
                break;
            }
            if (msg.type == prima::message_type::reset) {
                break;
            }
            const uint32_t local_win = msg.flags & 0xFFFFu;
            const bool want_token = (msg.flags & 0x10000u) != 0;
            if (local_win >= stages.size()) {
                fail("ring: bad window id");
            }
            prima::stage_model & st = stages[local_win];
            const uint32_t p = static_cast<uint32_t>(msg.sequence);
            int rc;
            if (st.start == 0) {
                if (msg.payload.size() != sizeof(uint32_t)) {
                    fail("ring: bad token payload");
                }
                uint32_t tok = 0;
                std::memcpy(&tok, msg.payload.data(), sizeof(tok));
                rc = prima::stage_decode_token(st, static_cast<llama_token>(tok), p);
            } else {
                if (msg.payload.size() != sizeof(float) * st.n_embd) {
                    fail("ring: bad hidden payload");
                }
                rc = prima::stage_decode_hidden(st, reinterpret_cast<const float *>(msg.payload.data()), p);
            }
            if (rc != 0) {
                fail("ring decode failed");
            }
            prima::message out;
            out.type = want_token ? prima::message_type::token : prima::message_type::hidden_state;
            out.sequence = msg.sequence;
            if (want_token) {
                const uint32_t tok = static_cast<uint32_t>(llama_sampler_sample(sampler, st.ctx, -1));
                out.payload.resize(sizeof(tok));
                std::memcpy(out.payload.data(), &tok, sizeof(tok));
            } else {
                const float * hid = llama_get_embeddings_ith(st.ctx, -1);
                if (hid == nullptr) {
                    fail("ring: no embeddings on emitter window");
                }
                out.payload.assign(reinterpret_cast<const uint8_t *>(hid),
                                   reinterpret_cast<const uint8_t *>(hid) + sizeof(float) * st.n_embd);
            }
            if (!upstream.send(out, error)) {
                fail("ring: send failed");
            }
        }
        std::fflush(stdout);
        std::_Exit(0);
    }

    const uint32_t start = config.bounds[config.index];
    const uint32_t end = config.bounds[config.index + 1];
    const bool is_tail = config.index + 1 == config.n_workers;

    // 3. Load this stage's layer window. The coordinator may broadcast a
    //    per-stage GPU offload count (window-relative) computed from a memory
    //    budget; it wins over this process's command-line -ngl.
    int32_t ngl = n_gpu_layers;
    if (!config.ngl.empty()) {
        ngl = config.ngl[config.index];
    }
    prima::stage_model stage;
    if (!prima::stage_load(stage, model_path, start, end, /*embeddings=*/false,
                           config.n_ctx == 0 ? 2048 : config.n_ctx, error,
                           /*tail=*/is_tail, ngl)) {
        std::fprintf(stderr, "worker: stage load failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("WORKER stage %u/%u layers [%u,%u) ngl=%d loaded (%u embeddings)\n",
                config.index, config.n_workers, start, end, ngl, stage.n_embd);
    std::fflush(stdout);

    // Profile mode (--bench): the coordinator only wants this stage's
    // realized decode throughput to feed the layer scheduler. Decode a short
    // synthetic sequence through the window, time it, and report tokens/sec
    // back to the coordinator over the same (upstream) connection, then exit.
    // Stage 0 consumes tokens; deeper stages consume a hidden-state vector.
    if (bench_mode) {
        const uint32_t n_positions = 64; // long enough for a stable timing
        std::vector<float> hidden(stage.n_embd, 0.5f); // dummy hidden input
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t p = 0; p < n_positions; ++p) {
            int rc;
            if (start == 0) {
                const uint32_t tok = (p % 997u) + 1u;
                rc = prima::stage_decode_token(stage, static_cast<llama_token>(tok), p);
            } else {
                rc = prima::stage_decode_hidden(stage, hidden.data(), p);
            }
            if (rc != 0) {
                fail("benchmark decode failed");
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        const float speed = secs > 0.0 ? static_cast<float>(n_positions / secs) : 0.0f;
        std::printf("WORKER bench stage %u speed %.1f tok/s\n", config.index, speed);
        std::fflush(stdout);
        prima::message res;
        res.type = prima::message_type::profile_result;
        res.sequence = 0;
        res.payload.resize(sizeof(speed));
        std::memcpy(res.payload.data(), &speed, sizeof(speed));
        if (!upstream.send(res, error)) {
            fail("cannot send profile result");
        }
        // Skip ggml-metal teardown on exit (its residency-set assert fires at
        // natural teardown), the same approach the head takes.
        std::fflush(stdout);
        std::_Exit(0);
    }

    // The tail owns the LM head, so it is the only stage that samples. The
    // chain is temperature/top-p/seed from the coordinator; config.temp <= 0
    // is greedy. The head's reference builds an identical chain with the same
    // seed so sampled parity still holds.
    llama_sampler * sampler = nullptr;
    if (is_tail) {
        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        sampler = llama_sampler_chain_init(sparams);
        if (config.temp > 0.0f) {
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(config.temp));
            if (config.top_p > 0.0f && config.top_p < 1.0f) {
                llama_sampler_chain_add(sampler, llama_sampler_init_top_p(config.top_p, 1));
            }
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(config.seed));
        } else {
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        }
    }

    // §8 prefetch: warm the model file into the OS page cache in the
    // background. With mmap loading the weight pages are only faulted in on
    // first use (the first chain decode); reading the file here overlaps that
    // disk I/O with the downstream wiring and ready handshake below, so the
    // first decode does not stall on cold-cache faults.
    std::thread prefetcher;
    size_t prefetch_bytes = 0;
    double prefetch_secs = 0.0;
    if (prefetch_flag) {
        prefetcher = std::thread([&] {
            const auto t0 = std::chrono::steady_clock::now();
            FILE * f = std::fopen(model_path.c_str(), "rb");
            if (f != nullptr) {
                std::vector<char> buf(1 << 20); // heap: spawned-thread stacks are small
                size_t n;
                while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) {
                    prefetch_bytes += n;
                }
                std::fclose(f);
            }
            prefetch_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        });
    }

    // 4. Wire the downstream link: stage i connects to stage i+1; the tail
    //    connects back to the coordinator's result listener.
    prima::tcp_channel downstream;
    if (is_tail) {
        downstream = prima::connect_retry(config.head_link.host, config.head_link.port, 600, 100, error);
        if (!downstream.valid()) {
            fail("tail cannot connect back to coordinator");
        }
        std::printf("WORKER tail connecting to head %s:%u\n",
                    config.head_link.host.c_str(), config.head_link.port);
    } else {
        const prima::node_addr & next = config.workers[config.index + 1];
        // The next stage is still starting up (backend init, up to tens of
        // seconds when several workers contend on one machine), so retry
        // until its listener exists.
        downstream = prima::connect_retry(next.host, next.port, 600, 100, error);
        if (!downstream.valid()) {
            fail("cannot connect to downstream stage");
        }
        prima::node_config child = config;
        child.index = config.index + 1;
        child.tail = child.index + 1 == config.n_workers; // child becomes the tail
        // head_link propagates unchanged; only the tail uses it.
        std::vector<uint8_t> payload;
        if (!prima::encode_config(child, payload)) {
            fail("cannot encode child node_config");
        }
        prima::message cfg;
        cfg.type = prima::message_type::node_config;
        cfg.sequence = 0;
        cfg.payload = std::move(payload);
        if (!downstream.send(cfg, error)) {
            fail("cannot send node_config downstream");
        }
        // Wait for the child's ready before reporting ours, so the coordinator
        // only sees "ready" once the whole chain is wired.
        if (!downstream.receive(msg, error) || msg.type != prima::message_type::ready) {
            fail("downstream never became ready");
        }
        std::printf("WORKER stage %u downstream %s:%u ready\n",
                    config.index, next.host.c_str(), next.port);
    }

    // 5. Report readiness upstream (propagates tail -> head -> coordinator).
    prima::message ready;
    ready.type = prima::message_type::ready;
    ready.sequence = 1;
    if (!upstream.send(ready, error)) {
        fail("cannot send ready");
    }

    // The setup phase (downstream wiring + ready propagation) just ran; the
    // prefetch thread overlapped the disk warm with it. Report the stats so
    // the prefetch path is measurable, then continue.
    if (prefetch_flag) {
        prefetcher.join();
        std::printf("WORKER prefetch: warmed %zu bytes in %.2fs\n", prefetch_bytes, prefetch_secs);
        std::fflush(stdout);
    }

    // 6. Decode loop. Every stage decodes every position so recurrent state
    //    stays in sync; only the tail's token matters.
    uint32_t pos = 0;
    for (;;) {
        if (!upstream.receive(msg, error)) {
            std::fprintf(stderr, "worker: upstream closed after %u positions\n", pos);
            break;
        }
        if (msg.type == prima::message_type::reset) {
            break;
        }
        // §11/§12 batched decode: one round carries many (pos, seq) entries.
        // Stage 0 receives batch_decode (token ids in the payload); every
        // deeper stage receives batch_result (hidden states). The answer is a
        // batch_result in both cases: hidden states between stages, one
        // argmax token per entry from the tail. The tail does not run the RNG
        // sampler in batch mode: verification (spec) and greedy comparisons
        // (batching) both need the deterministic argmax.
        if (msg.type == prima::message_type::batch_decode ||
            msg.type == prima::message_type::batch_result) {
            const bool from_head = msg.type == prima::message_type::batch_decode;
            if (from_head != (start == 0)) {
                fail("batch message arrived at the wrong stage");
            }
            std::vector<int32_t> bpos, bseq, btokens;
            std::vector<float> bhidden;
            int32_t clear = 0, trim_to = -1;
            const size_t n_embd_hint = from_head ? 0 : stage.n_embd;
            if (!prima::decode_batch_payload(msg.payload.data(), msg.payload.size(),
                                             n_embd_hint, clear, trim_to, bpos, bseq, btokens, bhidden, error)) {
                fail("bad batch payload");
            }
            // §11 speculative decoding re-prefills the committed prefix from
            // scratch each round: clear the KV so the round's positions restart
            // at 0 (correct even for recurrent/hybrid targets, whose state
            // cannot survive trimming).
            if (clear != 0) {
                llama_memory_clear(llama_get_memory(stage.ctx), true);
            } else if (trim_to >= 0) {
                (void)llama_memory_seq_rm(llama_get_memory(stage.ctx), 0, trim_to, -1);
            }
            const uint32_t n_entries = static_cast<uint32_t>(bpos.size());
            int rc;
            if (from_head) {
                rc = prima::stage_decode_tokens_batch(stage, btokens.data(), bpos.data(), bseq.data(), n_entries);
            } else {
                rc = prima::stage_decode_hidden_batch(stage, bhidden.data(), bpos.data(), bseq.data(), n_entries);
            }
            if (rc != 0) {
                fail("batch decode failed");
            }

            prima::message out;
            out.type = prima::message_type::batch_result;
            out.rank = config.index;
            out.sequence = msg.sequence;
            if (is_tail) {
                // One argmax token per entry (the target's greedy prediction
                // for the position after that entry).
                std::vector<int32_t> results(n_entries);
                for (uint32_t i = 0; i < n_entries; ++i) {
                    const float * logits = llama_get_logits_ith(stage.ctx, static_cast<int32_t>(i));
                    if (logits == nullptr) {
                        fail("tail batch produced no logits");
                    }
                    results[i] = prima::argmax_token(logits, stage.n_vocab);
                }
                out.dtype = prima::data_type::i32;
                if (!prima::encode_batch_payload(bpos, bseq, results, nullptr, 0, clear, trim_to, out.payload)) {
                    fail("cannot encode batch token reply");
                }
            } else {
                std::vector<float> hidden(n_entries * stage.n_embd);
                for (uint32_t i = 0; i < n_entries; ++i) {
                    const float * h = llama_get_embeddings_ith(stage.ctx, static_cast<int32_t>(i));
                    if (h == nullptr) {
                        fail("batch stage produced no embeddings");
                    }
                    std::memcpy(hidden.data() + static_cast<size_t>(i) * stage.n_embd, h,
                                sizeof(float) * stage.n_embd);
                }
                out.dtype = prima::data_type::f32;
                if (!prima::encode_batch_payload(bpos, bseq, std::vector<int32_t>{},
                                                 hidden.data(), stage.n_embd, clear, trim_to, out.payload)) {
                    fail("cannot encode batch hidden reply");
                }
            }
            if (!downstream.send(out, error)) {
                fail("cannot send batch result");
            }
            continue;
        }

        if (start == 0 && msg.type != prima::message_type::token) {
            fail("stage 0 expected token, got other message");
        }
        if (start > 0 && msg.type != prima::message_type::hidden_state) {
            fail("middle stage expected hidden_state, got other message");
        }

        int rc;
        if (start == 0) {
            if (msg.payload.size() != sizeof(uint32_t)) {
                fail("bad token payload");
            }
            uint32_t token = 0;
            std::memcpy(&token, msg.payload.data(), sizeof(token));
            rc = prima::stage_decode_token(stage, static_cast<llama_token>(token), pos);
        } else {
            if (msg.payload.size() != sizeof(float) * stage.n_embd) {
                fail("bad hidden-state payload");
            }
            rc = prima::stage_decode_hidden(stage, reinterpret_cast<const float *>(msg.payload.data()), pos);
        }
        if (rc != 0) {
            fail("decode failed");
        }

        prima::message out;
        out.sequence = pos;

        // Verify the position counter advanced in lock-step with the sender.
        if (msg.sequence != 0 && msg.sequence != pos) {
            std::fprintf(stderr, "worker: sequence mismatch got %llu expected %u\n",
                         static_cast<unsigned long long>(msg.sequence), pos);
            fail("sequence mismatch");
        }

        if (is_tail) {
            const uint32_t token = static_cast<uint32_t>(llama_sampler_sample(sampler, stage.ctx, -1));
            out.type = prima::message_type::token;
            out.rank = config.index;
            out.dtype = prima::data_type::i32;
            out.shape = {1};
            out.payload.resize(sizeof(token));
            std::memcpy(out.payload.data(), &token, sizeof(token));
            if (!downstream.send(out, error)) {
                fail("tail cannot send token");
            }
        } else {
            const float * hidden = llama_get_embeddings_ith(stage.ctx, 0);
            if (hidden == nullptr) {
                fail("no embeddings on non-tail stage");
            }
            out.type = prima::message_type::hidden_state;
            out.rank = config.index;
            out.dtype = prima::data_type::f32;
            out.shape = {1, stage.n_embd};
            out.payload.resize(sizeof(float) * stage.n_embd);
            std::memcpy(out.payload.data(), hidden, out.payload.size());
            if (!downstream.send(out, error)) {
                fail("cannot forward hidden state");
            }
        }
        ++pos;
    }

    prima::stage_free(stage);
    if (sampler != nullptr) {
        llama_sampler_free(sampler);
    }
    upstream = prima::tcp_channel{};
    downstream = prima::tcp_channel{};
    llama_backend_free();
    return 0;
}