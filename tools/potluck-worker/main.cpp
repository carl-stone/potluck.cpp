// potluck-worker: legacy PTLK/raw-TCP stage used by component checks.
//
// It can consume tokens or hidden state, execute assigned windows, and preserve
// recurrent and attention state across positions. Its current chain and ring
// connections do not implement ADR 0007: the finished product must use direct
// ZeroMQ communication between adjacent ring peers, configured automatically
// by the integrated server. Keep useful stage execution behavior during the
// cutover, but do not preserve this transport or coordinator topology.
//
// Usage: potluck-worker <model.gguf> <host> <port> [-ngl N]

#include "llama-ext.h"
#include "potluck-transport.h"
#include "gguf.h"
#include "potluck_runtime.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <thread>
#include <sys/resource.h>


namespace {

void fail(const char * what) {
    std::fprintf(stderr, "worker: %s\n", what);
    std::_Exit(1); // skip ggml-metal teardown, which asserts on failure paths
}

bool validate_shard(const std::string & path, uint32_t start, uint32_t end, std::string & error) {
    ggml_context * meta = nullptr;
    gguf_init_params params = { true, &meta };
    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (ctx == nullptr) {
        error = "cannot read GGUF metadata from " + path;
        return false;
    }
    const int count_id = gguf_find_key(ctx, "potluck.shard.count");
    if (count_id < 0) {
        gguf_free(ctx);
        if (meta != nullptr) {
            ggml_free(meta);
        }
        return true;
    }
    const int shard_start_id = gguf_find_key(ctx, "potluck.shard.start");
    const int shard_end_id = gguf_find_key(ctx, "potluck.shard.end");
    if (shard_start_id < 0 || shard_end_id < 0 ||
        gguf_get_kv_type(ctx, count_id) != GGUF_TYPE_UINT32 ||
        gguf_get_kv_type(ctx, shard_start_id) != GGUF_TYPE_UINT32 ||
        gguf_get_kv_type(ctx, shard_end_id) != GGUF_TYPE_UINT32) {
        error = "invalid potluck shard metadata in " + path;
        gguf_free(ctx);
        if (meta != nullptr) {
            ggml_free(meta);
        }
        return false;
    }
    const uint32_t shard_start = gguf_get_val_u32(ctx, shard_start_id);
    const uint32_t shard_end = gguf_get_val_u32(ctx, shard_end_id);
    const bool inside = start >= shard_start && end <= shard_end && start < end;
    if (!inside) {
        error = "assigned layers [" + std::to_string(start) + "," + std::to_string(end) +
                ") but shard file " + path + " holds [" + std::to_string(shard_start) +
                "," + std::to_string(shard_end) + "); re-run potluck-shard with matching --bounds";
    }
    gguf_free(ctx);
    if (meta != nullptr) {
        ggml_free(meta);
    }
    return inside;
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


} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        fail("usage: potluck-worker <model.gguf> <host> <port> [-ngl N]");
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

    // Legacy listener: accepts a coordinator or previous static-chain stage.
    potluck::tcp_listener listener = potluck::tcp_listener::bind_host(host, port);
    if (!listener.valid()) {
        fail("cannot bind listener");
    }
    std::printf("WORKER %s:%u listening\n", host.c_str(), listener.port());
    std::fflush(stdout);

    std::string error;
    // Legacy PTLK schedule path. It accepts probe connections before the real
    // node_config. ADR 0007 requires this lifecycle to be replaced by the
    // direct-peer ZeroMQ ring.
    potluck::tcp_channel upstream;
    potluck::message msg;
    for (;;) {
        upstream = listener.accept(error);
        if (!upstream.valid()) {
            fail("accept failed");
        }
        bool got = upstream.receive(msg, error);
        if (got && msg.type == potluck::message_type::node_config) {
            break;
        }
        if (!got) {
            std::fprintf(stderr, "worker: %s\n", error.c_str());
            if (error.rfind("protocol version mismatch:", 0) == 0) {
                fail(error.c_str());
            }
        } else {
            std::fprintf(stderr, "worker: ignoring connection message type %u; expected node_config\n",
                         static_cast<unsigned>(msg.type));
        }
    }
    potluck::node_config config;
    if (!potluck::decode_config(msg.payload.data(), msg.payload.size(), config, error)) {
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
        const uint32_t n_layer = config.bounds.back();
        if (n_layer == 0) {
            fail("ring config has no global layer count");
        }
        std::vector<potluck::stage_model> stages(config.ring.size());
        std::string err;
        for (size_t ri = config.ring.size(); ri-- > 0;) {
            const auto & w = config.ring[ri];
            const bool ceiling = w.second == n_layer;
            if (!validate_shard(model_path, w.first, w.second, err)) {
                std::fprintf(stderr, "worker: %s\n", err.c_str());
                return 1;
            }
            if (!potluck::stage_load(stages[ri], model_path, w.first, w.second, /*embeddings=*/false,
                                   config.n_ctx == 0 ? 4096 : config.n_ctx,
                                   config.n_seq_max == 0 ? 1 : config.n_seq_max,
                                   config.n_ubatch == 0 ? 512 : config.n_ubatch,
                                   err, /*tail=*/ceiling, n_gpu_layers,
                                   nullptr, /*explicit_gpu_head=*/false, /*single_thread=*/false)) {
                std::fprintf(stderr, "worker: ring window [%u,%u) load failed: %s\n",
                             w.first, w.second, err.c_str());
                return 1;
            }
        }
        std::printf("WORKER ring rank %u/%u loaded %zu windows\n",
                    config.index, config.n_workers, stages.size());
        std::fflush(stdout);

        // The ceiling window (the one ending at n_layer) runs the LM head and
        // samples every position request, prefill included. With temp=0 that
        // is argmax (zero RNG draws), matching the batched reference, so
        // greedy parity holds exactly; with temp>0 the ring draws once per
        // prefill position where the reference draws zero, so sampled ring
        // parity is not asserted.
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

        potluck::message ready;
        ready.type = potluck::message_type::ready;
        ready.sequence = 1;
        if (!upstream.send(ready, error)) {
            fail("cannot send ring ready");
        }
        // Ring requests are per-position decode traffic: decode windows, not
        // the accept-time handshake window.
        upstream.set_timeouts(potluck::decode_timeout_s(), potluck::decode_timeout_s());

        for (;;) {
            if (!upstream.receive(msg, error)) {
                if (error.rfind("protocol version mismatch:", 0) == 0) {
                    fail(error.c_str());
                }
                std::fprintf(stderr, "worker: ring upstream closed: %s\n", error.c_str());
                break;
            }
            if (msg.type == potluck::message_type::reset) {
                break;
            }
            const uint32_t local_win = msg.flags & 0xFFFFu;
            const bool want_token = (msg.flags & 0x10000u) != 0;
            if (local_win >= stages.size()) {
                fail("ring: bad window id");
            }
            potluck::stage_model & st = stages[local_win];
            if (msg.type == potluck::message_type::batch_decode ||
                msg.type == potluck::message_type::batch_result) {
                const bool from_head = msg.type == potluck::message_type::batch_decode;
                if (from_head != (st.start == 0)) {
                    fail("ring batch message arrived at the wrong window");
                }
                std::vector<int32_t> bpos, bseq, btokens;
                std::vector<float> bhidden;
                int32_t clear = 0;
                int32_t trim_to = -1;
                uint32_t n_logits = 0;
                const size_t n_embd_hint = from_head ? 0 : st.n_embd;
                if (!potluck::decode_batch_payload(msg.payload.data(), msg.payload.size(),
                                                   n_embd_hint, clear, trim_to, n_logits,
                                                   bpos, bseq, btokens, bhidden, error)) {
                    fail("ring: bad batch payload");
                }
                if (bpos.empty() || bseq.size() != bpos.size() ||
                    (from_head ? btokens.size() != bpos.size()
                               : bhidden.size() != bpos.size() * st.n_embd)) {
                    fail("ring: invalid batch entry count");
                }
                if (clear != 0) {
                    llama_memory_clear(llama_get_memory(st.ctx), true);
                } else if (trim_to >= 0) {
                    (void) llama_memory_seq_rm(llama_get_memory(st.ctx), 0, trim_to, -1);
                }
                const uint32_t n_entries = static_cast<uint32_t>(bpos.size());
                const bool is_tail_window = st.end == n_layer;
                if (is_tail_window && n_logits > n_entries) {
                    fail("ring: invalid tail logits count");
                }
                int batch_rc;
                if (from_head) {
                    batch_rc = potluck::stage_decode_tokens_batch(
                        st, btokens.data(), bpos.data(), bseq.data(), n_entries,
                        is_tail_window ? n_logits : n_entries);
                } else {
                    batch_rc = potluck::stage_decode_hidden_batch(
                        st, bhidden.data(), bpos.data(), bseq.data(), n_entries, n_logits);
                }
                if (batch_rc != 0) {
                    fail("ring batch decode failed");
                }
                potluck::message out;
                out.type = potluck::message_type::batch_result;
                out.rank = config.index;
                out.sequence = msg.sequence;
                if (is_tail_window) {
                    std::vector<int32_t> results(n_entries, 0);
                    for (uint32_t i = n_entries - n_logits; i < n_entries; ++i) {
                        const float * logits = llama_get_logits_ith(st.ctx, static_cast<int32_t>(i));
                        if (logits == nullptr) {
                            fail("ring: tail batch produced no logits");
                        }
                        results[i] = potluck::argmax_token(logits, st.n_vocab);
                    }
                    if (!potluck::encode_batch_payload(bpos, bseq, results, nullptr, 0,
                                                       clear, trim_to, n_logits, out.payload)) {
                        fail("ring: cannot encode token batch result");
                    }
                } else {
                    std::vector<float> hidden(n_entries * st.n_embd);
                    for (uint32_t i = 0; i < n_entries; ++i) {
                        const float * h = llama_get_embeddings_ith(st.ctx, static_cast<int32_t>(i));
                        if (h == nullptr) {
                            fail("ring: batch window produced no embeddings");
                        }
                        std::memcpy(hidden.data() + static_cast<size_t>(i) * st.n_embd, h,
                                    sizeof(float) * st.n_embd);
                    }
                    if (!potluck::encode_batch_payload(bpos, bseq, std::vector<int32_t>{},
                                                       hidden.data(), st.n_embd,
                                                       clear, trim_to, n_logits, out.payload)) {
                        fail("ring: cannot encode hidden batch result");
                    }
                }
                if (!upstream.send(out, error)) {
                    fail("ring: batch send failed");
                }
                continue;
            }
            const uint32_t p = static_cast<uint32_t>(msg.sequence);
            int rc;
            if (st.start == 0) {
                if (msg.payload.size() != sizeof(uint32_t)) {
                    fail("ring: bad token payload");
                }
                uint32_t tok = 0;
                std::memcpy(&tok, msg.payload.data(), sizeof(tok));
                rc = potluck::stage_decode_token(st, static_cast<llama_token>(tok), p);
            } else {
                if (msg.payload.size() != sizeof(float) * st.n_embd) {
                    fail("ring: bad hidden payload");
                }
                rc = potluck::stage_decode_hidden(st, reinterpret_cast<const float *>(msg.payload.data()), p);
            }
            if (rc != 0) {
                fail("ring decode failed");
            }
            potluck::message out;
            out.type = want_token ? potluck::message_type::token : potluck::message_type::hidden_state;
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
    potluck::stage_model stage;
    if (!validate_shard(model_path, start, end, error)) {
        std::fprintf(stderr, "worker: %s\n", error.c_str());
        return 1;
    }
    if (!potluck::stage_load(stage, model_path, start, end, /*embeddings=*/false,
                           config.n_ctx == 0 ? 4096 : config.n_ctx,
                           config.n_seq_max == 0 ? 1 : config.n_seq_max,
                           config.n_ubatch == 0 ? 512 : config.n_ubatch,
                           error, /*tail=*/is_tail, ngl,
                           nullptr, /*explicit_gpu_head=*/false, /*single_thread=*/true)) {
        std::fprintf(stderr, "worker: stage load failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("WORKER stage %u/%u layers [%u,%u) ngl=%d loaded (%u embeddings, peak-rss-mb=%.1f)\n",
                config.index, config.n_workers, start, end, ngl, stage.n_embd, peak_rss_mb());

    uint64_t decoded_positions = 0;
    double decode_seconds = 0.0;

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
                rc = potluck::stage_decode_token(stage, static_cast<llama_token>(tok), p);
            } else {
                rc = potluck::stage_decode_hidden(stage, hidden.data(), p);
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
        potluck::message res;
        res.type = potluck::message_type::profile_result;
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
    potluck::tcp_channel downstream;
    if (is_tail) {
        downstream = potluck::connect_retry(config.head_link.host, config.head_link.port, 600, 100, error);
        if (!downstream.valid()) {
            fail("tail cannot connect back to coordinator");
        }
        // The result channel starts in the handshake phase (ready report),
        // then carries decode traffic; timeouts switch below after ready.
        downstream.set_timeouts(potluck::handshake_timeout_s(), potluck::handshake_timeout_s());
    } else {
        const potluck::node_addr & next = config.workers[config.index + 1];
        // The next stage is still starting up (backend init, up to tens of
        // seconds when several workers contend on one machine), so retry
        // until its listener exists.
        downstream = potluck::connect_retry(next.host, next.port, 600, 100, error);
        if (!downstream.valid()) {
            fail("cannot connect to downstream stage");
        }
        downstream.set_timeouts(potluck::handshake_timeout_s(), potluck::handshake_timeout_s());
        potluck::node_config child = config;
        child.index = config.index + 1;
        child.tail = child.index + 1 == config.n_workers; // child becomes the tail
        // head_link propagates unchanged; only the tail uses it.
        std::vector<uint8_t> payload;
        if (!potluck::encode_config(child, payload)) {
            fail("cannot encode child node_config");
        }
        potluck::message cfg;
        cfg.type = potluck::message_type::node_config;
        cfg.sequence = 0;
        cfg.payload = std::move(payload);
        if (!downstream.send(cfg, error)) {
            fail("cannot send node_config downstream");
        }
        // Wait for the child's ready before reporting ours, so the coordinator
        // only sees "ready" once the whole chain is wired.
        if (!downstream.receive(msg, error)) {
            fail(error.c_str());
        }
        if (msg.type != potluck::message_type::ready) {
            fail("downstream never became ready");
        }
        std::printf("WORKER stage %u downstream %s:%u ready\n",
                    config.index, next.host.c_str(), next.port);
    }

    // 5. Report readiness upstream (propagates tail -> head -> coordinator).
    potluck::message ready;
    ready.type = potluck::message_type::ready;
    ready.sequence = 1;
    if (!upstream.send(ready, error)) {
        fail("cannot send ready");
    }

    // Handshake is over: every remaining exchange is per-position decode
    // traffic, which may legitimately wait minutes on weak hardware.
    upstream.set_timeouts(potluck::decode_timeout_s(), potluck::decode_timeout_s());
    if (downstream.valid()) {
        downstream.set_timeouts(potluck::decode_timeout_s(), potluck::decode_timeout_s());
    }
    auto send_bench_metrics = [&]() {
        std::vector<potluck::worker_bench_metrics> metrics;
        if (!is_tail) {
            potluck::message request;
            request.type = potluck::message_type::profile_result;
            potluck::message child;
            if (!downstream.send(request, error) || !downstream.receive(child, error) ||
                child.type != potluck::message_type::profile_result) {
                fail("cannot collect downstream benchmark metrics");
            }
            if (!potluck::decode_worker_bench_metrics(child.payload.data(), child.payload.size(),
                                                       metrics, error)) {
                fail(error.c_str());
            }
        }
        potluck::worker_bench_metrics own;
        own.index = config.index;
        own.start = start;
        own.end = end;
        own.decode_tok_s = decode_seconds > 0.0
            ? static_cast<float>(decoded_positions / decode_seconds) : 0.0f;
        own.peak_rss_mb = static_cast<float>(peak_rss_mb());
        own.decoded_positions = decoded_positions;
        metrics.push_back(own);
        potluck::message response;
        response.type = potluck::message_type::profile_result;
        if (!potluck::encode_worker_bench_metrics(metrics, response.payload) ||
            !upstream.send(response, error)) {
            fail("cannot send benchmark metrics");
        }
    };

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
            if (error.rfind("protocol version mismatch:", 0) == 0) {
                fail(error.c_str());
            }
            std::fprintf(stderr, "worker: upstream closed after %u positions: %s\n",
                         pos, error.c_str());
            break;
        }
        if (msg.type == potluck::message_type::profile_result && msg.payload.empty()) {
            send_bench_metrics();
            continue;
        }
        if (msg.type == potluck::message_type::reset) {
            break;
        }
        // §11/§12 batched decode: one round carries many (pos, seq) entries.
        // Stage 0 receives batch_decode (token ids in the payload); every
        // deeper stage receives batch_result (hidden states). The answer is a
        // batch_result in both cases: hidden states between stages, one
        // argmax token per entry from the tail. The tail does not run the RNG
        // sampler in batch mode: verification (spec) and greedy comparisons
        // (batching) both need the deterministic argmax.
        if (msg.type == potluck::message_type::batch_decode ||
            msg.type == potluck::message_type::batch_result) {
            const bool from_head = msg.type == potluck::message_type::batch_decode;
            if (from_head != (start == 0)) {
                fail("batch message arrived at the wrong stage");
            }
            std::vector<int32_t> bpos, bseq, btokens;
            std::vector<float> bhidden;
            int32_t clear = 0, trim_to = -1;
            uint32_t n_logits = 0;
            const size_t n_embd_hint = from_head ? 0 : stage.n_embd;
            if (!potluck::decode_batch_payload(msg.payload.data(), msg.payload.size(),
                                             n_embd_hint, clear, trim_to, n_logits,
                                             bpos, bseq, btokens, bhidden, error)) {
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
            const auto stage_decode_start = std::chrono::steady_clock::now();
            int rc;
            if (from_head) {
                // A non-tail stage is an embeddings context, which requires
                // every batch entry to be an output; its n_outputs_max covers
                // the whole batch (stage_context_params), and the windowed
                // graph computes no logits rows anyway. The tail uses the
                // coordinator's n_logits (trailing entries only).
                rc = potluck::stage_decode_tokens_batch(stage, btokens.data(), bpos.data(), bseq.data(),
                                                        n_entries, is_tail ? n_logits : n_entries);
            } else {
                // The helper applies the per-stage logits policy: emitter
                // stages output every entry (embeddings context), the tail
                // only the trailing n_logits.
                rc = potluck::stage_decode_hidden_batch(stage, bhidden.data(), bpos.data(), bseq.data(),
                                                        n_entries, n_logits);
            }
            decode_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - stage_decode_start).count();
            decoded_positions += n_entries;
            if (rc != 0) {
                fail("batch decode failed");
            }

            potluck::message out;
            out.type = potluck::message_type::batch_result;
            out.rank = config.index;
            out.sequence = msg.sequence;
            if (is_tail) {
                // One argmax token per trailing logits entry (the target's
                // greedy prediction for the position after that entry). Rows
                // without logits (a prefill's earlier positions) are reported
                // as 0; the coordinator ignores prefill results anyway.
                std::vector<int32_t> results(n_entries, 0);
                for (uint32_t i = n_entries - n_logits; i < n_entries; ++i) {
                    const float * logits = llama_get_logits_ith(stage.ctx, static_cast<int32_t>(i));
                    if (logits == nullptr) {
                        fail("tail batch produced no logits");
                    }
                    results[i] = potluck::argmax_token(logits, stage.n_vocab);
                }
                out.dtype = potluck::data_type::i32;
                if (!potluck::encode_batch_payload(bpos, bseq, results, nullptr, 0, clear, trim_to, n_logits, out.payload)) {
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
                out.dtype = potluck::data_type::f32;
                if (!potluck::encode_batch_payload(bpos, bseq, std::vector<int32_t>{},
                                                 hidden.data(), stage.n_embd, clear, trim_to, n_logits, out.payload)) {
                    fail("cannot encode batch hidden reply");
                }
            }
            if (!downstream.send(out, error)) {
                fail("cannot send batch result");
            }
            // Per-token messages continue where the batch left off; the
            // worker's position counter must advance past the entries decoded
            // above (batch payloads carry positions explicitly and do not use
            // it, so this only affects the transition to per-token decoding).
            pos = static_cast<uint32_t>(bpos.back()) + 1;
            continue;
        }

        if (start == 0 && msg.type != potluck::message_type::token) {
            fail("stage 0 expected token, got other message");
        }
        if (start > 0 && msg.type != potluck::message_type::hidden_state) {
            fail("middle stage expected hidden_state, got other message");
        }

        const auto stage_decode_start = std::chrono::steady_clock::now();
        int rc;
        uint32_t token = 0;
        if (start == 0) {
            if (msg.payload.size() != sizeof(uint32_t)) {
                fail("bad token payload");
            }
            std::memcpy(&token, msg.payload.data(), sizeof(token));
            rc = potluck::stage_decode_token(stage, static_cast<llama_token>(token), pos);
        } else {
            if (msg.payload.size() != sizeof(float) * stage.n_embd) {
                fail("bad hidden-state payload");
            }
            rc = potluck::stage_decode_hidden(stage, reinterpret_cast<const float *>(msg.payload.data()), pos);
        }
        decode_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - stage_decode_start).count();
        ++decoded_positions;
        if (rc != 0) {
            fail("decode failed");
        }
        if (!is_tail) {
            const float * hidden = llama_get_embeddings_ith(stage.ctx, 0);
            if (hidden == nullptr) {
                fail("no embeddings on non-tail stage");
            }
        }

        potluck::message out;
        out.sequence = pos;

        // Verify the position counter advanced in lock-step with the sender.
        if (msg.sequence != 0 && msg.sequence != pos) {
            std::fprintf(stderr, "worker: sequence mismatch got %llu expected %u\n",
                         static_cast<unsigned long long>(msg.sequence), pos);
            fail("sequence mismatch");
        }

        if (is_tail) {
            const uint32_t token = static_cast<uint32_t>(llama_sampler_sample(sampler, stage.ctx, -1));
            out.type = potluck::message_type::token;
            out.rank = config.index;
            out.dtype = potluck::data_type::i32;
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
            out.type = potluck::message_type::hidden_state;
            out.rank = config.index;
            out.dtype = potluck::data_type::f32;
            out.shape = {1, stage.n_embd};
            out.payload.resize(sizeof(float) * stage.n_embd);
            std::memcpy(out.payload.data(), hidden, out.payload.size());
            if (!downstream.send(out, error)) {
                fail("stage cannot send hidden state");
            }
        }
        ++pos;
    }

    potluck::stage_free(stage);
    if (sampler != nullptr) {
        llama_sampler_free(sampler);
    }
    upstream = potluck::tcp_channel{};
    downstream = potluck::tcp_channel{};
    llama_backend_free();
    return 0;
}