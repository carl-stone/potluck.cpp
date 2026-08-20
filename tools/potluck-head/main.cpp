// potluck-head: coordinator for a potluck-style layer pipeline.
//
// Reads a chain of worker nodes from a config file (host:port one per line),
// automatically tessellates the full model's layers into one contiguous window
// per worker, wires the neighbor-to-neighbor chain, drives prompt prefill plus
// n_predict greedy decode steps, and asserts the pipeline output matches a
// local full-model reference run. The full-model reference is a correctness
// harness (the 0.8B fixture fits on one machine); it is not used in the
// pipeline's data path.
//
// Usage: potluck-head <model.gguf> <workers_file> [n_predict] [host] [-ngl N]

#include "llama-ext.h"
#include "potluck-transport.h"
#include "potluck_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/resource.h>


#if defined(POTLUCK_HIGHS)
#include "Highs.h"
#endif

namespace {

void fail(const char * what) {
    std::fprintf(stderr, "head: %s\n", what);
    std::_Exit(1); // skip ggml-metal teardown, which asserts on failure paths
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


// Builds the identical sampler chain the tail worker builds from the same
// config. temp <= 0 is greedy; otherwise temp -> (optional top-p) -> seed.
// The reference and the chain must consume the same number of RNG draws per
// position for sampled parity, so both call llama_sampler_sample once per
// decoded position, whether or not the token is kept.
llama_sampler * make_sampler(float temp, float top_p, uint32_t seed) {
    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temp > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
        if (top_p > 0.0f && top_p < 1.0f) {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    }
    return sampler;
}

// Full-model greedy reference. Returns generated token ids (excluding prompt).
std::vector<llama_token> run_full(const std::string & model_path,
                                  const std::vector<llama_token> & prompt,
                                  uint32_t n_predict, uint32_t & n_embd,
                                  uint32_t & n_vocab, uint32_t & n_layer,
                                  uint32_t n_ctx, uint32_t n_seq_max, uint32_t n_ubatch,
                                  const std::vector<uint32_t> * gpu_layers,
                                  bool gpu_head,
                                  float temp, float top_p, uint32_t seed,
                                  bool sequential_prefill = false, bool single_thread = false) {
    // The reference runs the full model through the same stage construction as
    // the workers so both share identical numerics (two independently-built
    // llama_contexts of the same model can differ at float-epsilon on Metal,
    // which flips greedy near-ties and makes an exact cross-construction
    // comparison meaningless). It must also share the chain's n_ctx /
    // n_seq_max / n_ubatch: a different ubatch changes batched numerics.
    std::string serr;
    potluck::stage_model sm;
    if (!potluck::stage_load(sm, model_path, 0, 0, /*embeddings=*/false, n_ctx, n_seq_max, n_ubatch,
                           serr, /*tail=*/true, /*n_gpu_layers=*/0, gpu_layers, gpu_head,
                           single_thread)) {
        fail(("reference stage load failed: " + serr).c_str());
    }
    n_embd = sm.n_embd;
    n_vocab = sm.n_vocab;
    n_layer = sm.n_layer;
    llama_sampler * sampler = make_sampler(temp, top_p, seed);
    if (!prompt.empty()) {
        if (sequential_prefill) {
            for (size_t i = 0; i < prompt.size(); ++i) {
                if (potluck::stage_decode_token(sm, prompt[i], static_cast<uint32_t>(i)) != 0) {
                    fail("full sequential prefill decode failed");
                }
            }
        } else {
            std::vector<int32_t> pos(prompt.size());
            std::vector<int32_t> seq(prompt.size(), 0);
            std::vector<int32_t> tok(prompt.size());
            for (size_t i = 0; i < prompt.size(); ++i) {
                pos[i] = static_cast<int32_t>(i);
                tok[i] = static_cast<int32_t>(prompt[i]);
            }
            if (potluck::stage_decode_tokens_batch(sm, tok.data(), pos.data(), seq.data(),
                                                   static_cast<uint32_t>(prompt.size()), /*n_logits=*/static_cast<uint32_t>(prompt.size())) != 0) {
                fail("full batched prefill decode failed");
            }
        }
    }

    const llama_vocab * vocab = llama_model_get_vocab(sm.model);
    const llama_token eos = llama_vocab_eos(vocab);

    std::vector<llama_token> generated;
    generated.reserve(n_predict);
    llama_token prev = prompt.back();
    uint32_t pos = static_cast<uint32_t>(prompt.size());
    for (uint32_t step = 0; step < n_predict; ++step) {
        if (potluck::stage_decode_token(sm, prev, pos) != 0) {
            fail("full generation decode failed");
        }
        const llama_token next = static_cast<llama_token>(llama_sampler_sample(sampler, sm.ctx, -1));
        generated.push_back(next);
        if (next == eos) {
            break;
        }
        prev = next;
        ++pos;
    }

    return generated;
}

struct worker_entry {
    potluck::node_addr addr;
    uint32_t weight = 1;     // integer capability score (default 1)
    bool has_ram = false;    // optional per-machine RAM budget present
    bool has_vram = false;   // optional per-machine VRAM budget present
    float ram_mb = 0.0f;     // MiB budget for this machine's layer weights + KV
    float vram_mb = 0.0f;    // MiB budget for this machine's GPU-offloaded weights (0 = CPU-only)
};

// Each line is "host:port [weight] [ram_mb] [vram_mb]". Weight is an integer
// capability score (default 1): the layer tessellation is proportional to it.
// ram_mb / vram_mb are the §5 LP constraint budgets: per-machine RAM (layer
// weights + KV cache) and GPU memory (offloaded layer weights; 0 forces CPU).
// Both are optional and positional; an absent budget is unlimited. Lines
// starting with '#' and empty lines are ignored.
std::vector<worker_entry> read_workers(const std::string & path) {
    std::vector<worker_entry> workers;
    std::ifstream in(path);
    if (!in) {
        fail("cannot open workers file");
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const size_t colon = line.find(':');
        if (colon == std::string::npos) {
            fail("workers file must be 'host:port [weight] [ram_mb] [vram_mb]' per line");
        }
        worker_entry entry;
        entry.addr.host = line.substr(0, colon);
        std::string rest = line.substr(colon + 1);
        const size_t space = rest.find_first_of(" \t");
        const std::string port_str = rest.substr(0, space);
        entry.addr.port = static_cast<uint16_t>(std::stoi(port_str));
        if (space != std::string::npos) {
            const std::string tail = rest.substr(space + 1);
            std::vector<std::string> fields;
            size_t pos = 0;
            while (pos <= tail.size()) {
                const size_t sp = tail.find_first_of(" \t", pos);
                fields.push_back(tail.substr(pos, sp == std::string::npos ? std::string::npos : sp - pos));
                if (sp == std::string::npos) {
                    break;
                }
                pos = sp + 1;
            }
            entry.weight = static_cast<uint32_t>(std::stoul(fields[0]));
            if (fields.size() >= 2) {
                entry.has_ram = true;
                entry.ram_mb = std::stof(fields[1]);
            }
            if (fields.size() >= 3) {
                entry.has_vram = true;
                entry.vram_mb = std::stof(fields[2]);
            }
        }
        if (entry.weight == 0) {
            fail("worker weight must be >= 1");
        }
        if (entry.has_ram && entry.ram_mb < 0.0f) {
            fail("worker ram_mb must be >= 0");
        }
        if (entry.has_vram && entry.vram_mb < 0.0f) {
            fail("worker vram_mb must be >= 0");
        }
        workers.push_back(std::move(entry));
    }
    if (workers.empty()) {
        fail("workers file is empty");
    }
    return workers;
}

// §6 device profiling: probe every worker (each must run in `--bench` mode),
// measure its realized decode throughput (tokens/sec), and report the speed
// vector. The probe uses a uniform layer split so every stage times the same
// amount of work; the result is a measured per-device capability, independent
// of the eventual tessellation. Returns speeds[i] for workers[i].
std::vector<float> profile_workers(const std::vector<worker_entry> & workers,
                                   uint32_t n_layer, uint32_t n_ctx, uint32_t n_seq_max, uint32_t n_ubatch) {
    const uint32_t n = static_cast<uint32_t>(workers.size());
    std::vector<uint32_t> ub(n + 1, 0);
    for (uint32_t w = 0; w < n; ++w) {
        ub[w + 1] = static_cast<uint32_t>((static_cast<uint64_t>(n_layer) * (w + 1)) / n);
    }
    ub[n] = n_layer;

    std::vector<potluck::node_addr> addrs;
    addrs.reserve(workers.size());
    for (const auto & w : workers) {
        addrs.push_back(w.addr);
    }

    std::vector<float> speeds(n, 0.0f);
    std::string error;
    for (uint32_t i = 0; i < n; ++i) {
        potluck::tcp_channel ch = potluck::connect_retry(workers[i].addr.host, workers[i].addr.port, 300, 100, error);
        if (!ch.valid()) {
            fail("cannot connect to worker for profiling");
        }
        potluck::node_config cfg;
        cfg.n_workers = n;
        cfg.index = i;
        cfg.n_ctx = n_ctx;
        cfg.n_seq_max = n_seq_max;
        cfg.n_ubatch = n_ubatch;
        cfg.bounds = ub;
        cfg.workers = addrs;
        cfg.tail = false;
        std::vector<uint8_t> payload;
        if (!potluck::encode_config(cfg, payload)) {
            fail("cannot encode profile config");
        }
        potluck::message m;
        m.type = potluck::message_type::node_config;
        m.sequence = 0;
        m.payload = std::move(payload);
        if (!ch.send(m, error)) {
            fail("cannot send profile config");
        }
        potluck::message res;
        if (!ch.receive(res, error)) {
            fail(error.c_str());
        }
        if (res.type != potluck::message_type::profile_result ||
            res.payload.size() != sizeof(float)) {
            fail("worker did not return a profile result");
        }
        std::memcpy(&speeds[i], res.payload.data(), sizeof(float));
        std::printf("head: profile[%u] %s:%u  %.1f tok/s\n",
                    i, workers[i].addr.host.c_str(), workers[i].addr.port, speeds[i]);
    }
    std::fflush(stdout);
    return speeds;
}

std::vector<potluck::worker_bench_metrics> request_worker_metrics(
        potluck::tcp_channel & stage0, std::string & error) {
    potluck::message request;
    request.type = potluck::message_type::profile_result;
    if (!stage0.send(request, error)) {
        fail("cannot request worker benchmark metrics");
    }
    potluck::message response;
    if (!stage0.receive(response, error) ||
        response.type != potluck::message_type::profile_result) {
        fail(error.empty() ? "worker benchmark metrics response missing" : error.c_str());
    }
    std::vector<potluck::worker_bench_metrics> metrics;
    if (!potluck::decode_worker_bench_metrics(response.payload.data(), response.payload.size(),
                                               metrics, error)) {
        fail(error.c_str());
    }
    return metrics;
}

#if defined(POTLUCK_HIGHS)
// §5 HiGHS LP allocation. Decide w[m] (layers) and n[m] (GPU-offloaded
// layers) per worker so the slowest stage's wall time per token is minimized,
// subject to the workers file's per-machine budgets:
//   w[m] in [1, n_layer], integers;  n[m] in [0, w[m]], integers;  T >= 0
//   sum(w) = n_layer
//   T >= alpha[m] * w[m]                       (alpha = 1/weight, load balance)
//   n[m] - w[m] <= 0
//   L * n[m] <= vram_mb[m] * 1e6               (only when a VRAM budget is set)
//   (L+kv)*w[m] - L*n[m] <= ram_mb[m] * 1e6    (only when a RAM budget is set;
//                                                KV always lives in RAM)
// T measures the slowest stage only; stages run in parallel on different
// tokens in steady state, so minimizing the max balances the pipeline.
// Offloaded layers are capped at the VRAM budget by the n[m] rows; the solver
// is free to pick any n in [0, min(w, vram/L)], and the caller re-derives the
// actual per-worker ngl deterministically afterward (min(w, vram cap) always
// keeps the RAM row feasible: more offload only frees RAM).
struct lp_plan {
    bool feasible = false;
    std::vector<uint32_t> w;
    std::vector<uint32_t> n;
    double makespan = 0.0;
};

lp_plan lp_allocate(const std::vector<worker_entry> & workers, uint32_t n_layer,
                    double layer_bytes, double kv_bytes) {
    lp_plan plan;
    const int n = static_cast<int>(workers.size());
    const int col_w = 0, col_n = n, col_T = 2 * n;
    const int n_cols = 2 * n + 1;
    const int n_rows = 1 + 4 * n;

    plan.w.assign(n, 0);
    plan.n.assign(n, 0);

    HighsModel model;
    model.lp_.num_col_ = n_cols;
    model.lp_.num_row_ = n_rows;
    model.lp_.sense_ = ObjSense::kMinimize;
    model.lp_.col_cost_.assign(n_cols, 0.0);
    model.lp_.col_cost_[col_T] = 1.0;
    model.lp_.col_lower_ = std::vector<double>(n_cols, 0.0);
    std::fill(model.lp_.col_lower_.begin(), model.lp_.col_lower_.begin() + n, 1.0); // at least 1 layer each
    model.lp_.col_lower_[col_T] = 0.0;
    model.lp_.col_upper_ = std::vector<double>(n_cols, n_layer);
    model.lp_.col_upper_[col_T] = 1.0e30;
    model.lp_.integrality_ = std::vector<HighsVarType>(n_cols, HighsVarType::kContinuous);
    std::fill(model.lp_.integrality_.begin(), model.lp_.integrality_.begin() + 2 * n, HighsVarType::kInteger);
    model.lp_.row_lower_ = std::vector<double>(n_rows, -1.0e30);
    model.lp_.row_upper_ = std::vector<double>(n_rows, 1.0e30);
    model.lp_.offset_ = 0.0;

    int row = 0;
    // sum(w) = n_layer
    model.lp_.row_lower_[row] = n_layer;
    model.lp_.row_upper_[row] = n_layer;
    ++row;
    // T >= alpha[m]*w[m]  <=>  alpha[m]*w[m] - T <= 0
    for (int m = 0; m < n; ++m) {
        model.lp_.row_upper_[row] = 0.0;
        ++row;
    }
    // n[m] - w[m] <= 0
    for (int m = 0; m < n; ++m) {
        model.lp_.row_upper_[row] = 0.0;
        ++row;
    }
    // VRAM: L*n[m] <= vram_mb*1e6 (absent budget -> no row bound)
    for (int m = 0; m < n; ++m) {
        if (workers[m].has_vram) {
            model.lp_.row_upper_[row] = workers[m].vram_mb * 1e6;
        }
        ++row;
    }
    // RAM: (L+kv)*w[m] - L*n[m] <= ram_mb*1e6
    for (int m = 0; m < n; ++m) {
        if (workers[m].has_ram) {
            model.lp_.row_upper_[row] = workers[m].ram_mb * 1e6;
        }
        ++row;
    }

    // Sparse column-wise A matrix with the non-zero pattern above.
    struct entry { int r, c; double v; };
    std::vector<entry> nz;
    row = 0;
    for (int m = 0; m < n; ++m) {
        nz.push_back({row, col_w + m, 1.0});
    }
    ++row;
    for (int m = 0; m < n; ++m) {
        nz.push_back({row, col_w + m, 1.0 / static_cast<double>(workers[m].weight)});
        nz.push_back({row, col_T, -1.0});
        ++row;
    }
    for (int m = 0; m < n; ++m) {
        nz.push_back({row, col_n + m, 1.0});
        nz.push_back({row, col_w + m, -1.0});
        ++row;
    }
    for (int m = 0; m < n; ++m) {
        nz.push_back({row, col_n + m, layer_bytes});
        ++row;
    }
    for (int m = 0; m < n; ++m) {
        nz.push_back({row, col_w + m, layer_bytes + kv_bytes});
        nz.push_back({row, col_n + m, -layer_bytes});
        ++row;
    }

    model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    model.lp_.a_matrix_.start_.assign(n_cols + 1, 0);
    for (int c = 0; c < n_cols; ++c) {
        for (const auto & e : nz) {
            if (e.c == c) {
                model.lp_.a_matrix_.index_.push_back(e.r);
                model.lp_.a_matrix_.value_.push_back(e.v);
            }
        }
        model.lp_.a_matrix_.start_[c + 1] = static_cast<int>(model.lp_.a_matrix_.index_.size());
    }

    Highs highs;
    highs.setOptionValue("log_to_console", false);
    if (highs.passModel(model) != HighsStatus::kOk) {
        return plan;
    }
    if (highs.run() != HighsStatus::kOk ||
        highs.getModelStatus() != HighsModelStatus::kOptimal) {
        return plan;
    }
    const HighsSolution sol = highs.getSolution();
    if (!sol.value_valid) {
        return plan;
    }
    for (int m = 0; m < n; ++m) {
        plan.w[m] = static_cast<uint32_t>(std::llround(sol.col_value[col_w + m]));
        plan.n[m] = static_cast<uint32_t>(std::llround(sol.col_value[col_n + m]));
    }
    plan.makespan = sol.col_value[col_T];
    plan.feasible = true;
    return plan;
}
#endif

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fail("usage: potluck-head <model.gguf> <workers_file> [n_predict] [host] "
             "[-p PROMPT] [--temp F] [--top-p F] [--seed N] [--parity-check] "
             "[--ctx N] [--batch N] [--gpu-layers K] [--gpu-mem MB] "
             "[--profile|--bench --out PATH] [--drop-slowest N] "
             "[--ring W0,W1,..] [--lp] [--stream] [--chat MSG]");
    }
    const std::string model_path = argv[1];
    const std::string workers_file = argv[2];
    uint32_t n_predict = 48;
    std::string head_host = "127.0.0.1";
    std::vector<std::string> prompt_texts;
    std::string draft_path; // §11: proposal model for speculative decoding
    uint32_t draft_n = 4;   // §11: draft length per verify round
    uint32_t batch_n = 0;   // §12: concurrent one-turn requests in one chain
    uint32_t gpu_layers_total = 0; // total GPU-offload layers across the model
    float gpu_mem_mb = 0.0f;       // if >0, derive gpu_layers_total from this budget
    float temp = 0.0f;
    float top_p = 0.0f;
    uint32_t seed = 0;
    uint32_t n_ctx = 4096;   // context length, decided once for the whole cluster
    uint32_t n_seq_max = 1;  // context sequences; --batch N raises it to N
    uint32_t n_ubatch = 1; // one-token static chain numerics; batch mode overrides internally
    bool parity_check = false;
    bool profile_mode = false;
    bool bench_mode = false;
    bool lp_mode = false; // §5: solve the tessellation with the HiGHS LP
    bool stream_mode = false; // §13: print each generated token as it is produced
    bool chat_mode = false;   // §13: apply the model's chat template before generating
    std::vector<uint32_t> ring_sizes; // --ring W0,W1,..: per-rank piped-ring window sizes
    std::vector<uint32_t> forced_bounds; // --bounds A,B,...: explicit shard windows
    uint32_t drop_slowest = 0;
    std::string out_path;
    int positional = 0;
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (!arg.empty() && arg[0] != '-') {
            if (positional == 0) {
                n_predict = static_cast<uint32_t>(std::stoul(arg));
            } else if (positional == 1) {
                head_host = arg;
            } else {
                fail("too many positional arguments");
            }
            ++positional;
        } else if (arg == "-p" || arg == "--prompt") {
            if (i + 1 >= argc) {
                fail("missing value for -p/--prompt");
            }
            // Repeated -p accumulates prompts: §12 batches one request per -p.
            prompt_texts.push_back(argv[++i]);
        } else if (arg == "--draft") {
            // §11: enable speculative decoding with this proposal model. The
            // draft proposes greedy tokens; the chain verifies them against
            // the target's argmax. Garbage drafts only cost acceptance rate,
            // never correctness: every round re-prefills the committed prefix
            // and re-emits the target's own argmax at the first mismatch.
            if (i + 1 >= argc) {
                fail("missing value for --draft");
            }
            draft_path = argv[++i];
        } else if (arg == "--draft-n") {
            if (i + 1 >= argc) {
                fail("missing value for --draft-n");
            }
            const int32_t v = std::stoi(argv[++i]);
            draft_n = static_cast<uint32_t>(v < 1 ? 1 : v);
        } else if (arg == "--batch") {
            // §12: serve this many concurrent one-turn requests through one
            // chain. Each request is one -p prompt; all are decoded together
            // in every round, and each must produce the tokens of its
            // isolated run. The context is sized for exactly this many
            // sequences (n_seq_max), not a fixed 64.
            if (i + 1 >= argc) {
                fail("missing value for --batch");
            }
            const int32_t v = std::stoi(argv[++i]);
            batch_n = static_cast<uint32_t>(v < 1 ? 1 : v);
            n_seq_max = batch_n;
        } else if (arg == "--ctx") {
            if (i + 1 >= argc) {
                fail("missing value for --ctx");
            }
            const int32_t v = std::stoi(argv[++i]);
            n_ctx = static_cast<uint32_t>(v < 1 ? 4096 : v);
        } else if (arg == "--temp") {
            if (i + 1 >= argc) {
                fail("missing value for --temp");
            }
            temp = std::stof(argv[++i]);
        } else if (arg == "--top-p") {
            if (i + 1 >= argc) {
                fail("missing value for --top-p");
            }
            top_p = std::stof(argv[++i]);
        } else if (arg == "--seed") {
            if (i + 1 >= argc) {
                fail("missing value for --seed");
            }
            seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--parity-check") {
            // Opt-in: load and run a full-model in-process reference and
            // require the chain to match it token for token. Off by default;
            // the head must never load the whole model unless asked.
            parity_check = true;
        } else if (arg == "--gpu-mem") {
            if (i + 1 >= argc) {
                fail("missing value for --gpu-mem");
            }
            gpu_mem_mb = std::stof(argv[++i]);
            if (gpu_mem_mb < 0.0f) {
                fail("--gpu-mem must be >= 0");
            }
        } else if (arg == "--gpu-layers" || arg == "-ngl" || arg == "--n-gpu-layers") {
            if (i + 1 >= argc) {
                fail("missing value for --gpu-layers");
            }
            const int32_t v = std::stoi(argv[++i]);
            gpu_layers_total = static_cast<uint32_t>(v < 0 ? 0 : v);
        } else if (arg == "--bounds") {
            if (i + 1 >= argc) {
                fail("missing value for --bounds");
            }
            const std::string spec = argv[++i];
            size_t at = 0;
            for (;;) {
                const size_t comma = spec.find(',', at);
                const std::string value = spec.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
                if (value.empty()) {
                    fail("--bounds needs comma-separated integers");
                }
                forced_bounds.push_back(static_cast<uint32_t>(std::stoul(value)));
                if (comma == std::string::npos) {
                    break;
                }
                at = comma + 1;
            }
        } else if (arg == "--ring") {
            if (i + 1 >= argc) {
                fail("missing value for --ring");
            }
            const std::string spec = argv[++i];
            size_t pos = 0;
            while (pos <= spec.size()) {
                const size_t comma = spec.find(',', pos);
                const std::string part = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (part.empty()) {
                    fail("--ring sizes must be positive integers");
                }
                const uint32_t w = static_cast<uint32_t>(std::stoul(part));
                if (w == 0) {
                    fail("--ring window sizes must be >= 1");
                }
                ring_sizes.push_back(w);
                if (comma == std::string::npos) {
                    break;
                }
                pos = comma + 1;
            }
        } else if (arg == "--drop-slowest") {
            if (i + 1 >= argc) {
                fail("missing value for --drop-slowest");
            }
            const int32_t v = std::stoi(argv[++i]);
            drop_slowest = static_cast<uint32_t>(v < 0 ? 0 : v);
        } else if (arg == "--lp") {
            lp_mode = true;
        } else if (arg == "--profile") {
            profile_mode = true;
        } else if (arg == "--bench") {
            bench_mode = true;
        } else if (arg == "--stream") {
            // §13: stream the generated tokens to stdout as they are produced
            // (one detokenized piece per token, flushed) instead of printing
            // the whole text at the end.
            stream_mode = true;
        } else if (arg == "--chat") {
            // §13: apply the model's chat template to the following literal
            // user message and generate the assistant reply.
            if (i + 1 >= argc) {
                fail("missing value for --chat");
            }
            chat_mode = true;
            prompt_texts.push_back(argv[++i]);
            stream_mode = true;
        } else if (arg == "--out") {
            if (i + 1 >= argc) {
                fail("missing value for --out");
            }
            out_path = argv[++i];
        } else {
            fail(("unknown argument: " + arg).c_str());
        }
    }

    llama_backend_init();

    // Tokenize each prompt (BOS-prefixed), matching the reference harness.
    // no_alloc keeps every hparam (n_layer, n_embd, ...) and the vocab +
    // chat template but allocates only dummy weight buffers and skips all
    // tensor data I/O, so the head reads just the GGUF metadata section.
    // (vocab_only skips hparams entirely and breaks n_layer accessors.)
    llama_model_params mparams = llama_model_default_params();
    mparams.no_alloc      = true;
    mparams.load_mode     = LLAMA_LOAD_MODE_NONE;
    mparams.check_tensors = false;
    llama_model * meta = llama_model_load_from_file(model_path.c_str(), mparams);
    if (meta == nullptr) {
        fail("cannot load model for tokenization");
    }
    const llama_vocab * vocab = llama_model_get_vocab(meta);
    if (prompt_texts.empty()) {
        prompt_texts.push_back("The capital of France is");
    }
    // §13: in --chat mode, wrap the user message in the model's chat template
    // (llama_chat_apply_template supports the pre-defined Jinja subset; the
    // 0.8B fixture ships a built-in Qwen-style template) and render it with
    // the assistant marker so the sampler starts producing the reply.
    if (chat_mode) {
        if (prompt_texts.size() != 1) {
            fail("--chat takes exactly one message");
        }
        const char * tmpl = llama_model_chat_template(meta, nullptr);
        if (tmpl == nullptr) {
            fail("model has no chat template (required for --chat)");
        }
        llama_chat_message msg;
        msg.role = "user";
        msg.content = prompt_texts[0].c_str();
        char buf[8192];
        const int32_t len = llama_chat_apply_template(tmpl, &msg, 1, /*add_ass=*/true,
                                                      buf, static_cast<int32_t>(sizeof(buf)));
        if (len < 0) {
            fail("chat template output would not fit the buffer");
        }
        prompt_texts[0].assign(buf, static_cast<size_t>(len));
    }
    std::vector<std::vector<llama_token>> prompts;
    for (const std::string & text : prompt_texts) {
        std::vector<llama_token> toks(static_cast<size_t>(llama_vocab_n_tokens(vocab)));
        const int32_t n_tok = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                             toks.data(), static_cast<int32_t>(toks.size()),
                                             /*add_special=*/false, /*parse_special=*/chat_mode);
        if (n_tok <= 0) {
            fail("tokenization failed");
        }
        toks.resize(static_cast<size_t>(n_tok));
        toks.insert(toks.begin(), llama_vocab_bos(vocab));
        prompts.push_back(std::move(toks));
    }
    const std::vector<llama_token> & prompt = prompts[0];

    // §6 device profiling mode: measure each worker's decode throughput (the
    // workers must run with `--bench`) and write a measured-capability weights
    // file (host:port <int weight>, slowest = 1) that later feeds §4's
    // tessellation. This is a standalone diagnostic; it does not run the chain.
    if (profile_mode) {
        if (out_path.empty()) {
            fail("--profile requires --out PATH");
        }
        const uint32_t n_pl = static_cast<uint32_t>(llama_model_n_layer(meta));
        std::vector<worker_entry> pworkers = read_workers(workers_file);
        std::vector<float> speeds = profile_workers(pworkers, n_pl, n_ctx, n_seq_max, n_ubatch);
        float mn = *std::min_element(speeds.begin(), speeds.end());
        if (mn <= 0.0f) {
            mn = 1.0f;
        }
        std::vector<uint32_t> weights(pworkers.size());
        for (size_t i = 0; i < pworkers.size(); ++i) {
            const int64_t w = static_cast<int64_t>(std::llround(speeds[i] / mn));
            weights[i] = static_cast<uint32_t>(w < 1 ? 1 : w);
        }
        {
            std::ofstream of(out_path);
            if (!of) {
                fail("cannot open --out file");
            }
            // §7 weak-device removal: the N slowest profiled devices are
            // excluded from the rebuilt topology (they never appear in the
            // weights file, so the chain never connects to them). The
            // remaining devices re-tessellate the full layer span and must
            // still reproduce the reference exactly.
            std::vector<size_t> order(pworkers.size());
            for (size_t i = 0; i < order.size(); ++i) {
                order[i] = i;
            }
            std::sort(order.begin(), order.end(),
                      [&](size_t a, size_t b) { return speeds[a] < speeds[b]; });
            std::vector<bool> dropped(pworkers.size(), false);
            for (uint32_t d = 0; d < drop_slowest && d < static_cast<uint32_t>(order.size()); ++d) {
                dropped[order[d]] = true;
            }
            uint32_t kept = 0;
            for (size_t i = 0; i < pworkers.size(); ++i) {
                if (dropped[i]) {
                    std::printf("head: dropped device %s:%u (%.1f tok/s)\n",
                                pworkers[i].addr.host.c_str(), pworkers[i].addr.port, speeds[i]);
                    continue;
                }
                // §5: carry the input file's per-machine budgets into the
                // measured weights file so the LP chain run can read them.
                of << pworkers[i].addr.host << ":" << pworkers[i].addr.port
                   << " " << weights[i];
                if (pworkers[i].has_ram) {
                    of << " " << std::fixed << std::setprecision(1) << pworkers[i].ram_mb;
                }
                if (pworkers[i].has_vram) {
                    of << " " << std::fixed << std::setprecision(1) << pworkers[i].vram_mb;
                }
                of << "\n";
                ++kept;
            }
            if (kept == 0) {
                fail("--drop-slowest removed every device");
            }
        }
        std::printf("head: measured weights (pre-drop):");
        for (uint32_t w : weights) {
            std::printf(" %u", w);
        }
        std::printf(" -> wrote %s\n", out_path.c_str());
        std::fflush(stdout);
        std::_Exit(0);
    }

    // Resolve the total GPU-offload layer budget K (§9 GPU memory planning).
    // A --gpu-mem cap in MiB derives K from an estimated bytes-per-layer
    // (total model bytes / n_layer); otherwise --gpu-layers/-ngl gives K
    // directly. The offload count is a model-level boundary: layers [0,K) go
    // to GPU, split across stages by their window, and the monolithic
    // reference offloads the same K so reference and chain share placement.
    // Bytes-per-layer comes from the GGUF file size on disk: the head's meta
    // model is no_alloc, so llama_model_size() reports 0.
    const uint32_t n_layer_meta = static_cast<uint32_t>(llama_model_n_layer(meta));
    uint32_t K = std::min<uint32_t>(gpu_layers_total, n_layer_meta);
    if (gpu_mem_mb > 0.0f) {
        const uint64_t bytes_per_layer = std::max<uint64_t>(1, potluck::model_file_bytes(model_path) / n_layer_meta);
        K = static_cast<uint32_t>((static_cast<uint64_t>(gpu_mem_mb * 1024.0f * 1024.0f)) / bytes_per_layer);
        K = std::min(K, n_layer_meta);
    }
    uint32_t n_embd = 0;
    uint32_t n_vocab = 0;
    uint32_t n_layer = 0;
    std::vector<llama_token> reference;


    if (n_embd == 0) {
        n_embd = static_cast<uint32_t>(llama_model_n_embd(meta));
        n_vocab = static_cast<uint32_t>(llama_vocab_n_tokens(llama_model_get_vocab(meta)));
        n_layer = static_cast<uint32_t>(llama_model_n_layer(meta));
    }

    std::vector<worker_entry> workers = read_workers(workers_file);
    const uint32_t n_workers = static_cast<uint32_t>(workers.size());

    if (lp_mode && !ring_sizes.empty()) {
        fail("--lp and --ring are mutually exclusive (the ring takes explicit window sizes)");
    }

    // §3 piped ring. The model's layers are covered by a set of windows, one
    // per rank per cycle; within a cycle the windows are owned in ring order
    // (rank 1, 2, .., n-1, 0, mirroring potluck's this_layer_is_mine). Each rank
    // therefore owns several disjoint windows, and a single token's forward
    // pass hops between ranks' windows multiple times. The coordinator routes
    // every hop (request/response over one channel per worker); the worker
    // hosts all of its windows and decodes the requested one.
    if (!ring_sizes.empty()) {
        if (ring_sizes.size() != n_workers) {
            fail("--ring needs one window size per worker");
        }
        uint64_t cycle_size = 0;
        for (uint32_t w : ring_sizes) {
            cycle_size += w;
        }
        if (cycle_size == 0 || n_layer % cycle_size != 0) {
            fail("--ring window sizes must divide the model's layer count exactly");
        }
        std::vector<uint32_t> order;
        for (uint32_t r = 1; r < n_workers; ++r) {
            order.push_back(r);
        }
        order.push_back(0);
        struct ring_win { uint32_t owner, start, end; };
        std::vector<ring_win> route;
        uint32_t base = 0;
        const uint32_t n_cycles = n_layer / static_cast<uint32_t>(cycle_size);
        for (uint32_t c = 0; c < n_cycles; ++c) {
            for (uint32_t r : order) {
                route.push_back({r, base, base + ring_sizes[r]});
                base += ring_sizes[r];
            }
        }
        // Per-worker window lists and each route window's local id in its owner.
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> per_rank(n_workers);
        std::vector<uint32_t> route_local(route.size());
        std::vector<uint32_t> seen(n_workers, 0);
        for (size_t wi = 0; wi < route.size(); ++wi) {
            const uint32_t owner = route[wi].owner;
            route_local[wi] = seen[owner]++;
            per_rank[owner].push_back({route[wi].start, route[wi].end});
        }

        std::printf("head: RING %zu windows over %u layers, route:", route.size(), n_layer);
        for (const auto & rw : route) {
            std::printf(" r%u[%u,%u)", rw.owner, rw.start, rw.end);
        }
        std::printf("\n");

        // Connect one channel per worker and hand each its window list.
        std::vector<potluck::tcp_channel> channels;
        channels.reserve(n_workers);
        std::string error;
        for (uint32_t i = 0; i < n_workers; ++i) {
            potluck::tcp_channel ch = potluck::connect_retry(workers[i].addr.host, workers[i].addr.port, 300, 100, error);
            if (!ch.valid()) {
                fail("cannot connect to ring worker");
            }
            potluck::node_config cfg;
            cfg.n_workers = n_workers;
            cfg.n_ctx = n_ctx;
            cfg.n_seq_max = n_seq_max;
            cfg.n_ubatch = n_ubatch;
            cfg.index = i;
            cfg.temp = temp;
            cfg.top_p = top_p;
            cfg.bounds.assign(n_workers + 1, 0);
            cfg.bounds.back() = n_layer;
            for (const auto & w : workers) {
                cfg.workers.push_back(w.addr);
            }
            cfg.ring = per_rank[i];
            std::vector<uint8_t> payload;
            if (!potluck::encode_config(cfg, payload)) {
                fail("cannot encode ring config");
            }
            potluck::message m;
            m.type = potluck::message_type::node_config;
            m.payload = std::move(payload);
            if (!ch.send(m, error)) {
                fail("cannot send ring config");
            }
            potluck::message ready;
            if (!ch.receive(ready, error)) {
                fail(error.c_str());
            }
            if (ready.type != potluck::message_type::ready) {
                fail("ring worker never became ready");
            }
            channels.push_back(std::move(ch));
        }
        std::printf("head: ring ready (%u workers)\n", n_workers);
        std::fflush(stdout);
        if (parity_check && draft_path.empty()) {
            std::vector<uint32_t> ring_gpu_layers;
            ring_gpu_layers.reserve(K);
            for (uint32_t layer = 0; layer < K; ++layer) {
                ring_gpu_layers.push_back(layer);
            }
            reference = run_full(model_path, prompt, n_predict, n_embd, n_vocab, n_layer,
                                n_ctx, n_seq_max, n_ubatch, &ring_gpu_layers,
                                K == n_layer, temp, top_p, seed, true, false);
        }

        auto drive_position = [&](const uint8_t * input, size_t input_size, bool input_is_token, uint32_t pos,
                                  uint8_t * out, size_t out_size) -> bool {
            std::vector<uint8_t> in(input, input + input_size);
            for (size_t wi = 0; wi < route.size(); ++wi) {
                const ring_win & rw = route[wi];
                const bool want_token = (wi + 1 == route.size());
                potluck::message req;
                req.type = (rw.start == 0) ? potluck::message_type::token : potluck::message_type::hidden_state;
                req.flags = route_local[wi] | (want_token ? 0x10000u : 0u);
                req.sequence = pos;
                req.payload = std::move(in);
                if (!channels[rw.owner].send(req, error)) {
                    return false;
                }
                potluck::message res;
                if (!channels[rw.owner].receive(res, error)) {
                    return false;
                }
                const bool res_token = (rw.end == n_layer);
                if (res_token && res.type != potluck::message_type::token) {
                    return false;
                }
                if (!res_token && res.type != potluck::message_type::hidden_state) {
                    return false;
                }
                in = std::move(res.payload);
            }
            if (in.size() != out_size) {
                return false;
            }
            std::memcpy(out, in.data(), in.size());
            return true;
        };


        // Prefill: ring routing is a per-position pipe. Batched prefill does
        // not apply because each position must cross every window in order.
        for (size_t i = 0; i < prompt.size(); ++i) {
            uint32_t tok = static_cast<uint32_t>(prompt[i]);
            uint8_t out[sizeof(uint32_t)];
            if (!drive_position(reinterpret_cast<uint8_t *>(&tok), sizeof(tok), true,
                                static_cast<uint32_t>(i), out, sizeof(out))) {
                fail("ring prefill failed");
            }
        }

        // Generation.
        std::vector<llama_token> chain_tokens;
        chain_tokens.reserve(n_predict);
        llama_token prev = prompt.back();
        uint32_t pos = static_cast<uint32_t>(prompt.size());
        for (uint32_t step = 0; step < n_predict; ++step) {
            uint32_t tok = static_cast<uint32_t>(prev);
            uint8_t out[sizeof(uint32_t)];
            if (!drive_position(reinterpret_cast<uint8_t *>(&tok), sizeof(tok), true, pos, out, sizeof(out))) {
                fail("ring generation failed");
            }
            uint32_t next = 0;
            std::memcpy(&next, out, sizeof(next));
            chain_tokens.push_back(static_cast<llama_token>(next));
            if (static_cast<llama_token>(next) == llama_vocab_eos(vocab)) {
                break;
            }
            prev = static_cast<llama_token>(next);
            ++pos;
        }

        const bool match = (chain_tokens == reference);
        std::printf("head: reference_tokens(%zu):", reference.size());
        for (size_t i = 0; i < reference.size() && i < 12; ++i) {
            std::printf(" %d", reference[i]);
        }
        std::printf("\nhead: ring_tokens(%zu):", chain_tokens.size());
        for (size_t i = 0; i < chain_tokens.size() && i < 12; ++i) {
            std::printf(" %d", chain_tokens[i]);
        }
        std::printf("\n");
        std::fflush(stdout);
        if (match) {
            std::printf("RING PASSED: %zu windows across %u workers, %zu generated tokens match full model\n",
                        route.size(), n_workers, chain_tokens.size());
        } else if (!parity_check) {
            std::printf("RING RUN: chain differs from monolithic reference (expected when sampling); "
                        "%zu generated tokens, all in-vocab\n", chain_tokens.size());
        } else {
            std::fprintf(stderr, "head: RING MISMATCH: ring differs from full model\n");
            std::fflush(nullptr);
            std::_Exit(1);
        }
        std::fflush(stdout);
        std::_Exit(0);
    }

    // The scheduler: tessellate n_layer across workers. The default splits
    // proportionally to each worker's capability weight (integer, default 1;
    // pure integer arithmetic keeps the cut deterministic across machines).
    // With --lp (§5) a HiGHS LP solves the allocation instead: minimize the
    // slowest stage's wall time per token subject to the per-machine RAM/VRAM
    // budgets in the workers file.
#if defined(POTLUCK_HIGHS)
    const double layer_bytes = static_cast<double>(std::max<uint64_t>(1, potluck::model_file_bytes(model_path) / n_layer_meta));
    const double kv_bytes =
        2.0 * static_cast<double>(llama_model_n_head_kv(meta)) *
        (llama_model_n_head(meta) > 0 ? static_cast<double>(llama_model_n_embd(meta)) / llama_model_n_head(meta) : 0.0) *
        static_cast<double>(n_ctx) * 4.0; // f32 KV estimate: 2 arrays x kv-heads x head-dim x n_ctx x 4B
#endif
    std::vector<uint32_t> bounds(n_workers + 1);
    bounds[0] = 0;
    std::vector<int32_t> lp_ngl; // §5 deterministic per-worker offload caps
    if (!forced_bounds.empty()) {
        if (lp_mode || forced_bounds.size() != n_workers + 1 ||
            forced_bounds.front() != 0 || forced_bounds.back() != n_layer) {
            fail("--bounds must contain n_workers+1 strictly increasing values from 0 to n_layer and cannot be combined with --lp");
        }
        for (size_t i = 1; i < forced_bounds.size(); ++i) {
            if (forced_bounds[i] <= forced_bounds[i - 1]) {
                fail("--bounds values must be strictly increasing");
            }
        }
        bounds = forced_bounds;
    } else if (lp_mode) {
#if defined(POTLUCK_HIGHS)
        lp_plan plan = lp_allocate(workers, n_layer, layer_bytes, kv_bytes);
        if (!plan.feasible) {
            fail("LP allocation infeasible: the per-machine RAM/VRAM budgets cannot hold the model; "
                 "raise the budgets in the workers file or remove a machine");
        }
        for (uint32_t w = 0; w < n_workers; ++w) {
            bounds[w + 1] = bounds[w] + plan.w[w];
        }
        // Deterministic offload: as many layers as the machine's VRAM budget
        // holds, capped at its window. More offload only frees RAM, so this
        // never violates the LP's RAM rows; the LP's own n column was only a
        // feasibility aid and is not the operative plan.
        lp_ngl.resize(n_workers, 0);
        for (uint32_t w = 0; w < n_workers; ++w) {
            int64_t cap = bounds[w + 1] - bounds[w];
            if (workers[w].has_vram) {
                const int64_t by_vram = static_cast<int64_t>(workers[w].vram_mb * 1e6 / layer_bytes);
                if (by_vram < cap) {
                    cap = by_vram < 0 ? 0 : by_vram;
                }
            }
            lp_ngl[w] = static_cast<int32_t>(cap);
        }
        std::printf("head: LP plan: w=[");
        for (uint32_t w = 0; w < n_workers; ++w) {
            std::printf("%s%u", w ? " " : "", plan.w[w]);
        }
        std::printf("] ngl=[");
        for (uint32_t w = 0; w < n_workers; ++w) {
            std::printf("%s%d", w ? " " : "", lp_ngl[w]);
        }
        std::printf("] makespan=%.1f\n", plan.makespan);
        std::fflush(stdout);
#else
        fail("--lp requires a HiGHS build (reconfigure with -DPOTLUCK_HIGHS=ON)");
#endif
    } else {
        uint64_t total_weight = 0;
        for (const worker_entry & w : workers) {
            total_weight += w.weight;
        }
        uint64_t cum = 0;
        for (uint32_t w = 0; w < n_workers; ++w) {
            cum += workers[w].weight;
            bounds[w + 1] = static_cast<uint32_t>((static_cast<uint64_t>(n_layer) * cum) / total_weight);
            if (bounds[w + 1] <= bounds[w]) {
                fail("a worker would receive zero layers; reduce n_workers or rebalance weights");
            }
        }
        if (bounds[n_workers] != n_layer) {
            bounds[n_workers] = n_layer; // absorb any rounding remainder in the tail
        }
    }

    // Bind the result listener the tail stage will connect back to.
    potluck::tcp_listener result_listener = potluck::tcp_listener::bind_host(head_host, 0);
    if (!result_listener.valid()) {
        fail("cannot bind result listener");
    }

    potluck::node_config config;
    config.n_workers = n_workers;
    config.index = 0;
    config.n_ctx = n_ctx;
    config.n_seq_max = n_seq_max;
    config.n_ubatch = n_ubatch;
    config.seed = seed;
    config.temp = temp;
    config.top_p = top_p;
    config.bounds = bounds;
    std::vector<potluck::node_addr> worker_addrs;
    worker_addrs.reserve(workers.size());
    for (const worker_entry & w : workers) {
        worker_addrs.push_back(w.addr);
    }
    config.workers = worker_addrs;
    // Per-stage GPU offload. Default: stage i owns global layers
    // [bounds[i], b[i+1]) and offloads min(K - bounds[i], window) of them, so
    // the model-wide GPU/CPU boundary sits at layer K for every stage and the
    // reference. With --lp each machine's own VRAM budget caps its offload
    // instead (a vram_mb of 0 keeps that machine CPU-only); the cap is
    // min(window, floor(vram / layer_bytes)), deterministically re-derived
    // from the LP's feasibility rows.
    config.ngl.resize(n_workers);
    for (uint32_t w = 0; w < n_workers; ++w) {
        const uint32_t win = bounds[w + 1] - bounds[w];
        if (lp_mode) {
            config.ngl[w] = lp_ngl[w];
            continue;
        }
        const int64_t off = static_cast<int64_t>(K) - static_cast<int64_t>(bounds[w]);
        const int64_t clamped = off < 0 ? 0 : (off > static_cast<int64_t>(win) ? static_cast<int64_t>(win) : off);
        config.ngl[w] = static_cast<int32_t>(clamped);
    }
    // Build the exact layer map used by the workers for the in-process
    // reference. A plain n_gpu_layers count cannot represent LP placement
    // when a later window is CPU-only, and it would also put the LM head on
    // the wrong device. Match each block tensor and the tail head explicitly.
    std::vector<uint32_t> reference_gpu_layers;
    bool reference_gpu_head = false;
    for (uint32_t w = 0; w < n_workers; ++w) {
        const uint32_t off = config.ngl[w] > 0 ? static_cast<uint32_t>(config.ngl[w]) : 0;
        for (uint32_t layer = bounds[w]; layer < bounds[w] + off; ++layer) {
            reference_gpu_layers.push_back(layer);
        }
        if (w + 1 == n_workers && off > 0) {
            reference_gpu_head = true;
        }
    }
    if (parity_check && draft_path.empty() && batch_n == 0) {
        reference = run_full(model_path, prompt, n_predict, n_embd, n_vocab, n_layer,
                             n_ctx, n_seq_max, n_ubatch, &reference_gpu_layers,
                             reference_gpu_head, temp, top_p, seed,
                             std::getenv("POTLUCK_SEQ_PREFILL") != nullptr, false);
    }
    config.tail = false;
    config.head_link.host = head_host;
    config.head_link.port = result_listener.port();

    std::printf("head: %u workers, layers split:", n_workers);
    for (uint32_t b : bounds) {
        std::printf(" %u", b);
    }
    std::printf("\n");
    std::printf(lp_mode ? "head: GPU offload plan (per-machine VRAM caps), per-worker ngl:"
                        : "head: GPU offload plan (global boundary K=%u), per-worker ngl:", K);
    for (int32_t v : config.ngl) {
        std::printf(" %d", v);
    }
    std::printf("\n");
    std::fflush(stdout);

    // Connect to stage 0 and hand it the whole chain schedule.
    std::string error;
    potluck::tcp_channel stage0 = potluck::connect_retry(workers[0].addr.host, workers[0].addr.port, 600, 100, error);
    if (!stage0.valid()) {
        fail("cannot connect to worker 0");
    }
    stage0.set_timeouts(potluck::handshake_timeout_s(), potluck::handshake_timeout_s());
    std::vector<uint8_t> config_payload;
    if (!potluck::encode_config(config, config_payload)) {
        fail("cannot encode global config");
    }
    potluck::message cfg;
    cfg.type = potluck::message_type::node_config;
    cfg.payload = std::move(config_payload);
    if (!stage0.send(cfg, error)) {
        fail("cannot send config to worker 0");
    }

    // Wait for stage 0 to report chain-wide readiness before accepting the
    // result back-link. Every worker propagates readiness only after its
    // downstream peer is wired, so this receive has the handshake timeout and
    // cannot park forever when a downstream worker rejects its shard.
    potluck::message ready;
    if (!stage0.receive(ready, error)) {
        fail(error.c_str());
    }
    if (ready.type != potluck::message_type::ready) {
        fail("chain never became ready");
    }
    potluck::tcp_channel result = result_listener.accept(error);
    if (!result.valid()) {
        fail("tail never connected back");
    }
    std::printf("head: chain ready\n");
    std::fflush(stdout);

    // Handshake done: switch both chain channels to decode windows.
    stage0.set_timeouts(potluck::decode_timeout_s(), potluck::decode_timeout_s());
    result.set_timeouts(potluck::decode_timeout_s(), potluck::decode_timeout_s());

    // Drive the chain: tokens (stage 0) in, predicted token (tail) out.
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    uint64_t coordinator_payload_bytes = 0;
    auto drive = [&](llama_token token, uint32_t pos) -> llama_token {
        potluck::message in;
        in.type = potluck::message_type::token;
        in.sequence = pos;
        in.dtype = potluck::data_type::i32;
        in.shape = {1};
        uint32_t raw = static_cast<uint32_t>(token);
        in.payload.resize(sizeof(raw));
        std::memcpy(in.payload.data(), &raw, sizeof(raw));
        if (!stage0.send(in, error)) {
            fail("cannot send token to stage 0");
        }
        potluck::message out;
        if (!result.receive(out, error)) {
            fail(error.c_str());
        }
        if (out.type != potluck::message_type::token || out.payload.size() != sizeof(uint32_t)) {
            fail("unexpected result message");
        }
        coordinator_payload_bytes += in.payload.size() + out.payload.size();
        uint32_t next = 0;
        std::memcpy(&next, out.payload.data(), sizeof(next));
        return static_cast<llama_token>(next);
    };


    // §11/§12 batched drive: one round decodes many (pos, seq, token) entries
    // in a single pass per stage; the tail returns the target argmax for every
    // entry. Both dynamic batching and speculative verification rely on it.
    auto drive_batch = [&](const std::vector<int32_t> & pos,
                           const std::vector<int32_t> & seq,
                           const std::vector<int32_t> & tok,
                           int32_t clear, int32_t trim_to, uint32_t n_logits) -> std::vector<int32_t> {
        potluck::message in;
        in.type = potluck::message_type::batch_decode;
        in.sequence = pos.empty() ? 0 : static_cast<uint64_t>(pos.back());
        if (!potluck::encode_batch_payload(pos, seq, tok, nullptr, 0, clear, trim_to, n_logits, in.payload)) {
            fail("cannot encode batch request");
        }
        if (!stage0.send(in, error)) {
            fail("cannot send batch to stage 0");
        }
        potluck::message out;
        if (!result.receive(out, error)) {
            fail(error.c_str());
        }
        coordinator_payload_bytes += in.payload.size() + out.payload.size();
        if (out.type != potluck::message_type::batch_result) {
            fail("unexpected batch result");
        }
        std::vector<int32_t> rpos, rseq, rtok;
        std::vector<float> rhidden;
        int32_t clear_ignored = 0, trim_ignored = -1;
        uint32_t n_logits_ignored = 0;
        if (!potluck::decode_batch_payload(out.payload.data(), out.payload.size(), 0,
                                         clear_ignored, trim_ignored, n_logits_ignored,
                                         rpos, rseq, rtok, rhidden, error)) {
            fail("cannot decode batch result");
        }
        if (rpos.size() != pos.size() || rtok.size() != pos.size()) {
            fail("batch result entry count mismatch");
        }
        return rtok;
    };

    // §5 batched prefill (single-request path): the whole prompt goes down
    // the chain as one batch_decode message — one llama_decode per stage.
    const auto prefill_start = std::chrono::steady_clock::now();
    if (batch_n == 0) {
        if (!prompt.empty()) {
            if (std::getenv("POTLUCK_SEQ_PREFILL") != nullptr) {
                for (size_t i = 0; i < prompt.size(); ++i) {
                    (void) drive(prompt[i], static_cast<uint32_t>(i));
                }
            } else {
                std::vector<int32_t> pos(prompt.size());
                std::vector<int32_t> seq(prompt.size(), 0);
                std::vector<int32_t> tok(prompt.size());
                for (size_t i = 0; i < prompt.size(); ++i) {
                    pos[i] = static_cast<int32_t>(i);
                    tok[i] = static_cast<int32_t>(prompt[i]);
                }
                (void) drive_batch(pos, seq, tok, /*clear=*/1, /*trim_to=*/-1, /*n_logits=*/1);
            }
        }
    }
    prefill_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prefill_start).count();

    const llama_token eos_ = llama_vocab_eos(vocab);

    // §12 dynamic batching: concurrent one-turn requests served by one chain.
    // Each request is a distinct sequence; every round feeds one token per
    // active request at its own position in a single llama_decode, so all
    // requests advance together. The DoD: every request's tokens must exactly
    // match its isolated greedy run.
    if (batch_n > 0) {
        if (prompts.size() < batch_n) {
            fail("--batch needs at least N -p prompts");
        }
        if (temp != 0.0f) {
            fail("--batch is argmax-only; remove --temp");
        }
        const uint32_t n_req = batch_n;
        struct conv {
            std::vector<llama_token> prompt;
            std::vector<llama_token> ref;
            std::vector<llama_token> stream;
        };
        std::vector<conv> convs(n_req);
        for (uint32_t c = 0; c < n_req; ++c) {
            convs[c].prompt = prompts[c];
            uint32_t fe = 0, fv = 0, fl = 0;
            convs[c].ref = run_full(model_path, prompts[c], n_predict, fe, fv, fl,
                                    n_ctx, n_seq_max, n_ubatch, &reference_gpu_layers,
                                    reference_gpu_head, 0.0f, 1.0f, seed);
        }
        // Preload the prompts one position at a time, one entry per request
        // that has a token at that position. Each request's state then
        // advances exactly like the isolated sequential runs it is compared
        // against (the same construction CHAIN PASSED relies on); a single
        // one-shot multi-token preload flips borderline argmaxes on this
        // recurrent/hybrid target and makes the comparison flaky.
        std::vector<int32_t> bpos, bseq, btok;
        size_t max_prompt = 0;
        for (uint32_t c = 0; c < n_req; ++c) {
            max_prompt = std::max(max_prompt, convs[c].prompt.size());
        }
        for (size_t p = 0; p < max_prompt; ++p) {
            bpos.clear(); bseq.clear(); btok.clear();
            for (uint32_t c = 0; c < n_req; ++c) {
                if (p < convs[c].prompt.size()) {
                    bpos.push_back(static_cast<int32_t>(p));
                    bseq.push_back(static_cast<int32_t>(c));
                    btok.push_back(static_cast<int32_t>(convs[c].prompt[p]));
                }
            }
            (void)drive_batch(bpos, bseq, btok, 0, -1, static_cast<uint32_t>(bpos.size()));
        }

        std::vector<llama_token> prev(n_req);
        std::vector<uint32_t> cur_pos(n_req);
        std::vector<bool> done(n_req, false);
        for (uint32_t c = 0; c < n_req; ++c) {
            prev[c] = convs[c].prompt.back();
            cur_pos[c] = static_cast<uint32_t>(convs[c].prompt.size());
        }
        uint32_t remaining = n_req;
        for (uint32_t step = 0; step < n_predict && remaining > 0; ++step) {
            bpos.clear(); bseq.clear(); btok.clear();
            std::vector<uint32_t> idx(n_req);
            for (uint32_t c = 0; c < n_req; ++c) {
                if (done[c]) {
                    continue;
                }
                idx[c] = static_cast<uint32_t>(bpos.size());
                bpos.push_back(static_cast<int32_t>(cur_pos[c]));
                bseq.push_back(static_cast<int32_t>(c));
                btok.push_back(static_cast<int32_t>(prev[c]));
            }
            const std::vector<int32_t> results = drive_batch(bpos, bseq, btok, 0, -1,
                                                             static_cast<uint32_t>(bpos.size()));
            for (uint32_t c = 0; c < n_req; ++c) {
                if (done[c]) {
                    continue;
                }
                const llama_token next = static_cast<llama_token>(results[idx[c]]);
                convs[c].stream.push_back(next);
                if (next == eos_) {
                    done[c] = true;
                    --remaining;
                } else {
                    prev[c] = next;
                    ++cur_pos[c];
                }
            }
        }
        bool all_match = true;
        for (uint32_t c = 0; c < n_req; ++c) {
            std::printf("head: request_%u_tokens(%zu):", c + 1, convs[c].stream.size());
            for (size_t i = 0; i < convs[c].stream.size() && i < 12; ++i) {
                std::printf(" %d", convs[c].stream[i]);
            }
            std::printf("\n");
            const bool ok = (convs[c].stream == convs[c].ref);
            std::printf("head: request_%u %s\n", c + 1, ok ? "MATCHED" : "MISMATCHED");
            all_match = all_match && ok;
        }
        if (!all_match) {
            std::printf("BATCH_MISMATCH\n");
            std::fflush(nullptr);
            std::_Exit(1);
        }
        std::printf("BATCH PASSED: %u concurrent requests, each matching its isolated run\n", n_req);
        std::fflush(nullptr);
        std::_Exit(0);
    }


    // §11 speculative decoding. Mirrors the reference examples/speculative:
    // every round the chain decodes one window [base@C, d1@C+1, ..., dn@C+n]
    // where C is the committed count and base is the last committed token
    // re-stored at a fresh position; A[j] is the target argmax at position
    // C+j (the prediction for slot C+j+1). The longest draft prefix that
    // matches A is accepted, then the target's own argmax A[k] is emitted.
    // Each round tells the workers to trim their KV to C, so rejected drafts
    // never poison the next round's monotone-position check (the workers own
    // their KV, so this mirrors the llama_memory_seq_rm the reference calls on
    // its local target context). The draft context is trimmed the same way.
    if (!draft_path.empty()) {
        if (temp != 0.0f) {
            fail("--draft is arg-only; remove --temp");
        }
        llama_model_params dp = llama_model_default_params();
        llama_model * dm = llama_model_load_from_file(draft_path.c_str(), dp);
        if (dm == nullptr) {
            fail("--draft model load failed");
        }
        llama_context_params dcpx = llama_context_default_params();
        dcpx.n_ctx = n_ctx;
        dcpx.n_batch = 1;
        dcpx.n_ubatch = 1;
        dcpx.n_seq_max = n_seq_max;
        dcpx.n_outputs_max = n_seq_max;
        llama_context * dctx = llama_init_from_model(dm, dcpx);
        if (dctx == nullptr) {
            fail("--draft context init failed");
        }
        llama_memory_t dmem = llama_get_memory(dctx);
        const uint32_t n_vocab_dft = static_cast<uint32_t>(llama_vocab_n_tokens(llama_model_get_vocab(dm)));

        // The chain is the verifying target: after the sequential prompt
        // prefill above, drive() hands back the chain's clean argmax (pick)
        // for the next slot. Each round the head proposes drafts from its own
        // copy of the model and checks them one at a time against pick; only
        // accepted drafts are ever decoded by the chain, so rejected tokens
        // cannot poison the recurrent/hybrid KV, and the committed output is
        // byte-identical to a no-spec greedy run (the §11 DoD). The draft
        // context runs open-loop (never trimmed), so stale proposals are
        // honestly rejected and only cost acceptance rate.
        std::vector<llama_token> committed = prompt;
        std::vector<llama_token> spec_tokens;
        spec_tokens.reserve(n_predict);
        uint32_t approved = 0, proposed = 0;
        llama_batch dtype = llama_batch_init(1, 0, 1);
        int32_t dpos = static_cast<int32_t>(prompt.size()) + 1;
        // The chain harness's first generation step is to re-decode the last
        // prompt token at position P; the pick handed back is that step's
        // token. Mirror it here for the chain (pick) and in the draft context
        // (prefill prompt then re-decode prompt.back() at P), so both sides
        // share the convention and the spec output matches a no-spec run.
        llama_token pick = static_cast<llama_token>(drive(prompt.back(), static_cast<uint32_t>(prompt.size())));
        size_t cursor = static_cast<size_t>(prompt.size()) + 1;
        for (size_t i = 0; i <= prompt.size(); ++i) {
            dtype.n_tokens = 1;
            dtype.token[0] = prompt[i < prompt.size() ? i : prompt.size() - 1];
            dtype.pos[0] = static_cast<int32_t>(i);
            dtype.n_seq_id[0] = 1;
            dtype.seq_id[0][0] = 0;
            dtype.logits[0] = 1;
            if (llama_decode(dctx, dtype) != 0) {
                fail("draft prefill decode failed");
            }
            llama_synchronize(dctx);
        }
        for (uint32_t step = 0; spec_tokens.size() < n_predict; ++step) {
            // Draft: propose draft_n greedy continuations from the draft
            // context's own running state (pure append, never re-prefilled).
            std::vector<llama_token> drafts;
            for (int32_t j = 0; j < static_cast<int32_t>(draft_n); ++j) {
                const float * dlogits = llama_get_logits_ith(dctx, -1);
                if (dlogits == nullptr) {
                    fail("draft produced no logits");
                }
                const uint32_t dv = std::min<uint32_t>(n_vocab_dft, n_vocab);
                const llama_token tok = static_cast<llama_token>(potluck::argmax_token(dlogits, dv));
                drafts.push_back(tok);
                if (tok == eos_) {
                    break;
                }
                dtype.n_tokens = 1;
                dtype.token[0] = tok;
                dtype.pos[0] = dpos++;
                dtype.n_seq_id[0] = 1;
                dtype.seq_id[0][0] = 0;
                dtype.logits[0] = 1;
                if (llama_decode(dctx, dtype) != 0) {
                    fail("draft decode failed");
                }
                llama_synchronize(dctx);
            }
            proposed += static_cast<uint32_t>(drafts.size());

            // Verify: accept drafts while they equal the chain's own argmax.
            // drive() decodes a token at the current cursor and returns the
            // chain's pick for the next slot; only accepted drafts reach the
            // chain, so the committed sequence is exactly the greedy one.
            size_t k = 0;
            while (k < drafts.size() &&
                   static_cast<llama_token>(drafts[k]) == pick) {
                ++approved;
                committed.push_back(drafts[k]);
                spec_tokens.push_back(drafts[k]);
                pick = static_cast<llama_token>(drive(drafts[k], static_cast<uint32_t>(cursor++)));
                ++k;
            }
            committed.push_back(pick);
            spec_tokens.push_back(pick);
            if (pick == eos_) {
                break;
            }
            pick = static_cast<llama_token>(drive(pick, static_cast<uint32_t>(cursor++)));
            if (committed.size() > 2048) {
                fail("spec context exhausted");
            }
        }
        llama_batch_free(dtype);
        std::printf("head: spec_tokens(%zu):", spec_tokens.size());
        for (size_t i = 0; i < spec_tokens.size() && i < 12; ++i) {
            std::printf(" %d", spec_tokens[i]);
        }
        std::printf("\nhead: spec accept %u / %u drafts\n", approved, proposed);
        std::fflush(stdout);
        std::fflush(nullptr);
        std::_Exit(0);
    }
    // Generation: collect the pipeline tokens and compare to the reference.
    // In --stream/--chat mode each token is detokenized and flushed to stdout
    // as it is produced (the streaming output loop); the buffered stream is
    // still kept so the parity comparison below stays intact.
    std::vector<llama_token> chain_tokens;
    chain_tokens.reserve(n_predict);
    if (stream_mode) {
        std::printf("head: stream:");
    }
    llama_token prev = prompt.back();
    uint32_t pos = static_cast<uint32_t>(prompt.size());
    const auto decode_start = std::chrono::steady_clock::now();
    for (uint32_t step = 0; step < n_predict; ++step) {
        const llama_token next = drive(prev, pos);
        chain_tokens.push_back(next);
        if (stream_mode) {
            char piece_buf[256];
            const int32_t n_piece = llama_token_to_piece(vocab, next, piece_buf,
                                                         static_cast<int32_t>(sizeof(piece_buf)), 0, false);
            if (n_piece > 0) {
                std::printf(" %.*s", n_piece, piece_buf);
                std::fflush(stdout);
            }
        }
        if (next == llama_vocab_eos(vocab)) {
            break;
        }
        prev = next;
        ++pos;
    }
    decode_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - decode_start).count();
    if (stream_mode) {
        std::printf("\n");
        std::fflush(stdout);
    }

    if (bench_mode) {
        const double prefill_tps = prefill_seconds > 0.0
            ? prompt.size() / prefill_seconds : 0.0;
        const double decode_tps = decode_seconds > 0.0
            ? chain_tokens.size() / decode_seconds : 0.0;
        const double total_seconds = prefill_seconds + decode_seconds;
        const double aggregate_tps = total_seconds > 0.0
            ? chain_tokens.size() / total_seconds : 0.0;
        const double bytes_per_token = chain_tokens.empty()
            ? 0.0 : static_cast<double>(coordinator_payload_bytes) / chain_tokens.size();
        std::string metrics_error;
        const std::vector<potluck::worker_bench_metrics> metrics =
            request_worker_metrics(stage0, metrics_error);
        if (metrics.size() != workers.size()) {
            fail("worker benchmark metrics count mismatch");
        }
        std::vector<float> worker_speed(workers.size(), 0.0f);
        std::vector<float> worker_rss(workers.size(), 0.0f);
        for (const auto & metric : metrics) {
            if (metric.index >= workers.size()) {
                fail("worker benchmark metrics index out of range");
            }
            worker_speed[metric.index] = metric.decode_tok_s;
            worker_rss[metric.index] = metric.peak_rss_mb;
        }
        const float max_worker_rss = worker_rss.empty()
            ? 0.0f : *std::max_element(worker_rss.begin(), worker_rss.end());
        const uint64_t model_bytes = potluck::model_file_bytes(model_path);
        std::printf("bench worker host window       weight-bytes gpu-layers decode-tok/s peak-rss-mb\n");
        for (size_t i = 0; i < workers.size(); ++i) {
            const uint64_t bytes = model_bytes * (bounds[i + 1] - bounds[i]) /
                std::max<uint32_t>(1, n_layer);
            std::printf("bench %6zu %-15s [%u,%u) %12llu %10d %12.2f %11.1f\n", i,
                        workers[i].addr.host.c_str(), bounds[i], bounds[i + 1],
                        static_cast<unsigned long long>(bytes), config.ngl[i],
                        worker_speed[i], worker_rss[i]);
        }
        std::printf("bench cluster prefill-tok/s %.2f decode-tok/s %.2f aggregate-tok/s %.2f "
                    "ms/token %.2f wire-bytes/token %.1f coordinator-peak-rss-mb %.1f "
                    "worker-peak-rss-mb-max %.1f\n",
                    prefill_tps, decode_tps, aggregate_tps,
                    aggregate_tps > 0.0 ? 1000.0 / aggregate_tps : 0.0,
                    bytes_per_token, peak_rss_mb(), max_worker_rss);
        std::fflush(stdout);
    }

    const bool match = (chain_tokens == reference);
    std::printf("head: reference_tokens(%zu):", reference.size());
    for (size_t i = 0; i < reference.size() && i < 12; ++i) {
        std::printf(" %d", reference[i]);
    }
    std::printf("\nhead: chain_tokens(%zu):", chain_tokens.size());
    for (size_t i = 0; i < chain_tokens.size() && i < 12; ++i) {
        std::printf(" %d", chain_tokens[i]);
    }
    std::printf("\n");


    // Detokenize and render the chain's generated text (matching the old
    // potluck-stage behavior). No special tokens; retry with a bigger buffer if
    // the first estimate was too small.
    {
        std::vector<char> buf(1024);
        int32_t n = llama_detokenize(vocab, chain_tokens.data(), static_cast<int32_t>(chain_tokens.size()),
                                      buf.data(), static_cast<int32_t>(buf.size()),
                                      /*remove_special=*/true, /*unparse_special=*/false);
        if (n < 0) {
            buf.resize(static_cast<size_t>(-n) + 1);
            n = llama_detokenize(vocab, chain_tokens.data(), static_cast<int32_t>(chain_tokens.size()),
                                 buf.data(), static_cast<int32_t>(buf.size()),
                                 /*remove_special=*/true, /*unparse_special=*/false);
        }
        if (n > 0) {
            std::printf("\nhead: generated text (%zu tokens):\n%.*s\n", chain_tokens.size(), n, buf.data());
        }
    }
    // Free everything before tearing down backends so no Metal buffers are
    // alive at exit (upstream asserts on that). _exit skips static teardown,
    // which is where that assert fires when Metal command buffers are still
    // in flight; llama.cpp tools do the same to keep exit codes deterministic.
    // Same reasoning as run_full: leak the tokenizer model and skip
    // llama_backend_free so ggml-metal teardown (and its rsets assert) never
    // runs. The process _Exits immediately below.
    if (match) {
        std::printf("CHAIN PASSED: %u workers, split %u->%u, %zu generated tokens match full model\n",
                    n_workers, bounds[0], bounds[n_workers], chain_tokens.size());
    } else if (!parity_check) {
        std::printf("CHAIN RUN: no --parity-check, so no full-model reference was loaded; "
                    "%zu generated tokens, all in-vocab\n", chain_tokens.size());
    } else {
        std::fprintf(stderr, "head: MISMATCH: chain differs from full model\n");
        std::fflush(nullptr);
        std::_Exit(1);
    }
    std::fflush(nullptr);
    std::_Exit(0);
}