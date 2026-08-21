// test-potluck-run: end-to-end correctness for the finished potluck-stage pipeline.
//
// Strategy (greedy, deterministic):
//   1. Load the full model and run prompt prefill + n_predict autoregressive
//      steps, recording the full generated token sequence as the reference.
//   2. Load the head stage [0, split) and a remote stage [split, n_layer), run
//      the same prefill + generation across the split, recording the pipeline
//      token sequence.
//   3. Assert the two sequences are identical, and print the generated text.
//
// Uses an in-process remote stage for speed.
//
// Usage: test-potluck-run <model.gguf> [split] [n_predict]

#include "llama.h"
#include "potluck_runtime.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

// Runs prefill + greedy generation on a single (possibly full) model/context
// and returns the generated token ids (excluding the prompt tokens).
std::vector<llama_token> run_full(const std::string & model_path,
                                  const std::vector<llama_token> & prompt_tokens,
                                  uint32_t n_predict, uint32_t n_layer, uint32_t & n_embd, uint32_t & n_vocab) {
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    CHECK(model != nullptr);

    n_embd = static_cast<uint32_t>(llama_model_n_embd(model));
    n_vocab = static_cast<uint32_t>(llama_vocab_n_tokens(llama_model_get_vocab(model)));
    n_layer = static_cast<uint32_t>(llama_model_n_layer(model));

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 1;
    cparams.n_ubatch = 1;
    llama_context * ctx = llama_init_from_model(model, cparams);
    CHECK(ctx != nullptr);

    llama_batch batch = llama_batch_init(1, 0, 1);

    for (int i = 0; i < static_cast<int>(prompt_tokens.size()); ++i) {
        batch.n_tokens = 1;
        batch.token[0] = prompt_tokens[i];
        batch.pos[0] = i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        CHECK(llama_decode(ctx, batch) == 0);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token eos = llama_vocab_eos(vocab);

    std::vector<llama_token> generated;
    generated.reserve(n_predict);
    llama_token prev = prompt_tokens.back();
    uint32_t pos = static_cast<uint32_t>(prompt_tokens.size());
    for (uint32_t step = 0; step < n_predict; ++step) {
        batch.n_tokens = 1;
        batch.token[0] = prev;
        batch.pos[0] = static_cast<int32_t>(pos);
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        CHECK(llama_decode(ctx, batch) == 0);
        const float * logits = llama_get_logits_ith(ctx, 0);
        CHECK(logits != nullptr);
        const llama_token next = static_cast<llama_token>(potluck::argmax_token(logits, n_vocab));
        generated.push_back(next);
        if (next == eos) break;
        prev = next;
        ++pos;
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return generated;
}

// Runs prefill + greedy generation split across a head stage [0, split) and a
// remote stage [split, n_layer). The remote stage is driven in-process through
// an explicit hand-off of the head's hidden state (bypassing the socket so this
// test is hermetic and fast; the socket path is exercised by the network test).
std::vector<llama_token> run_split(const std::string & model_path,
                                   const std::vector<llama_token> & prompt_tokens,
                                   uint32_t split, uint32_t n_predict,
                                   uint32_t n_embd, uint32_t n_vocab, uint32_t n_layer) {
    CHECK(split > 0 && split < n_layer);

    potluck::stage_model head;
    std::string error;
    CHECK(potluck::stage_load(head, model_path, 0, split, /*embeddings=*/true, /*n_ctx=*/2048,
                              /*n_seq_max=*/1, /*n_ubatch=*/1, error));
    potluck::stage_model remote;
    CHECK(potluck::stage_load(remote, model_path, split, 0, /*embeddings=*/false, /*n_ctx=*/2048,
                              /*n_seq_max=*/1, /*n_ubatch=*/1, error, /*tail=*/true));

    const llama_vocab * vocab = llama_model_get_vocab(head.model);
    const llama_token eos = llama_vocab_eos(vocab);

    auto run_remote = [&](uint32_t pos) -> llama_token {
        const float * hidden = llama_get_embeddings_ith(head.ctx, 0);
        CHECK(hidden != nullptr);
        CHECK(potluck::stage_decode_hidden(remote, hidden, pos) == 0);
        const float * logits = llama_get_logits_ith(remote.ctx, 0);
        CHECK(logits != nullptr);
        return static_cast<llama_token>(potluck::argmax_token(logits, n_vocab));
    };

    // Prefill: feed tokens to the head. Decoding token at position i produces a
    // hidden state that predicts token i+1, so resolve the remote token for every
    // prefix position (sequences 0..size-1) to keep its recurrent state in sync.
    for (int i = 0; i < static_cast<int>(prompt_tokens.size()); ++i) {
        CHECK(potluck::stage_decode_token(head, prompt_tokens[i], static_cast<uint32_t>(i)) == 0);
        (void)run_remote(static_cast<uint32_t>(i));
    }

    std::vector<llama_token> generated;
    generated.reserve(n_predict);
    llama_token prev = prompt_tokens.back();
    uint32_t pos = static_cast<uint32_t>(prompt_tokens.size());
    for (uint32_t step = 0; step < n_predict; ++step) {
        CHECK(potluck::stage_decode_token(head, prev, pos) == 0);
        const llama_token next = run_remote(pos);
        generated.push_back(next);
        if (next == eos) break;
        prev = next;
        ++pos;
    }

    potluck::stage_free(remote);
    potluck::stage_free(head);
    return generated;
}

std::string detokenize(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::vector<char> buf(4096);
    int32_t n = llama_detokenize(vocab, tokens.data(), static_cast<int32_t>(tokens.size()),
                                 buf.data(), static_cast<int32_t>(buf.size()),
                                 /*remove_special=*/true, /*unparse_special=*/false);
    if (n < 0) {
        buf.resize(static_cast<size_t>(-n) + 1);
        n = llama_detokenize(vocab, tokens.data(), static_cast<int32_t>(tokens.size()),
                             buf.data(), static_cast<int32_t>(buf.size()),
                             /*remove_special=*/true, /*unparse_special=*/false);
    }
    return n > 0 ? std::string(buf.data(), static_cast<size_t>(n)) : std::string();
}

} // namespace

int main(int argc, char ** argv) {
    const std::string model_path = argc > 1 ? argv[1] : "/tmp/llama-upstream/models/Qwen3.5-0.8B-Q4_0.gguf";
    const uint32_t split = argc > 2 ? static_cast<uint32_t>(std::stoi(argv[2])) : 12;
    const uint32_t n_predict = argc > 3 ? static_cast<uint32_t>(std::stoi(argv[3])) : 48;

    const std::string prompt = "The capital of France is";
    llama_backend_init();

    // Load just for tokenization / meta.
    llama_model_params mparams = llama_model_default_params();
    llama_model * meta = llama_model_load_from_file(model_path.c_str(), mparams);
    CHECK(meta != nullptr);
    const llama_vocab * vocab = llama_model_get_vocab(meta);
    const llama_token bos = llama_vocab_bos(vocab);

    std::vector<llama_token> prompt_tokens(static_cast<size_t>(llama_vocab_n_tokens(vocab)));
    const int32_t n_prompt = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                            prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()),
                                            /*add_special=*/false, /*parse_special=*/false);
    CHECK(n_prompt > 0);
    prompt_tokens.resize(static_cast<size_t>(n_prompt));
    prompt_tokens.insert(prompt_tokens.begin(), bos);
    const uint32_t n_layer = static_cast<uint32_t>(llama_model_n_layer(meta));
    // NOTE: keep `meta` alive until after detokenize() below, which uses `vocab`
    // obtained from it. Freeing it here would leave `vocab` dangling.

    uint32_t n_embd = 0, n_vocab = 0, n_layer_out = 0;
    std::vector<llama_token> full = run_full(model_path, prompt_tokens, n_predict, n_layer, n_embd, n_vocab);
    // Note: run_full overwrites n_layer via its arg; capture the real layer count from meta above.
    (void)n_layer_out;

    std::vector<llama_token> split_tokens = run_split(model_path, prompt_tokens, split, n_predict, n_embd, n_vocab, n_layer);

    bool match = (full == split_tokens);
    std::printf("full_tokens(%zu):", full.size());
    for (size_t i = 0; i < full.size() && i < 12; ++i) std::printf(" %d", full[i]);
    std::printf(" ...\n");
    std::printf("split_tokens(%zu):", split_tokens.size());
    for (size_t i = 0; i < split_tokens.size() && i < 12; ++i) std::printf(" %d", split_tokens[i]);
    std::printf(" ...\n");
    std::printf("generated: %s\n", detokenize(vocab, split_tokens).c_str());

    if (!match) {
        std::fprintf(stderr, "MISMATCH: full vs split pipeline token sequences differ\n");
        llama_backend_free();
        return 1;
    }
    std::printf("potluck-run test passed (split=%u, generated=%zu tokens)\n", split, split_tokens.size());
    llama_model_free(meta);
    llama_backend_free();
    return 0;
}
