#include "llama.h"

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

void set_token_batch(llama_batch & batch, llama_token token) {
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
}

void set_hidden_batch(llama_batch & batch, const float * hidden, uint32_t n_embd) {
    batch.n_tokens = 1;
    std::memcpy(batch.embd, hidden, sizeof(float) * n_embd);
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

} // namespace

int main(int argc, char ** argv) {
    const std::string model_path = argc > 1 ? argv[1] : "/tmp/potluck.cpp/models/Qwen3.5-0.8B-Q4_0.gguf";

    llama_backend_init();
    llama_model_params full_params = model_params(0, 0);
    llama_model * full_model = llama_model_load_from_file(model_path.c_str(), full_params);
    CHECK(full_model != nullptr);

    const uint32_t n_embd = llama_model_n_embd(full_model);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(full_model));
    const llama_token input = llama_vocab_bos(llama_model_get_vocab(full_model));
    const uint32_t split = llama_model_n_layer(full_model) / 2;

    llama_model * first_model = llama_model_load_from_file(model_path.c_str(), model_params(0, split));
    llama_model * second_model = llama_model_load_from_file(model_path.c_str(), model_params(split, 0));
    CHECK(first_model != nullptr);
    CHECK(second_model != nullptr);

    llama_context * first = llama_init_from_model(first_model, context_params(0, split, true));
    CHECK(first != nullptr);
    llama_batch first_batch = llama_batch_init(1, 0, 1);
    set_token_batch(first_batch, input);
    CHECK(llama_decode(first, first_batch) == 0);
    const float * hidden = llama_get_embeddings_ith(first, 0);
    CHECK(hidden != nullptr);
    std::vector<float> hidden_copy(hidden, hidden + n_embd);

    llama_context * second = llama_init_from_model(second_model, context_params(split, 0, false));
    CHECK(second != nullptr);
    llama_batch second_batch = llama_batch_init(1, n_embd, 1);
    set_hidden_batch(second_batch, hidden_copy.data(), n_embd);
    CHECK(llama_decode(second, second_batch) == 0);
    const float * staged_logits = llama_get_logits_ith(second, 0);
    CHECK(staged_logits != nullptr);
    const int staged_token = argmax(staged_logits, n_vocab);

    llama_context * full = llama_init_from_model(full_model, context_params(0, 0, false));
    CHECK(full != nullptr);
    llama_batch full_batch = llama_batch_init(1, 0, 1);
    set_token_batch(full_batch, input);
    CHECK(llama_decode(full, full_batch) == 0);
    const float * full_logits = llama_get_logits_ith(full, 0);
    CHECK(full_logits != nullptr);
    const int full_token = argmax(full_logits, n_vocab);

    std::printf("staged_token=%d full_token=%d\n", staged_token, full_token);
    CHECK(staged_token == full_token);

    llama_batch_free(full_batch);
    llama_batch_free(second_batch);
    llama_batch_free(first_batch);
    llama_free(full);
    llama_free(second);
    llama_free(first);
    llama_model_free(second_model);
    llama_model_free(first_model);
    llama_model_free(full_model);
    llama_backend_free();
    return 0;
}
