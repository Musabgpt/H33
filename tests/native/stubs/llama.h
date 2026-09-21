#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
using llama_token = int;
struct llama_model {};
struct llama_vocab {};
struct llama_context { bool (*abort_callback)(void *) = nullptr; void * abort_data = nullptr; };
struct llama_model_params {};
struct llama_context_params {
    int n_ctx, n_batch, n_ubatch, n_threads, n_threads_batch;
    int type_k, type_v; bool no_perf;
};
struct llama_sampler { int next = 2; };
struct sampler_params { bool no_perf; };
// Matches the non-owning pointer contract of llama_batch_get_one.
struct llama_batch { int n_tokens; llama_token * token; };
inline std::atomic<bool> fake_slow_decode{false}, fake_decode_entered{false}, fake_aborted{false};
inline constexpr int GGML_TYPE_Q8_0 = 8;
inline void llama_backend_init() {}
inline void llama_backend_free() {}
inline llama_model_params llama_model_default_params() { return {}; }
inline llama_model * llama_model_load_from_file(const char *, llama_model_params) { return new llama_model; }
inline void llama_model_free(llama_model * m) { delete m; }
inline llama_context_params llama_context_default_params() { return {}; }
inline llama_context * llama_init_from_model(llama_model *, llama_context_params) { return new llama_context; }
inline void llama_free(llama_context * c) { delete c; }
inline void * llama_get_memory(llama_context *) { return nullptr; }
inline void llama_memory_clear(void *, bool) {}
inline const llama_vocab * llama_model_get_vocab(llama_model *) { static llama_vocab v; return &v; }
inline int llama_n_ctx(llama_context *) { return 384; }
inline int llama_tokenize(const llama_vocab *, const char *, int, llama_token * out, int n, bool, bool) {
    if (n < 1) return -1;
    *out = 1; return 1;
}
inline llama_batch llama_batch_get_one(llama_token * tokens, int n) { return {n, tokens}; }
inline sampler_params llama_sampler_chain_default_params() { return {}; }
inline llama_sampler * llama_sampler_chain_init(sampler_params) { return new llama_sampler; }
inline llama_sampler * llama_sampler_init_greedy() { return new llama_sampler; }
inline void llama_sampler_chain_add(llama_sampler *, llama_sampler * child) { delete child; }
inline void llama_sampler_free(llama_sampler * s) { delete s; }
inline llama_token llama_sampler_sample(llama_sampler * s, llama_context *, int) { return s->next++; }
inline bool llama_vocab_is_eog(const llama_vocab *, llama_token token) { return token == 4; }
inline int llama_token_to_piece(const llama_vocab *, llama_token token, char * out, int, int, bool) {
    *out = token == 2 ? 'O' : 'K'; return 1;
}
inline void llama_set_abort_callback(llama_context * c, bool (*fn)(void *), void * data) {
    c->abort_callback = fn; c->abort_data = data;
}
__attribute__((noinline)) inline int llama_decode(llama_context * c, llama_batch batch) {
    // Real llama.cpp reads the caller-owned token storage on each decode.
    volatile int token = *batch.token;
    (void) token;
    fake_decode_entered = true;
    if (fake_slow_decode) {
        const auto limit = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (std::chrono::steady_clock::now() < limit) {
            if (c->abort_callback && c->abort_callback(c->abort_data)) {
                fake_aborted = true; return 2;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    return 0;
}
