#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
#include <algorithm>
#include "llama.h"

namespace {
std::mutex g_mutex;
llama_model * g_model = nullptr;

void free_model_locked() {
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n <= 0) return {};
    return std::string(buf, static_cast<size_t>(n));
}
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_musab_aragpt2_NativeCodeModel_nativeLoad(
        JNIEnv * env, jobject, jstring path, jint) {
    const char * raw = env->GetStringUTFChars(path, nullptr);
    std::string modelPath = raw ? raw : "";
    env->ReleaseStringUTFChars(path, raw);

    std::lock_guard<std::mutex> lock(g_mutex);
    free_model_locked();
    llama_backend_init();

    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(modelPath.c_str(), params);
    return g_model ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_musab_aragpt2_NativeCodeModel_nativeGenerate(
        JNIEnv * env, jobject, jstring prompt, jint maxTokens, jint contextSize) {
    const char * raw = env->GetStringUTFChars(prompt, nullptr);
    std::string text = raw ? raw : "";
    env->ReleaseStringUTFChars(prompt, raw);

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || text.empty()) return env->NewStringUTF("");

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    int n_prompt = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) return env->NewStringUTF("");

    std::vector<llama_token> promptTokens(static_cast<size_t>(n_prompt));
    if (llama_tokenize(vocab, text.c_str(), text.size(), promptTokens.data(), n_prompt, true, true) < 0) {
        return env->NewStringUTF("");
    }

    const int n_predict = std::max(1, static_cast<int>(maxTokens));
    const int n_ctx = std::max(n_prompt + n_predict + 8, std::min(static_cast<int>(contextSize), 8192));

    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = static_cast<uint32_t>(n_ctx);
    ctxParams.n_batch = static_cast<uint32_t>(std::min(n_prompt, 2048));
    ctxParams.no_perf = true;

    llama_context * ctx = llama_init_from_model(g_model, ctxParams);
    if (!ctx) return env->NewStringUTF("");

    llama_sampler_chain_params samplerParams = llama_sampler_chain_default_params();
    samplerParams.no_perf = true;
    llama_sampler * sampler = llama_sampler_chain_init(samplerParams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(promptTokens.data(), promptTokens.size());
    std::string output;

    if (llama_decode(ctx, batch) != 0) {
        llama_sampler_free(sampler);
        llama_free(ctx);
        return env->NewStringUTF("");
    }

    for (int i = 0; i < n_predict; ++i) {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) break;
        output += token_piece(vocab, token);
        batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch) != 0) break;
    }

    llama_sampler_free(sampler);
    llama_free(ctx);
    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_musab_aragpt2_NativeCodeModel_nativeUnload(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    free_model_locked();
    llama_backend_free();
}
