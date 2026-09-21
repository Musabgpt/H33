#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

#include "llama.h"

#define TAG "H33Native"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

namespace {

struct Engine {
    llama_model * model = nullptr;
    FILE * model_file = nullptr;
    int n_ctx = 3072;
    int n_threads = 4;
    std::atomic<bool> cancel{false};
    std::mutex mutex;
};

std::once_flag backend_once;

void init_backend() {
    std::call_once(backend_once, [] {
        llama_backend_init();
        LOGI("llama.cpp backend initialized");
    });
}

bool abort_callback(void * data) {
    auto * engine = static_cast<Engine *>(data);
    return engine != nullptr && engine->cancel.load(std::memory_order_relaxed);
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::string out(256, '\0');
    int32_t n = llama_token_to_piece(vocab, token, out.data(),
                                     static_cast<int32_t>(out.size()), 0, true);
    if (n < 0) {
        out.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(vocab, token, out.data(),
                                 static_cast<int32_t>(out.size()), 0, true);
    }
    if (n <= 0) return {};
    out.resize(static_cast<size_t>(n));
    return out;
}

jlong create_engine_from_file(FILE * file, jint context_size, jint threads) {
    if (file == nullptr) return 0;
    init_backend();

    auto * engine = new Engine();
    engine->n_ctx = std::clamp(static_cast<int>(context_size), 1024, 4096);
    engine->n_threads = std::clamp(static_cast<int>(threads), 2, 6);

    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;

    engine->model_file = file;
    engine->model = llama_model_load_from_file_ptr(file, params);

    if (engine->model == nullptr) {
        LOGE("Failed to load GGUF model from file descriptor: errno=%d (%s)",
             errno, std::strerror(errno));
        std::fclose(file);
        delete engine;
        return 0;
    }

    LOGI("GGUF model loaded: %.3f GiB, %.3fB params",
         static_cast<double>(llama_model_size(engine->model)) / (1024.0 * 1024.0 * 1024.0),
         static_cast<double>(llama_model_n_params(engine->model)) / 1e9);

    return reinterpret_cast<jlong>(engine);
}

} // namespace

extern "C"
JNIEXPORT jlong JNICALL
Java_com_musab_aragpt2_NativeLlamaEngine_nativeCreate(
        JNIEnv * env, jobject, jstring jpath, jint context_size, jint threads) {
    if (jpath == nullptr) return 0;
    init_backend();

    const char * path = env->GetStringUTFChars(jpath, nullptr);
    if (path == nullptr) return 0;

    auto * engine = new Engine();
    engine->n_ctx = std::clamp(static_cast<int>(context_size), 1024, 4096);
    engine->n_threads = std::clamp(static_cast<int>(threads), 2, 6);

    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;

    engine->model = llama_model_load_from_file(path, params);
    env->ReleaseStringUTFChars(jpath, path);

    if (engine->model == nullptr) {
        LOGE("Failed to load GGUF model from path");
        delete engine;
        return 0;
    }

    LOGI("GGUF model loaded: %.3f GiB, %.3fB params",
         static_cast<double>(llama_model_size(engine->model)) / (1024.0 * 1024.0 * 1024.0),
         static_cast<double>(llama_model_n_params(engine->model)) / 1e9);

    return reinterpret_cast<jlong>(engine);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_musab_aragpt2_NativeLlamaEngine_nativeCreateFromFd(
        JNIEnv *, jobject, jint fd, jint context_size, jint threads) {
    if (fd < 0) return 0;

    int native_fd = ::dup(fd);
    if (native_fd < 0) {
        LOGE("dup(%d) failed: errno=%d (%s)", fd, errno, std::strerror(errno));
        return 0;
    }

    FILE * file = ::fdopen(native_fd, "rb");
    if (file == nullptr) {
        LOGE("fdopen(%d) failed: errno=%d (%s)", native_fd, errno, std::strerror(errno));
        ::close(native_fd);
        return 0;
    }

    return create_engine_from_file(file, context_size, threads);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_musab_aragpt2_NativeLlamaEngine_nativeGenerate(
        JNIEnv * env, jobject, jlong handle, jstring jprompt, jint max_tokens) {
    auto * engine = reinterpret_cast<Engine *>(handle);
    if (engine == nullptr || engine->model == nullptr || jprompt == nullptr) return nullptr;

    std::lock_guard<std::mutex> lock(engine->mutex);
    engine->cancel.store(false, std::memory_order_relaxed);

    const char * prompt_chars = env->GetStringUTFChars(jprompt, nullptr);
    if (prompt_chars == nullptr) return nullptr;
    std::string prompt(prompt_chars);
    env->ReleaseStringUTFChars(jprompt, prompt_chars);

    const llama_vocab * vocab = llama_model_get_vocab(engine->model);

    int32_t n_prompt = -llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            nullptr,
            0,
            true,
            true);

    if (n_prompt <= 0 || n_prompt >= engine->n_ctx - 16) {
        LOGE("Prompt too large or failed to tokenize: %d tokens", n_prompt);
        return nullptr;
    }

    std::vector<llama_token> prompt_tokens(static_cast<size_t>(n_prompt));
    if (llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            prompt_tokens.data(),
            n_prompt,
            true,
            true) < 0) {
        LOGE("Prompt tokenization failed");
        return nullptr;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(engine->n_ctx);
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads = engine->n_threads;
    ctx_params.n_threads_batch = engine->n_threads;
    ctx_params.no_perf = true;

    llama_context * ctx = llama_init_from_model(engine->model, ctx_params);
    if (ctx == nullptr) {
        LOGE("Failed to create llama context");
        return nullptr;
    }

    llama_set_abort_callback(ctx, abort_callback, engine);

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    if (sampler == nullptr) {
        llama_free(ctx);
        return nullptr;
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        LOGE("Prompt decode failed");
        llama_sampler_free(sampler);
        llama_free(ctx);
        return nullptr;
    }

    const int limit = std::clamp(static_cast<int>(max_tokens), 16, 384);
    std::string result;
    result.reserve(static_cast<size_t>(limit) * 4);

    for (int i = 0; i < limit && !engine->cancel.load(std::memory_order_relaxed); ++i) {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        if (token == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, token)) break;

        result += token_piece(vocab, token);

        batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch) != 0) {
            LOGE("Generation decode failed at token %d", i);
            break;
        }
    }

    llama_sampler_free(sampler);
    llama_free(ctx);

    return env->NewStringUTF(result.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_musab_aragpt2_NativeLlamaEngine_nativeCancel(
        JNIEnv *, jobject, jlong handle) {
    auto * engine = reinterpret_cast<Engine *>(handle);
    if (engine != nullptr) engine->cancel.store(true, std::memory_order_relaxed);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_musab_aragpt2_NativeLlamaEngine_nativeDestroy(
        JNIEnv *, jobject, jlong handle) {
    auto * engine = reinterpret_cast<Engine *>(handle);
    if (engine == nullptr) return;

    std::lock_guard<std::mutex> lock(engine->mutex);
    engine->cancel.store(true, std::memory_order_relaxed);
    if (engine->model != nullptr) {
        llama_model_free(engine->model);
        engine->model = nullptr;
    }
    if (engine->model_file != nullptr) {
        std::fclose(engine->model_file);
        engine->model_file = nullptr;
    }
    delete engine;
}
