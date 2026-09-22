#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"

#define TAG "H33Native"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)

namespace {

struct Engine {
    llama_model * model = nullptr;
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

bool contains_stop_marker(const std::string & piece, size_t * marker_pos) {
    static const char * markers[] = {
            "<|im_end|>", "<|im_start|>", "<|EOT|>", "<|eot_id|>",
            "<|end_of_turn|>", "<|endoftext|>", "<｜end▁of▁sentence｜>"
    };
    size_t best = std::string::npos;
    for (const char * marker : markers) {
        const size_t p = piece.find(marker);
        if (p != std::string::npos && (best == std::string::npos || p < best)) {
            best = p;
        }
    }
    if (marker_pos != nullptr) *marker_pos = best;
    return best != std::string::npos;
}

std::string format_chat(const llama_model * model,
                        const std::vector<std::string> & roles,
                        const std::vector<std::string> & contents) {
    const char * metadata_template = llama_model_chat_template(model, nullptr);

    // This app targets DeepSeek-Coder. Prefer the GGUF's tokenizer.chat_template,
    // but fail closed to the known DeepSeek-Coder template if an old/incorrect
    // GGUF was produced without usable template metadata.
    const char * tmpl = metadata_template;
    if (tmpl == nullptr || std::strlen(tmpl) == 0) {
        LOGW("GGUF has no tokenizer.chat_template; using built-in DeepSeek template");
        tmpl = "deepseek";
    } else {
        LOGI("GGUF chat template metadata: %.500s", tmpl);
        const std::string t(tmpl);
        if (t.find("### Instruction:") == std::string::npos ||
            t.find("<|EOT|>") == std::string::npos) {
            LOGW("GGUF template does not look like DeepSeek-Coder; using built-in DeepSeek template");
            tmpl = "deepseek";
        }
    }

    std::vector<llama_chat_message> messages;
    messages.reserve(roles.size());
    for (size_t i = 0; i < roles.size(); ++i) {
        messages.push_back({roles[i].c_str(), contents[i].c_str()});
    }

    const int32_t required = llama_chat_apply_template(
            tmpl, messages.data(), messages.size(), true, nullptr, 0);
    if (required < 0) {
        LOGE("Failed to apply chat template");
        return {};
    }

    std::string prompt(static_cast<size_t>(required), '\0');
    const int32_t written = llama_chat_apply_template(
            tmpl, messages.data(), messages.size(), true,
            prompt.data(), static_cast<int32_t>(prompt.size() + 1));
    if (written < 0) {
        LOGE("Failed to render chat template");
        return {};
    }
    prompt.resize(static_cast<size_t>(written));
    LOGI("Resolved chat prompt bytes=%d", written);
    return prompt;
}

std::string run_generation(Engine * engine, const std::string & prompt, int max_tokens) {
    const llama_vocab * vocab = llama_model_get_vocab(engine->model);

    const int32_t n_prompt = -llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            nullptr,
            0,
            true,
            true);

    if (n_prompt <= 0 || n_prompt >= engine->n_ctx - 16) {
        LOGE("Prompt too large or failed to tokenize: %d tokens", n_prompt);
        return {};
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
        return {};
    }
    LOGI("Prompt token count=%d", n_prompt);

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
        return {};
    }

    llama_set_abort_callback(ctx, abort_callback, engine);

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    if (sampler == nullptr) {
        llama_free(ctx);
        return {};
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        LOGE("Prompt decode failed");
        llama_sampler_free(sampler);
        llama_free(ctx);
        return {};
    }

    const int limit = std::clamp(max_tokens, 16, 384);
    const llama_token eos = llama_vocab_eos(vocab);
    const llama_token eot = llama_vocab_eot(vocab);
    std::string result;
    result.reserve(static_cast<size_t>(limit) * 4);
    int generated = 0;
    const char * stop_reason = "length";

    for (int i = 0; i < limit && !engine->cancel.load(std::memory_order_relaxed); ++i) {
        const llama_token token = llama_sampler_sample(sampler, ctx, -1);
        if (token == LLAMA_TOKEN_NULL) {
            stop_reason = "null-token";
            break;
        }
        if (llama_vocab_is_eog(vocab, token) || token == eos || (eot != LLAMA_TOKEN_NULL && token == eot)) {
            stop_reason = "eos/eot";
            break;
        }

        std::string piece = token_piece(vocab, token);
        size_t marker = std::string::npos;
        if (contains_stop_marker(piece, &marker)) {
            if (marker > 0) result.append(piece, 0, marker);
            stop_reason = "special-marker";
            break;
        }
        result += piece;
        ++generated;

        batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch) != 0) {
            LOGE("Generation decode failed at token %d", i);
            stop_reason = "decode-error";
            break;
        }
    }

    if (engine->cancel.load(std::memory_order_relaxed)) stop_reason = "cancelled";
    LOGI("Generated token count=%d stop_reason=%s", generated, stop_reason);

    llama_sampler_free(sampler);
    llama_free(ctx);
    return result;
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
        LOGE("Failed to load GGUF model");
        delete engine;
        return 0;
    }

    LOGI("GGUF model loaded: %.3f GiB, %.3fB params",
         static_cast<double>(llama_model_size(engine->model)) / (1024.0 * 1024.0 * 1024.0),
         static_cast<double>(llama_model_n_params(engine->model)) / 1e9);

    const char * tmpl = llama_model_chat_template(engine->model, nullptr);
    LOGI("Model chat template present=%s", (tmpl != nullptr && std::strlen(tmpl) > 0) ? "yes" : "no");

    return reinterpret_cast<jlong>(engine);
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
    const std::string prompt(prompt_chars);
    env->ReleaseStringUTFChars(jprompt, prompt_chars);

    const std::string result = run_generation(engine, prompt,
                                               std::clamp(static_cast<int>(max_tokens), 16, 384));
    return env->NewStringUTF(result.c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_musab_aragpt2_NativeLlamaEngine_nativeGenerateChat(
        JNIEnv * env, jobject, jlong handle, jobjectArray jroles, jobjectArray jcontents, jint max_tokens) {
    auto * engine = reinterpret_cast<Engine *>(handle);
    if (engine == nullptr || engine->model == nullptr || jroles == nullptr || jcontents == nullptr) return nullptr;

    const jsize role_count = env->GetArrayLength(jroles);
    const jsize content_count = env->GetArrayLength(jcontents);
    if (role_count <= 0 || role_count != content_count) {
        LOGE("Invalid chat arrays: roles=%d contents=%d", role_count, content_count);
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(engine->mutex);
    engine->cancel.store(false, std::memory_order_relaxed);

    std::vector<std::string> roles;
    std::vector<std::string> contents;
    roles.reserve(static_cast<size_t>(role_count));
    contents.reserve(static_cast<size_t>(content_count));

    for (jsize i = 0; i < role_count; ++i) {
        auto role = static_cast<jstring>(env->GetObjectArrayElement(jroles, i));
        auto content = static_cast<jstring>(env->GetObjectArrayElement(jcontents, i));
        if (role == nullptr || content == nullptr) {
            if (role != nullptr) env->DeleteLocalRef(role);
            if (content != nullptr) env->DeleteLocalRef(content);
            LOGE("Null chat message at index %d", i);
            return nullptr;
        }

        const char * role_chars = env->GetStringUTFChars(role, nullptr);
        const char * content_chars = env->GetStringUTFChars(content, nullptr);
        if (role_chars == nullptr || content_chars == nullptr) {
            if (role_chars != nullptr) env->ReleaseStringUTFChars(role, role_chars);
            if (content_chars != nullptr) env->ReleaseStringUTFChars(content, content_chars);
            env->DeleteLocalRef(role);
            env->DeleteLocalRef(content);
            return nullptr;
        }
        roles.emplace_back(role_chars);
        contents.emplace_back(content_chars);
        env->ReleaseStringUTFChars(role, role_chars);
        env->ReleaseStringUTFChars(content, content_chars);
        env->DeleteLocalRef(role);
        env->DeleteLocalRef(content);
    }

    const std::string prompt = format_chat(engine->model, roles, contents);
    if (prompt.empty()) return nullptr;

    const std::string result = run_generation(engine, prompt,
                                               std::clamp(static_cast<int>(max_tokens), 16, 384));
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
    delete engine;
}
