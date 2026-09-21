#include <jni.h>
#include <android/log.h>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include "llama.h"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "H33Llama", __VA_ARGS__)

static llama_model * g_model = nullptr;
static llama_context * g_ctx = nullptr;
static std::atomic<bool> g_cancel(false);
static std::mutex g_lock;

static std::string from_jstring(JNIEnv * env, jstring value) {
    if (!value) return {};
    const char * chars = env->GetStringUTFChars(value, nullptr);
    std::string out(chars ? chars : "");
    if (chars) env->ReleaseStringUTFChars(value, chars);
    return out;
}

static jstring js(JNIEnv * env, const std::string & value) {
    jbyteArray bytes = env->NewByteArray(static_cast<jsize>(value.size()));
    if (!bytes) return nullptr;
    env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(value.size()),
            reinterpret_cast<const jbyte *>(value.data()));
    jclass string_class = env->FindClass("java/lang/String");
    jmethodID ctor = env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
    jstring charset = env->NewStringUTF("UTF-8");
    jstring out = static_cast<jstring>(env->NewObject(string_class, ctor, bytes, charset));
    env->DeleteLocalRef(charset);
    env->DeleteLocalRef(string_class);
    env->DeleteLocalRef(bytes);
    return out;
}

static void release_all() {
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_musab_aragpt2_LlamaNative_open(JNIEnv * env, jclass, jstring path,
                                        jint context_size, jint threads) {
    std::lock_guard<std::mutex> guard(g_lock);
    release_all();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    std::string model_path = from_jstring(env, path);
    g_model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!g_model) return js(env, "تعذر فتح نموذج GGUF");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = context_size;
    cp.n_batch = 128;
    cp.n_ubatch = 128;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;
    cp.no_perf = true;
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) { release_all(); return js(env, "ذاكرة الهاتف غير كافية لإنشاء سياق النموذج"); }
    return js(env, "");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_musab_aragpt2_LlamaNative_generate(JNIEnv * env, jclass, jstring jprompt,
                                            jint max_tokens) {
    std::lock_guard<std::mutex> guard(g_lock);
    if (!g_model || !g_ctx) return js(env, "");
    g_cancel.store(false);
    llama_memory_clear(llama_get_memory(g_ctx), true);
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    std::string prompt = from_jstring(env, jprompt);
    int count = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (count <= 0) return js(env, "");
    if (count > 350) {
        prompt = prompt.substr(prompt.size() / 2);
        count = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    }
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), count, true, true) < 0) {
        return js(env, "");
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    auto params = llama_sampler_chain_default_params();
    params.no_perf = true;
    llama_sampler * sampler = llama_sampler_chain_init(params);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    std::string output;
    int position = 0;

    while (!g_cancel.load() && position + batch.n_tokens < 510 && max_tokens-- > 0) {
        if (llama_decode(g_ctx, batch) != 0) break;
        position += batch.n_tokens;
        llama_token token = llama_sampler_sample(sampler, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) break;
        char small[256];
        int n = llama_token_to_piece(vocab, token, small, sizeof(small), 0, true);
        if (n < 0) {
            std::vector<char> large(static_cast<size_t>(-n));
            n = llama_token_to_piece(vocab, token, large.data(), large.size(), 0, true);
            if (n > 0) output.append(large.data(), n);
        } else if (n > 0) output.append(small, n);
        batch = llama_batch_get_one(&token, 1);
        if (output.find("<|EOT|>") != std::string::npos) break;
    }
    llama_sampler_free(sampler);
    return js(env, output);
}

extern "C" JNIEXPORT void JNICALL
Java_com_musab_aragpt2_LlamaNative_cancel(JNIEnv *, jclass) { g_cancel.store(true); }

extern "C" JNIEXPORT void JNICALL
Java_com_musab_aragpt2_LlamaNative_close(JNIEnv *, jclass) {
    g_cancel.store(true);
    std::lock_guard<std::mutex> guard(g_lock);
    release_all();
    llama_backend_free();
}
