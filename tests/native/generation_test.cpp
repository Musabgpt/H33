// Contract tests of the real JNI adapter. These do not measure model speed or
// claim to reproduce Android/ARM inference; that needs an on-device test.
#include "../../app/src/main/cpp/h33_llama.cpp"
#include <cstdio>
int main(int argc, char ** argv) {
    JNIEnv env;
    JavaValue path{"fixture.gguf"}, prompt{"Hi"};
    Java_com_musab_aragpt2_LlamaNative_open(&env, nullptr, &path, 384, 4);
    const bool cancel_test = argc > 1 && std::string(argv[1]) == "cancel";
    fake_slow_decode = cancel_test;
    std::thread canceller;
    if (cancel_test) {
        canceller = std::thread([] {
            while (!fake_decode_entered) std::this_thread::yield();
            Java_com_musab_aragpt2_LlamaNative_cancel(nullptr, nullptr);
        });
    }
    const auto start = std::chrono::steady_clock::now();
    jstring answer = Java_com_musab_aragpt2_LlamaNative_generate(&env, nullptr, &prompt, 8, nullptr);
    if (canceller.joinable()) canceller.join();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
    bool ok = cancel_test ? fake_aborted.load() && elapsed < 400
                          : answer && answer->text == "OK" && !env.ExceptionCheck();
    // A per-call callback must not outlive its stack-owned state.
    if (g_ctx->abort_callback != nullptr) ok = false;
    Java_com_musab_aragpt2_LlamaNative_close(&env, nullptr);
    std::printf("%s %s (%lld ms)\n", cancel_test ? "cancel_during_decode" : "token_storage_lifetime",
            ok ? "PASS" : "FAIL", static_cast<long long>(elapsed));
    return ok ? 0 : 1;
}
