package com.musab.aragpt2;

final class LlamaNative {
    interface TokenCallback { void onToken(String token); }
    private static boolean loaded;
    private LlamaNative() {}
    static synchronized void load() {
        if (!loaded) { System.loadLibrary("h33_llama"); loaded = true; }
    }
    static native String open(String modelPath, int contextSize, int threads);
    static native String generate(String prompt, int maxTokens, TokenCallback callback);
    static native void cancel();
    static native void close();
}
