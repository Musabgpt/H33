package com.musab.aragpt2;

import java.io.File;

public final class NativeCodeModel implements AutoCloseable {
    static { System.loadLibrary("h33native"); }
    private boolean loaded;

    public synchronized void load(File modelFile, int contextSize) {
        if (modelFile == null || !modelFile.isFile() || modelFile.length() < 50L * 1024L * 1024L) {
            throw new IllegalArgumentException("GGUF model file is missing or invalid");
        }
        loaded = nativeLoad(modelFile.getAbsolutePath(), contextSize);
        if (!loaded) throw new IllegalStateException("llama.cpp could not load the GGUF model");
    }

    public synchronized String generate(String prompt, int maxTokens, int contextSize) {
        if (!loaded) throw new IllegalStateException("model not loaded");
        return nativeGenerate(prompt, maxTokens, contextSize);
    }

    @Override public synchronized void close() {
        if (loaded) {
            nativeUnload();
            loaded = false;
        }
    }

    private native boolean nativeLoad(String path, int contextSize);
    private native String nativeGenerate(String prompt, int maxTokens, int contextSize);
    private native void nativeUnload();
}
