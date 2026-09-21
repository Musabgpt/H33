package com.musab.aragpt2;

import java.io.File;

/**
 * Small JNI facade around the pinned llama.cpp runtime.
 * The GGUF file stays outside the APK and is memory-mapped by llama.cpp.
 */
public final class NativeLlamaEngine implements AutoCloseable {
    static {
        System.loadLibrary("h33native");
    }

    private volatile long handle;

    public NativeLlamaEngine(File modelFile) throws Exception {
        if (modelFile == null || !modelFile.isFile() || !modelFile.canRead()) {
            throw new IllegalArgumentException("GGUF model file is missing or unreadable");
        }
        handle = nativeCreate(modelFile.getAbsolutePath(), 3072, 4);
        if (handle == 0L) {
            throw new IllegalStateException("llama.cpp could not load the GGUF model");
        }
    }

    public synchronized String generate(String prompt, int maxTokens) throws Exception {
        long h = handle;
        if (h == 0L) throw new IllegalStateException("Native model is closed");
        String answer = nativeGenerate(h, prompt == null ? "" : prompt,
                Math.max(16, Math.min(384, maxTokens)));
        if (answer == null) throw new IllegalStateException("Native inference failed");
        return answer;
    }

    /** Cancellation is intentionally not synchronized so it can interrupt native generation. */
    public void cancel() {
        long h = handle;
        if (h != 0L) nativeCancel(h);
    }

    @Override
    public synchronized void close() {
        if (handle != 0L) {
            nativeDestroy(handle);
            handle = 0L;
        }
    }

    private static native long nativeCreate(String path, int contextSize, int threads);
    private static native String nativeGenerate(long handle, String prompt, int maxTokens);
    private static native void nativeCancel(long handle);
    private static native void nativeDestroy(long handle);
}
