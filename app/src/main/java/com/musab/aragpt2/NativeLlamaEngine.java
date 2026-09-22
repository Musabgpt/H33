package com.musab.aragpt2;

import android.content.ContentResolver;
import android.net.Uri;
import android.os.ParcelFileDescriptor;

import java.io.File;

/**
 * Small JNI facade around the pinned llama.cpp runtime.
 * The GGUF file remains external to the APK and is opened through the
 * persisted Android Storage Access Framework URI.
 */
public final class NativeLlamaEngine implements AutoCloseable {
    static {
        System.loadLibrary("h33native");
    }

    private volatile long handle;
    private ParcelFileDescriptor modelFd;

    /** Loads an external GGUF without copying it into APK/app-private storage. */
    public NativeLlamaEngine(ContentResolver resolver, Uri modelUri) throws Exception {
        if (resolver == null || modelUri == null) {
            throw new IllegalArgumentException("GGUF model URI is missing");
        }

        ParcelFileDescriptor fd = resolver.openFileDescriptor(modelUri, "r");
        if (fd == null) throw new IllegalArgumentException("GGUF model cannot be opened");
        modelFd = fd;

        // Keep the SAF descriptor open so /proc/self/fd/<fd> remains valid
        // for the native model lifetime.
        String procPath = "/proc/self/fd/" + fd.getFd();
        handle = nativeCreate(procPath, 3072, 4);
        if (handle == 0L) {
            closeFd();
            throw new IllegalStateException("llama.cpp could not load the selected GGUF model");
        }
    }

    /** Legacy constructor retained for native/unit compatibility. */
    public NativeLlamaEngine(File modelFile) throws Exception {
        if (modelFile == null || !modelFile.isFile() || !modelFile.canRead()) {
            throw new IllegalArgumentException("GGUF model file is missing or unreadable");
        }
        handle = nativeCreate(modelFile.getAbsolutePath(), 3072, 4);
        if (handle == 0L) {
            throw new IllegalStateException("llama.cpp could not load the GGUF model");
        }
    }

    /**
     * Legacy raw-prompt generation. Kept for compatibility; chat generation
     * should use generateChat() so the GGUF's chat template is applied natively.
     */
    public synchronized String generate(String prompt, int maxTokens) throws Exception {
        long h = handle;
        if (h == 0L) throw new IllegalStateException("Native model is closed");
        String answer = nativeGenerate(h, prompt == null ? "" : prompt,
                Math.max(16, Math.min(384, maxTokens)));
        if (answer == null) throw new IllegalStateException("Native inference failed");
        return answer;
    }

    /** Chat generation using llama.cpp's model-native tokenizer.chat_template. */
    public synchronized String generateChat(String[] roles, String[] contents, int maxTokens) throws Exception {
        long h = handle;
        if (h == 0L) throw new IllegalStateException("Native model is closed");
        if (roles == null || contents == null || roles.length != contents.length || roles.length == 0) {
            throw new IllegalArgumentException("Chat messages are invalid");
        }
        String answer = nativeGenerateChat(h, roles, contents,
                Math.max(16, Math.min(384, maxTokens)));
        if (answer == null) throw new IllegalStateException("Native chat inference failed");
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
        closeFd();
    }

    private void closeFd() {
        if (modelFd != null) {
            try { modelFd.close(); } catch (Exception ignored) {}
            modelFd = null;
        }
    }

    private static native long nativeCreate(String path, int contextSize, int threads);
    private static native String nativeGenerate(long handle, String prompt, int maxTokens);
    private static native String nativeGenerateChat(long handle, String[] roles, String[] contents, int maxTokens);
    private static native void nativeCancel(long handle);
    private static native void nativeDestroy(long handle);
}
