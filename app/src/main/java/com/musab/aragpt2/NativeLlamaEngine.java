package com.musab.aragpt2;

import android.content.ContentResolver;
import android.net.Uri;
import android.os.Environment;
import android.os.ParcelFileDescriptor;
import android.provider.DocumentsContract;

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

        // Pass the actual granted descriptor to llama.cpp. The pinned runtime exposes
        // llama_model_load_from_file_ptr(), so no /proc/self/fd re-open is required.
        handle = nativeCreateFromFd(fd.getFd(), 3072, 4);

        // Some local providers expose a real shared-storage path as a fallback.
        // This still keeps the GGUF outside the APK and does not copy the weights.
        if (handle == 0L) {
            String directPath = directSharedStoragePath(resolver, modelUri);
            if (directPath != null) {
                File model = new File(directPath);
                if (model.isFile() && model.canRead()) {
                    handle = nativeCreate(model.getAbsolutePath(), 3072, 4);
                }
            }
        }

        if (handle == 0L) {
            closeFd();
            throw new IllegalStateException(
                    "llama.cpp could not load the selected GGUF model. " +
                    "The selected provider did not expose a seekable GGUF file descriptor.");
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

    private static String directSharedStoragePath(ContentResolver resolver, Uri uri) {
        try {
            if ("file".equalsIgnoreCase(uri.getScheme())) {
                return uri.getPath();
            }

            if (DocumentsContract.isDocumentUri(null, uri)) {
                String documentId = DocumentsContract.getDocumentId(uri);
                String path = ExternalGgufPathPolicy.primaryStoragePath(
                        documentId,
                        Environment.getExternalStorageDirectory().getAbsolutePath());
                if (path != null) return path;
            }
        } catch (Exception ignored) {
            // Fall through: SAF descriptor remains the primary path.
        }
        return null;
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
        closeFd();
    }

    private void closeFd() {
        if (modelFd != null) {
            try { modelFd.close(); } catch (Exception ignored) {}
            modelFd = null;
        }
    }

    private static native long nativeCreate(String path, int contextSize, int threads);
    private static native long nativeCreateFromFd(int fd, int contextSize, int threads);
    private static native String nativeGenerate(long handle, String prompt, int maxTokens);
    private static native void nativeCancel(long handle);
    private static native void nativeDestroy(long handle);
}
