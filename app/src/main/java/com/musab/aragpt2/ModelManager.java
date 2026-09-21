package com.musab.aragpt2;

import android.content.Context;
import android.net.Uri;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.security.MessageDigest;

public final class ModelManager {
    private static final String MODEL_NAME = "deepseek-coder-1.3b.gguf";
    private ModelManager() {}

    public static File modelFile(Context context) {
        return new File(new File(context.getApplicationContext().getFilesDir(), "models"), MODEL_NAME);
    }

    public static boolean isInstalled(Context context) {
        File f = modelFile(context);
        return f.isFile() && f.length() >= 50L * 1024L * 1024L && isGgufQuiet(f);
    }

    public static void importModel(Context context, Uri uri) throws Exception {
        File destination = modelFile(context);
        File parent = destination.getParentFile();
        if (parent == null || (!parent.exists() && !parent.mkdirs())) {
            throw new IllegalStateException("Cannot create model directory");
        }
        File partial = new File(parent, MODEL_NAME + ".part");
        try (InputStream in = context.getContentResolver().openInputStream(uri);
             FileOutputStream out = new FileOutputStream(partial)) {
            if (in == null) throw new IllegalStateException("Cannot open selected model");
            byte[] buffer = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buffer)) != -1) out.write(buffer, 0, n);
            out.getFD().sync();
        }
        if (partial.length() < 50L * 1024L * 1024L || !isGguf(partial)) {
            partial.delete();
            throw new IllegalStateException("Selected file is not a valid GGUF model");
        }
        if (destination.exists() && !destination.delete()) {
            partial.delete();
            throw new IllegalStateException("Cannot replace existing model");
        }
        if (!partial.renameTo(destination)) {
            partial.delete();
            throw new IllegalStateException("Cannot finalize model import");
        }
    }

    public static String sha256(Context context) throws Exception {
        File file = modelFile(context);
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (FileInputStream in = new FileInputStream(file)) {
            byte[] buffer = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buffer)) != -1) digest.update(buffer, 0, n);
        }
        StringBuilder out = new StringBuilder(64);
        for (byte b : digest.digest()) out.append(String.format("%02x", b));
        return out.toString();
    }

    private static boolean isGgufQuiet(File file) {
        try { return isGguf(file); } catch (Exception ignored) { return false; }
    }

    private static boolean isGguf(File file) throws Exception {
        try (FileInputStream in = new FileInputStream(file)) {
            return in.read() == 'G' && in.read() == 'G' && in.read() == 'U' && in.read() == 'F';
        }
    }
}
