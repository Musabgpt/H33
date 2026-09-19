package com.musab.aragpt2;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class TextFileTool {
    private TextFileTool() {}

    public static boolean wantsTextFile(String request) {
        if (request == null) return false;
        String q = request.toLowerCase(Locale.ROOT);
        return q.contains("ملف نصي") ||
                q.contains("ملف txt") ||
                q.contains(".txt") ||
                q.contains("انشئ ملف") ||
                q.contains("أنشئ ملف") ||
                q.contains("احفظه كملف") ||
                q.contains("احفظ كملف");
    }

    public static String suggestedFileName(String request) {
        String base = "H33_" + System.currentTimeMillis();
        if (request != null) {
            Pattern p = Pattern.compile("(?:باسم|اسمه|اسم الملف)\\s*[\\"'«»]?([^\\n\\r\\"'«»]{1,40})");
            Matcher m = p.matcher(request);
            if (m.find()) base = m.group(1).trim();
        }
        base = base.replaceAll("[\\/:*?\\"<>|]", "_")
                .replaceAll("\\s+", "_")
                .replaceAll("\\.txt$", "");
        if (base.length() > 48) base = base.substring(0, 48);
        if (base.isEmpty()) base = "H33_" + System.currentTimeMillis();
        return base + ".txt";
    }

    public static String save(Context context, String fileName, String content) throws Exception {
        String safe = sanitize(fileName);
        byte[] bytes = (content == null ? "" : content).getBytes(StandardCharsets.UTF_8);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ContentResolver resolver = context.getContentResolver();
            ContentValues values = new ContentValues();
            values.put(MediaStore.Downloads.DISPLAY_NAME, safe);
            values.put(MediaStore.Downloads.MIME_TYPE, "text/plain");
            values.put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/H33");
            values.put(MediaStore.Downloads.IS_PENDING, 1);

            Uri uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
            if (uri == null) throw new IllegalStateException("تعذر إنشاء الملف");

            try {
                try (OutputStream out = resolver.openOutputStream(uri, "w")) {
                    if (out == null) throw new IllegalStateException("تعذر فتح الملف للكتابة");
                    out.write(bytes);
                    out.flush();
                }

                ContentValues done = new ContentValues();
                done.put(MediaStore.Downloads.IS_PENDING, 0);
                resolver.update(uri, done, null, null);
                return "Downloads/H33/" + safe;
            } catch (Exception e) {
                resolver.delete(uri, null, null);
                throw e;
            }
        }

        File root = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS);
        if (root == null) root = context.getFilesDir();
        File dir = new File(root, "H33");
        if (!dir.exists() && !dir.mkdirs()) throw new IllegalStateException("تعذر إنشاء مجلد H33");
        File file = new File(dir, safe);
        try (FileOutputStream out = new FileOutputStream(file, false)) {
            out.write(bytes);
            out.getFD().sync();
        }
        return file.getAbsolutePath();
    }

    private static String sanitize(String fileName) {
        String name = fileName == null ? "H33.txt" : fileName.trim();
        if (name.isEmpty()) name = "H33.txt";
        name = name.replaceAll("[\\/:*?\\"<>|]", "_");
        if (!name.toLowerCase(Locale.ROOT).endsWith(".txt")) name += ".txt";
        if (name.length() > 64) {
            name = name.substring(0, Math.min(60, name.length())) + ".txt";
        }
        return name;
    }
}
