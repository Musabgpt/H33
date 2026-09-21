package com.musab.aragpt2;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.widget.TextView;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * System picker for an external GGUF model.
 *
 * The selected model stays where the user already has it. We persist the SAF URI
 * instead of copying the model into the APK or app-private storage.
 */
public final class ModelImportActivity extends Activity {
    public static final String PREFS = "h33_model";
    public static final String KEY_URI = "gguf_uri";
    private static final int PICK_MODEL = 7001;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private TextView status;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        status = new TextView(this);
        status.setText("H33\n\nاختر ملف DeepSeek-Coder GGUF الموجود على جهازك");
        status.setTextSize(20f);
        status.setPadding(48, 80, 48, 48);
        setContentView(status);
        openPicker();
    }

    private void openPicker() {
        Intent picker = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        picker.addCategory(Intent.CATEGORY_OPENABLE);
        picker.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        picker.setType("*/*");
        picker.putExtra(Intent.EXTRA_LOCAL_ONLY, true);
        startActivityForResult(picker, PICK_MODEL);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_MODEL) return;
        if (resultCode != RESULT_OK || data == null || data.getData() == null) {
            finish();
            return;
        }

        Uri uri = data.getData();
        int takeFlags = data.getFlags() & Intent.FLAG_GRANT_READ_URI_PERMISSION;
        try {
            if ((data.getFlags() & Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION) != 0) {
                getContentResolver().takePersistableUriPermission(uri, takeFlags);
            }
        } catch (SecurityException ignored) {
            // Some providers do not expose persistable permissions. We can still
            // validate and use the URI for this session, but it may need re-selection later.
        }

        status.setText("جاري فحص ملف GGUF…");
        executor.execute(() -> validateAndSave(uri));
    }

    private void validateAndSave(Uri uri) {
        try {
            String name = queryDisplayName(uri);
            if (name != null && !isGgufName(name)) {
                throw new IllegalArgumentException("اختر ملفًا بامتداد .gguf");
            }
            if (!hasGgufMagic(uri)) {
                throw new IllegalArgumentException("الملف المحدد ليس GGUF صالحًا");
            }

            getSharedPreferences(PREFS, MODE_PRIVATE)
                    .edit()
                    .putString(KEY_URI, uri.toString())
                    .apply();

            runOnUiThread(() -> {
                status.setText("تم ربط نموذج GGUF الخارجي. جاري تشغيل H33…");
                setResult(RESULT_OK);
                finish();
            });
        } catch (Exception ex) {
            runOnUiThread(() -> {
                status.setText("فشل اختيار النموذج: " +
                        (ex.getMessage() == null ? "خطأ غير معروف" : ex.getMessage()));
            });
        }
    }

    static boolean isGgufName(String name) {
        return name != null && name.toLowerCase(java.util.Locale.ROOT).endsWith(".gguf");
    }

    private String queryDisplayName(Uri uri) {
        android.database.Cursor cursor = getContentResolver().query(
                uri,
                new String[]{android.provider.OpenableColumns.DISPLAY_NAME},
                null,
                null,
                null);
        if (cursor == null) return null;
        try {
            if (!cursor.moveToFirst()) return null;
            int index = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME);
            return index >= 0 ? cursor.getString(index) : null;
        } finally {
            cursor.close();
        }
    }

    private boolean hasGgufMagic(Uri uri) throws IOException {
        try (ParcelFileDescriptor pfd = getContentResolver().openFileDescriptor(uri, "r")) {
            if (pfd == null) throw new IOException("Cannot open selected model");
            try (FileInputStream in = new FileInputStream(pfd.getFileDescriptor())) {
                return in.read() == 'G' && in.read() == 'G' && in.read() == 'U' && in.read() == 'F';
            }
        }
    }

    @Override
    protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }
}
