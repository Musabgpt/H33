package com.musab.aragpt2;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.TextView;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;

/** One-time system picker used to import the external GGUF model into app-private storage. */
public final class ModelImportActivity extends Activity {
    private static final int PICK_MODEL = 7001;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        TextView status = new TextView(this);
        status.setText("H33\n\nاختر ملف DeepSeek-Coder GGUF\n\nسيتم نسخه إلى تخزين التطبيق مرة واحدة.");
        status.setTextSize(20f);
        status.setPadding(48, 80, 48, 48);
        setContentView(status);

        Intent picker = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        picker.addCategory(Intent.CATEGORY_OPENABLE);
        picker.setType("application/octet-stream");
        picker.putExtra(Intent.EXTRA_MIME_TYPES,
                new String[]{"application/octet-stream", "application/x-gguf"});
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
        try {
            File dir = new File(getFilesDir(), "models");
            if (!dir.exists() && !dir.mkdirs()) {
                throw new IllegalStateException("Cannot create model directory");
            }
            File temp = new File(dir, "deepseek-coder.tmp.gguf");
            File target = new File(dir, "deepseek-coder-1.3b-instruct.Q4_K_M.gguf");

            try (InputStream in = getContentResolver().openInputStream(uri);
                 FileOutputStream out = new FileOutputStream(temp)) {
                if (in == null) throw new IllegalStateException("Cannot open selected file");
                byte[] buffer = new byte[8 * 1024 * 1024];
                int n;
                long total = 0;
                while ((n = in.read(buffer)) != -1) {
                    out.write(buffer, 0, n);
                    total += n;
                    if (total > 2L * 1024L * 1024L * 1024L) {
                        throw new IllegalStateException("Model is unexpectedly larger than 2 GB");
                    }
                }
                out.getFD().sync();
            }

            if (!hasGgufMagic(temp)) {
                temp.delete();
                throw new IllegalArgumentException("الملف المحدد ليس GGUF صالحًا");
            }

            if (target.exists() && !target.delete()) {
                throw new IllegalStateException("Cannot replace existing model");
            }
            if (!temp.renameTo(target)) {
                try (FileInputStream in = new FileInputStream(temp);
                     FileOutputStream out = new FileOutputStream(target)) {
                    byte[] buffer = new byte[8 * 1024 * 1024];
                    int n;
                    while ((n = in.read(buffer)) != -1) out.write(buffer, 0, n);
                    out.getFD().sync();
                }
                temp.delete();
            }
        } catch (Exception ex) {
            new android.app.AlertDialog.Builder(this)
                    .setTitle("فشل استيراد النموذج")
                    .setMessage(ex.getMessage() == null ? "خطأ غير معروف" : ex.getMessage())
                    .setPositiveButton("إغلاق", (d, w) -> finish())
                    .show();
            return;
        }

        finish();
    }

    private static boolean hasGgufMagic(File file) throws Exception {
        try (FileInputStream in = new FileInputStream(file)) {
            return in.read() == 'G' && in.read() == 'G' && in.read() == 'U' && in.read() == 'F';
        }
    }
}
