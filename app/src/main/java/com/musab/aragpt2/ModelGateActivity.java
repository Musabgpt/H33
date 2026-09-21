package com.musab.aragpt2;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class ModelGateActivity extends Activity {
    private static final int PICK_MODEL = 7001;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private TextView status;
    private Button importButton;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (ModelManager.isInstalled(this)) { openChat(); return; }
        buildUi();
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setGravity(Gravity.CENTER);
        root.setPadding(40, 40, 40, 40);

        TextView title = new TextView(this);
        title.setText("H33 DeepSeek Coder");
        title.setTextSize(26f);
        title.setGravity(Gravity.CENTER);

        TextView info = new TextView(this);
        info.setText("التطبيق نفسه صغير والنموذج منفصل عن APK.\n\n"
                + "اختر ملف DeepSeek-Coder 1.3B Instruct بصيغة GGUF مرة واحدة.\n\n"
                + "Q3_K_M أصغر، وQ4_K_M أكبر مع جودة أعلى.");
        info.setTextSize(15f);
        info.setGravity(Gravity.CENTER);
        info.setPadding(0, 28, 0, 28);

        importButton = new Button(this);
        importButton.setText("استيراد نموذج GGUF");
        importButton.setOnClickListener(v -> pickModel());

        status = new TextView(this);
        status.setText("لم يتم تثبيت النموذج بعد.");
        status.setGravity(Gravity.CENTER);
        status.setPadding(0, 20, 0, 0);

        root.addView(title);
        root.addView(info);
        root.addView(importButton);
        root.addView(status);
        setContentView(root);
    }

    private void pickModel() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{"application/octet-stream", "application/x-gguf", "*/*"});
        startActivityForResult(intent, PICK_MODEL);
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_MODEL || resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        importButton.setEnabled(false);
        status.setText("جارٍ نسخ النموذج… لا تغلق التطبيق.");
        executor.execute(() -> {
            try {
                ModelManager.importModel(this, uri);
                String hash = ModelManager.sha256(this);
                runOnUiThread(() -> {
                    status.setText("تم تثبيت النموذج • SHA-256: " + hash.substring(0, 12) + "…");
                    openChat();
                });
            } catch (Exception ex) {
                runOnUiThread(() -> {
                    importButton.setEnabled(true);
                    status.setText("فشل الاستيراد: " + safe(ex));
                    Toast.makeText(this, "ملف GGUF غير صالح أو تعذر نسخه", Toast.LENGTH_LONG).show();
                });
            }
        });
    }

    private void openChat() {
        startActivity(new Intent(this, MainActivity.class));
        finish();
    }

    private static String safe(Throwable t) {
        String m = t == null ? null : t.getMessage();
        return m == null || m.trim().isEmpty() ? "خطأ غير معروف" : m;
    }

    @Override protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }
}
