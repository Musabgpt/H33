package com.musab.aragpt2;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.Gravity;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.*;
import androidx.appcompat.app.AppCompatActivity;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** One offline model, one clean chat, no browser/search/hosted AI. */
public final class MainActivity extends AppCompatActivity {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private CodeModelEngine engine;
    private ToolRegistry tools;
    private TextView status;
    private ScrollView scroll;
    private LinearLayout messages;
    private EditText input;
    private Button send, stop, newChat, toolsButton;
    private volatile long generationId;
    private boolean busy;
    private String lastAnswer = "";

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.activity_main);
        status = findViewById(R.id.status);
        scroll = findViewById(R.id.chatScroll);
        messages = findViewById(R.id.messagesContainer);
        input = findViewById(R.id.inputBox);
        send = findViewById(R.id.sendButton);
        stop = findViewById(R.id.stopButton);
        newChat = findViewById(R.id.newChatButton);
        toolsButton = findViewById(R.id.plusButton);
        tools = ToolRegistry.withBuiltIns(this);
        welcome();
        setBusy(true, "جاري تشغيل DeepSeek-Coder 1.3B…");
        worker.execute(() -> {
            try {
                engine = new CodeModelEngine(this, tools);
                List<CodeModelEngine.ChatTurn> history = engine.getConversationSnapshot();
                runOnUiThread(() -> { renderHistory(history); setBusy(false, ready()); });
            } catch (Exception e) {
                runOnUiThread(() -> setBusy(true, "تعذر تشغيل النموذج: " + safe(e)));
            }
        });
        send.setOnClickListener(v -> generate());
        stop.setOnClickListener(v -> cancel());
        newChat.setOnClickListener(v -> newConversation());
        toolsButton.setOnClickListener(v -> showTools());
        input.setOnEditorActionListener((v, action, event) -> {
            if (action == EditorInfo.IME_ACTION_SEND) { generate(); return true; }
            return false;
        });
    }

    private void generate() {
        if (busy || engine == null) return;
        String question = input.getText().toString().trim();
        if (question.isEmpty()) return;
        input.setText("");
        bubble(question, true);
        TextView answerView = bubble("…", false);
        long id = ++generationId;
        long[] paint = {0};
        setBusy(true, "DeepSeek يفكر محليًا…");
        worker.execute(() -> {
            try {
                String answer = engine.generateCandidate(question, 384, text -> {
                    long now = SystemClock.uptimeMillis();
                    if (id != generationId || now - paint[0] < 40) return;
                    paint[0] = now;
                    runOnUiThread(() -> { answerView.setText(text); down(); });
                });
                if (id != generationId) return;
                engine.commitCanonicalTurn(UUID.randomUUID().toString(), question, answer);
                lastAnswer = answer;
                runOnUiThread(() -> {
                    answerView.setText(answer.isEmpty() ? "لم يولّد النموذج جوابًا." : answer);
                    setBusy(false, ready()); down();
                });
            } catch (Exception e) {
                runOnUiThread(() -> { answerView.setText("خطأ: " + safe(e)); setBusy(false, ready()); });
            }
        });
    }

    private void showTools() {
        PopupMenu menu = new PopupMenu(this, toolsButton);
        for (LocalTool tool : tools.snapshot()) menu.getMenu().add("🔧 " + tool.displayName());
        menu.getMenu().add("نسخ آخر جواب");
        menu.getMenu().add("حفظ آخر جواب TXT");
        menu.setOnMenuItemClickListener(item -> {
            String title = item.getTitle().toString();
            if (title.equals("نسخ آخر جواب")) copyLast();
            else if (title.equals("حفظ آخر جواب TXT")) saveLast();
            else Toast.makeText(this, "الأداة متاحة للنموذج عند طلبها", Toast.LENGTH_SHORT).show();
            return true;
        });
        menu.show();
    }

    private void cancel() {
        generationId++;
        if (engine != null) engine.cancelGeneration();
        setBusy(false, "تم إيقاف التوليد");
    }

    private void newConversation() {
        generationId++;
        if (engine == null) return;
        engine.cancelGeneration();
        worker.execute(() -> {
            try { engine.newConversation(); } catch (Exception ignored) {}
            runOnUiThread(() -> { messages.removeAllViews(); welcome(); setBusy(false, ready()); });
        });
    }

    private void welcome() { bubble("DeepSeek-Coder يعمل محليًا. اسأل عن Python أو اطلب أداة محلية.", false); }
    private String ready() { return "جاهز • بدون إنترنت • أدوات: " + tools.size(); }
    private void renderHistory(List<CodeModelEngine.ChatTurn> history) {
        if (history.isEmpty()) return;
        messages.removeAllViews();
        for (CodeModelEngine.ChatTurn turn : history) {
            bubble(turn.content, "user".equals(turn.role));
            if ("assistant".equals(turn.role)) lastAnswer = turn.content;
        }
    }

    private TextView bubble(String text, boolean user) {
        TextView view = new TextView(this);
        view.setText(text); view.setTextSize(16); view.setTextIsSelectable(true);
        view.setPadding(dp(14), dp(11), dp(14), dp(11));
        GradientDrawable bg = new GradientDrawable();
        bg.setCornerRadius(dp(18));
        bg.setColor(user ? Color.rgb(224,231,255) : Color.rgb(246,247,250));
        view.setBackground(bg);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-2, -2);
        lp.gravity = user ? Gravity.END : Gravity.START;
        lp.setMargins(0, dp(5), 0, dp(5));
        view.setLayoutParams(lp); messages.addView(view); down(); return view;
    }

    private void setBusy(boolean value, String message) {
        busy = value; status.setText(message);
        send.setEnabled(!value && engine != null); input.setEnabled(!value);
        stop.setVisibility(value && engine != null ? View.VISIBLE : View.GONE);
    }
    private void down() { scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN)); }
    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }
    private static String safe(Throwable e) { return e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage(); }
    private void copyLast() {
        if (lastAnswer.isEmpty()) return;
        ClipboardManager cm = (ClipboardManager)getSystemService(Context.CLIPBOARD_SERVICE);
        cm.setPrimaryClip(ClipData.newPlainText("DeepSeek answer", lastAnswer));
        Toast.makeText(this, "تم النسخ", Toast.LENGTH_SHORT).show();
    }
    private void saveLast() {
        if (lastAnswer.isEmpty()) return;
        worker.execute(() -> { try {
            String path = TextFileTool.save(this, "DeepSeek_" + System.currentTimeMillis() + ".txt", lastAnswer);
            runOnUiThread(() -> Toast.makeText(this, "تم الحفظ: " + path, Toast.LENGTH_LONG).show());
        } catch (Exception e) { runOnUiThread(() -> Toast.makeText(this, safe(e), Toast.LENGTH_LONG).show()); } });
    }
    @Override protected void onDestroy() {
        generationId++; worker.shutdownNow(); if (engine != null) engine.close(); super.onDestroy();
    }
}
