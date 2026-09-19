package com.musab.aragpt2;

import android.app.AlertDialog;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.res.Configuration;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.view.Gravity;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    private QwenEngine engine;
    private TextView status;
    private ScrollView chatScroll;
    private LinearLayout messagesContainer;
    private EditText inputBox;
    private Button sendButton, stopButton, newChatButton;

    private volatile long generationId = 0L;
    private boolean busy = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        status = findViewById(R.id.status);
        chatScroll = findViewById(R.id.chatScroll);
        messagesContainer = findViewById(R.id.messagesContainer);
        inputBox = findViewById(R.id.inputBox);
        sendButton = findViewById(R.id.sendButton);
        stopButton = findViewById(R.id.stopButton);
        newChatButton = findViewById(R.id.newChatButton);

        addWelcomeMessage();

        setBusy(true, "جاري تحميل Qwen2.5-0.5B INT4…");
        stopButton.setVisibility(Button.GONE);

        executor.execute(() -> {
            try {
                engine = new QwenEngine(this);
                List<QwenEngine.ChatTurn> turns = engine.getConversationSnapshot();
                int learned = engine.memoryCount();

                runOnUiThread(() -> {
                    messagesContainer.removeAllViews();
                    if (turns.isEmpty()) {
                        addWelcomeMessage();
                    } else {
                        renderConversation(turns);
                    }
                    setBusy(false, "جاهز • ذاكرة التعلم: " + learned);
                });
            } catch (Exception ex) {
                runOnUiThread(() -> {
                    setBusy(true, "تعذر تحميل Qwen: " + safeMessage(ex));
                    newChatButton.setEnabled(true);
                });
            }
        });

        sendButton.setOnClickListener(v -> sendMessage());
        stopButton.setOnClickListener(v -> stopGeneration());
        newChatButton.setOnClickListener(v -> startNewChat());

        inputBox.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                sendMessage();
                return true;
            }
            return false;
        });
    }

    private void sendMessage() {
        if (busy || engine == null) return;

        String question = inputBox.getText().toString().trim();
        if (question.isEmpty()) return;

        inputBox.setText("");
        addUserBubble(question);

        LinearLayout assistantBlock = new LinearLayout(this);
        assistantBlock.setOrientation(LinearLayout.VERTICAL);
        assistantBlock.setGravity(Gravity.START);
        assistantBlock.setPadding(dp(2), dp(4), dp(2), dp(8));

        TextView assistantBubble = createBubble("…", false);
        assistantBlock.addView(assistantBubble);
        messagesContainer.addView(assistantBlock);
        scrollToBottom();

        final long myGeneration = ++generationId;
        setBusy(true, "Qwen يكتب…");

        executor.execute(() -> {
            try {
                String answer = engine.generateStream(question, 128, fullText -> {
                    if (myGeneration != generationId) return;
                    runOnUiThread(() -> {
                        if (myGeneration != generationId) return;
                        assistantBubble.setText(fullText);
                        scrollToBottom();
                    });
                });

                runOnUiThread(() -> {
                    if (myGeneration != generationId) return;

                    if (answer == null || answer.trim().isEmpty()) {
                        assistantBubble.setText("تم إيقاف التوليد.");
                    } else {
                        assistantBubble.setText(answer);
                        addAssistantActions(assistantBlock, assistantBubble, question, answer);
                    }
                    setBusy(false, "جاهز • ذاكرة التعلم: " + engine.memoryCount());
                    scrollToBottom();
                });
            } catch (Exception ex) {
                runOnUiThread(() -> {
                    if (myGeneration != generationId) return;
                    assistantBubble.setText("حدث خطأ: " + safeMessage(ex));
                    setBusy(false, "جاهز");
                });
            }
        });
    }

    private void stopGeneration() {
        if (!busy || engine == null) return;
        engine.cancelGeneration();
        status.setText("جارٍ إيقاف التوليد…");
        stopButton.setEnabled(false);
    }

    private void startNewChat() {
        generationId++;
        if (engine != null) engine.cancelGeneration();

        messagesContainer.removeAllViews();
        addWelcomeMessage();
        inputBox.setText("");
        setBusy(engine == null, engine == null ? "النموذج غير جاهز" : "محادثة جديدة");

        if (engine != null) {
            executor.execute(() -> {
                try {
                    engine.newConversation();
                    runOnUiThread(() -> setBusy(false,
                            "جاهز • ذاكرة التعلم: " + engine.memoryCount()));
                } catch (Exception ex) {
                    runOnUiThread(() ->
                            status.setText("تعذر بدء محادثة جديدة: " + safeMessage(ex)));
                }
            });
        }
    }

    private void renderConversation(List<QwenEngine.ChatTurn> turns) {
        String pendingQuestion = null;
        for (QwenEngine.ChatTurn turn : turns) {
            if ("user".equals(turn.role)) {
                pendingQuestion = turn.content;
                addUserBubble(turn.content);
            } else if ("assistant".equals(turn.role)) {
                LinearLayout block = new LinearLayout(this);
                block.setOrientation(LinearLayout.VERTICAL);
                block.setGravity(Gravity.START);
                block.setPadding(dp(2), dp(4), dp(2), dp(8));

                TextView bubble = createBubble(turn.content, false);
                block.addView(bubble);

                String q = pendingQuestion == null ? "" : pendingQuestion;
                if (!q.isEmpty()) addAssistantActions(block, bubble, q, turn.content);

                messagesContainer.addView(block);
            }
        }
        scrollToBottom();
    }

    private void addUserBubble(String text) {
        LinearLayout wrapper = new LinearLayout(this);
        wrapper.setOrientation(LinearLayout.VERTICAL);
        wrapper.setGravity(Gravity.END);
        wrapper.setPadding(dp(2), dp(4), dp(2), dp(6));
        wrapper.addView(createBubble(text, true));
        messagesContainer.addView(wrapper);
    }

    private TextView createBubble(String text, boolean user) {
        TextView bubble = new TextView(this);
        bubble.setText(text);
        bubble.setTextSize(17f);
        bubble.setTextIsSelectable(true);
        bubble.setPadding(dp(14), dp(10), dp(14), dp(10));
        bubble.setMaxWidth((int) (getResources().getDisplayMetrics().widthPixels * 0.88f));

        boolean night = (getResources().getConfiguration().uiMode
                & Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES;

        int background;
        int foreground;
        if (user) {
            background = night ? Color.rgb(92, 64, 165) : Color.rgb(232, 224, 255);
            foreground = night ? Color.WHITE : Color.rgb(35, 20, 70);
        } else {
            background = night ? Color.rgb(38, 38, 38) : Color.rgb(245, 245, 245);
            foreground = night ? Color.rgb(245, 245, 245) : Color.rgb(25, 25, 25);
        }

        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(background);
        drawable.setCornerRadius(dp(18));
        bubble.setBackground(drawable);
        bubble.setTextColor(foreground);

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        bubble.setLayoutParams(lp);
        return bubble;
    }

    private void addAssistantActions(LinearLayout block, TextView bubble,
                                     String question, String answer) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.START);
        row.setPadding(dp(4), dp(4), dp(4), 0);

        Button correct = smallButton("✓ صحيح");
        Button edit = smallButton("✎ تصحيح");
        Button copy = smallButton("نسخ");

        correct.setOnClickListener(v -> {
            correct.setEnabled(false);
            executor.execute(() -> {
                try {
                    engine.rememberCorrect(question, bubble.getText().toString());
                    int count = engine.memoryCount();
                    runOnUiThread(() -> {
                        correct.setText("✓ تم الحفظ");
                        status.setText("تم تعزيز الذاكرة • العناصر: " + count);
                    });
                } catch (Exception ex) {
                    runOnUiThread(() -> {
                        correct.setEnabled(true);
                        status.setText("فشل الحفظ: " + safeMessage(ex));
                    });
                }
            });
        });

        edit.setOnClickListener(v ->
                showCorrectionDialog(question, bubble.getText().toString(), bubble, correct));

        copy.setOnClickListener(v -> {
            ClipboardManager clipboard =
                    (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            clipboard.setPrimaryClip(
                    ClipData.newPlainText("H33 answer", bubble.getText().toString()));
            Toast.makeText(this, "تم النسخ", Toast.LENGTH_SHORT).show();
        });

        row.addView(correct);
        row.addView(edit);
        row.addView(copy);
        block.addView(row);
    }

    private void showCorrectionDialog(String question, String oldAnswer,
                                      TextView bubble, Button correctButton) {
        EditText correction = new EditText(this);
        correction.setText(oldAnswer);
        correction.setSelectAllOnFocus(false);
        correction.setMinLines(3);
        correction.setMaxLines(10);
        correction.setPadding(dp(16), dp(12), dp(16), dp(12));

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle("اكتب الجواب الأفضل")
                .setView(correction)
                .setNegativeButton("إلغاء", null)
                .setPositiveButton("علّم", null)
                .create();

        dialog.setOnShowListener(ignored ->
                dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener(v -> {
                    String better = correction.getText().toString().trim();
                    if (better.isEmpty()) return;

                    dialog.getButton(AlertDialog.BUTTON_POSITIVE).setEnabled(false);
                    executor.execute(() -> {
                        try {
                            engine.learnCorrection(question, oldAnswer, better);
                            int count = engine.memoryCount();

                            runOnUiThread(() -> {
                                bubble.setText(better);
                                correctButton.setText("✓ تم التصحيح");
                                correctButton.setEnabled(false);
                                status.setText("تعلم التصحيح • العناصر: " + count);
                                dialog.dismiss();
                            });
                        } catch (Exception ex) {
                            runOnUiThread(() -> {
                                dialog.getButton(AlertDialog.BUTTON_POSITIVE).setEnabled(true);
                                Toast.makeText(this,
                                        "فشل التصحيح: " + safeMessage(ex),
                                        Toast.LENGTH_LONG).show();
                            });
                        }
                    });
                }));

        dialog.show();
    }

    private Button smallButton(String text) {
        Button b = new Button(this);
        b.setText(text);
        b.setTextSize(12f);
        b.setAllCaps(false);
        b.setMinWidth(0);
        b.setMinHeight(0);
        b.setPadding(dp(8), dp(2), dp(8), dp(2));

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, dp(40));
        lp.setMarginEnd(dp(4));
        b.setLayoutParams(lp);
        return b;
    }

    private void addWelcomeMessage() {
        LinearLayout wrapper = new LinearLayout(this);
        wrapper.setOrientation(LinearLayout.VERTICAL);
        wrapper.setGravity(Gravity.CENTER_HORIZONTAL);
        wrapper.setPadding(dp(20), dp(28), dp(20), dp(18));

        TextView title = new TextView(this);
        title.setText("H33 Qwen2.5");
        title.setTextSize(25f);
        title.setTextAlignment(TextView.TEXT_ALIGNMENT_CENTER);
        title.setGravity(Gravity.CENTER);

        TextView subtitle = new TextView(this);
        subtitle.setText("مساعد عربي محلي • المحادثة والتعلم محفوظان على جهازك");
        subtitle.setTextSize(14f);
        subtitle.setAlpha(0.75f);
        subtitle.setGravity(Gravity.CENTER);
        subtitle.setPadding(0, dp(8), 0, 0);

        wrapper.addView(title);
        wrapper.addView(subtitle);
        messagesContainer.addView(wrapper);
    }

    private void setBusy(boolean isBusy, String message) {
        busy = isBusy;
        status.setText(message);

        sendButton.setEnabled(!isBusy && engine != null);
        inputBox.setEnabled(!isBusy && engine != null);

        stopButton.setVisibility(isBusy && engine != null ? Button.VISIBLE : Button.GONE);
        stopButton.setEnabled(isBusy && engine != null);

        newChatButton.setEnabled(true);
    }

    private void scrollToBottom() {
        chatScroll.post(() -> chatScroll.fullScroll(ScrollView.FOCUS_DOWN));
    }

    private int dp(int value) {
        return (int) (value * getResources().getDisplayMetrics().density + 0.5f);
    }

    private static String safeMessage(Throwable t) {
        if (t == null) return "خطأ غير معروف";
        String m = t.getMessage();
        return (m == null || m.trim().isEmpty()) ? t.getClass().getSimpleName() : m;
    }

    @Override
    protected void onDestroy() {
        generationId++;
        if (engine != null) engine.cancelGeneration();
        executor.shutdownNow();

        if (engine != null) {
            try { engine.close(); } catch (Exception ignored) {}
        }

        super.onDestroy();
    }
}
