package com.musab.aragpt2;

import android.app.AlertDialog;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.res.Configuration;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.PopupMenu;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    private QwenEngine engine;
    private LinearLayout rootLayout;
    private TextView status;
    private ScrollView chatScroll;
    private LinearLayout messagesContainer;
    private LinearLayout composerBar;
    private EditText inputBox;
    private Button sendButton, stopButton, newChatButton, plusButton;

    private volatile long generationId = 0L;
    private boolean busy = false;
    private boolean forceWebSearchNext = false;
    private String lastAssistantAnswer = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        setContentView(R.layout.activity_main);

        rootLayout = findViewById(R.id.rootLayout);
        status = findViewById(R.id.status);
        chatScroll = findViewById(R.id.chatScroll);
        messagesContainer = findViewById(R.id.messagesContainer);
        composerBar = findViewById(R.id.composerBar);
        inputBox = findViewById(R.id.inputBox);
        sendButton = findViewById(R.id.sendButton);
        stopButton = findViewById(R.id.stopButton);
        newChatButton = findViewById(R.id.newChatButton);
        plusButton = findViewById(R.id.plusButton);

        styleComposer();
        installInsetsHandling();
        addWelcomeMessage();

        setBusy(true, "جاري تحميل Qwen2.5 INT4 + Fable v1…");
        stopButton.setVisibility(View.GONE);

        executor.execute(() -> {
            try {
                engine = new QwenEngine(this);
                List<QwenEngine.ChatTurn> turns = engine.getConversationSnapshot();
                int learned = engine.memoryCount();

                runOnUiThread(() -> {
                    messagesContainer.removeAllViews();
                    if (turns.isEmpty()) addWelcomeMessage();
                    else renderConversation(turns);
                    setBusy(false, "جاهز • Fable v1 • ذاكرة: " + learned);
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
        plusButton.setOnClickListener(v -> showComposerMenu());

        inputBox.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                sendMessage();
                return true;
            }
            return false;
        });
    }

    private void installInsetsHandling() {
        ViewCompat.setOnApplyWindowInsetsListener(rootLayout, (v, windowInsets) -> {
            Insets bars = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars());
            Insets ime = windowInsets.getInsets(WindowInsetsCompat.Type.ime());
            boolean imeVisible = windowInsets.isVisible(WindowInsetsCompat.Type.ime());

            rootLayout.setPadding(
                    dp(12) + bars.left,
                    dp(10) + bars.top,
                    dp(12) + bars.right,
                    0
            );

            ViewGroup.MarginLayoutParams lp =
                    (ViewGroup.MarginLayoutParams) composerBar.getLayoutParams();

            int targetBottom = (imeVisible ? ime.bottom : bars.bottom) + dp(8);
            if (lp.bottomMargin != targetBottom) {
                lp.bottomMargin = targetBottom;
                composerBar.setLayoutParams(lp);
            }

            if (imeVisible) scrollToBottom();
            return windowInsets;
        });

        ViewCompat.requestApplyInsets(rootLayout);
    }

    private void styleComposer() {
        boolean night = (getResources().getConfiguration().uiMode
                & Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES;

        int composerColor = night ? Color.rgb(47, 47, 47) : Color.rgb(242, 242, 242);
        int iconColor = night ? Color.rgb(73, 73, 73) : Color.rgb(225, 225, 225);
        int sendColor = night ? Color.WHITE : Color.rgb(25, 25, 25);
        int sendText = night ? Color.BLACK : Color.WHITE;
        int normalText = night ? Color.WHITE : Color.rgb(25, 25, 25);

        GradientDrawable composer = new GradientDrawable();
        composer.setColor(composerColor);
        composer.setCornerRadius(dp(28));
        composerBar.setBackground(composer);
        composerBar.setElevation(dp(6));

        inputBox.setTextColor(normalText);
        inputBox.setHintTextColor(night ? Color.rgb(175, 175, 175) : Color.rgb(110, 110, 110));

        styleRoundButton(plusButton, iconColor, normalText);
        styleRoundButton(stopButton, iconColor, normalText);
        styleRoundButton(sendButton, sendColor, sendText);
    }

    private void styleRoundButton(Button button, int background, int foreground) {
        GradientDrawable d = new GradientDrawable();
        d.setShape(GradientDrawable.OVAL);
        d.setColor(background);
        button.setBackground(d);
        button.setTextColor(foreground);
        button.setElevation(dp(2));
    }

    private void showComposerMenu() {
        PopupMenu menu = new PopupMenu(this, plusButton);
        menu.getMenu().add("🌐 بحث الإنترنت للرسالة التالية");
        menu.getMenu().add("💾 حفظ آخر جواب TXT");
        menu.getMenu().add("محادثة جديدة");
        menu.getMenu().add("مسح النص");
        menu.getMenu().add("إخفاء لوحة المفاتيح");

        menu.setOnMenuItemClickListener(item -> {
            String title = item.getTitle().toString();

            if (title.startsWith("🌐")) {
                forceWebSearchNext = true;
                status.setText("🌐 بحث الويب مفعّل للرسالة التالية");
            } else if (title.startsWith("💾")) {
                if (lastAssistantAnswer.isEmpty()) {
                    Toast.makeText(this, "لا يوجد جواب لحفظه", Toast.LENGTH_SHORT).show();
                } else {
                    saveTextFile("H33_" + System.currentTimeMillis() + ".txt", lastAssistantAnswer);
                }
            } else if ("محادثة جديدة".equals(title)) {
                startNewChat();
            } else if ("مسح النص".equals(title)) {
                inputBox.setText("");
            } else {
                InputMethodManager imm =
                        (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
                imm.hideSoftInputFromWindow(inputBox.getWindowToken(), 0);
            }
            return true;
        });
        menu.show();
    }

    private void sendMessage() {
        if (busy || engine == null) return;

        String question = inputBox.getText().toString().trim();
        if (question.isEmpty()) return;

        final boolean explicitWebSearch = forceWebSearchNext;
        forceWebSearchNext = false;

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
        setBusy(true, "Qwen يجهز الجواب…");

        executor.execute(() -> {
            try {
                String webContext = "";
                boolean searched = explicitWebSearch || WebSearchClient.shouldAutoSearch(question);

                if (searched) {
                    runOnUiThread(() -> status.setText("🌐 يبحث في الإنترنت…"));
                    try {
                        webContext = WebSearchClient.search(question, 5);
                    } catch (Exception searchError) {
                        runOnUiThread(() ->
                                status.setText("تعذر بحث الويب؛ سأكمل من المعرفة المحلية"));
                    }
                }

                if (searched && !webContext.isEmpty()) {
                    runOnUiThread(() -> status.setText("🌐 Qwen يقرأ نتائج البحث…"));
                } else {
                    runOnUiThread(() -> status.setText("Qwen يكتب…"));
                }

                final long[] lastUiUpdate = {0L};
                String answer = engine.generateStream(question, 128, webContext, fullText -> {
                    if (myGeneration != generationId) return;

                    long now = SystemClock.uptimeMillis();
                    if (now - lastUiUpdate[0] < 35) return;
                    lastUiUpdate[0] = now;

                    runOnUiThread(() -> {
                        if (myGeneration != generationId) return;
                        assistantBubble.setText(fullText);
                        scrollToBottom();
                    });
                });

                String savedPath = "";
                if (answer != null && !answer.trim().isEmpty()
                        && TextFileTool.wantsTextFile(question)) {
                    try {
                        savedPath = TextFileTool.save(
                                this,
                                TextFileTool.suggestedFileName(question),
                                answer
                        );
                    } catch (Exception ignored) {
                        savedPath = "";
                    }
                }

                final String finalAnswer = answer == null ? "" : answer.trim();
                final String finalSavedPath = savedPath;
                final boolean usedWeb = searched && !webContext.isEmpty();

                runOnUiThread(() -> {
                    if (myGeneration != generationId) return;

                    if (finalAnswer.isEmpty()) {
                        assistantBubble.setText("تم إيقاف التوليد.");
                    } else {
                        assistantBubble.setText(finalAnswer);
                        lastAssistantAnswer = finalAnswer;
                        addAssistantActions(assistantBlock, assistantBubble, question, finalAnswer);
                    }

                    String msg = "جاهز • Fable v1";
                    if (usedWeb) msg += " • 🌐 بحث ويب";
                    if (!finalSavedPath.isEmpty()) msg += " • 💾 " + finalSavedPath;
                    setBusy(false, msg);
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
        lastAssistantAnswer = "";
        setBusy(engine == null, engine == null ? "النموذج غير جاهز" : "محادثة جديدة");

        if (engine != null) {
            executor.execute(() -> {
                try {
                    engine.newConversation();
                    runOnUiThread(() ->
                            setBusy(false, "جاهز • Fable v1 • ذاكرة: " + engine.memoryCount()));
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

                lastAssistantAnswer = turn.content;
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
        Button txt = smallButton("TXT");

        correct.setOnClickListener(v -> {
            correct.setEnabled(false);
            executor.execute(() -> {
                try {
                    engine.rememberCorrect(question, bubble.getText().toString());
                    int count = engine.memoryCount();
                    runOnUiThread(() -> {
                        correct.setText("✓ تم الحفظ");
                        status.setText("تم حفظ التقييم • العناصر: " + count);
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

        txt.setOnClickListener(v ->
                saveTextFile("H33_" + System.currentTimeMillis() + ".txt",
                        bubble.getText().toString()));

        row.addView(correct);
        row.addView(edit);
        row.addView(copy);
        row.addView(txt);
        block.addView(row);
    }

    private void saveTextFile(String fileName, String content) {
        status.setText("💾 يحفظ الملف…");
        executor.execute(() -> {
            try {
                String path = TextFileTool.save(this, fileName, content);
                runOnUiThread(() -> {
                    status.setText("💾 تم الحفظ: " + path);
                    Toast.makeText(this, "تم إنشاء الملف النصي", Toast.LENGTH_LONG).show();
                });
            } catch (Exception ex) {
                runOnUiThread(() ->
                        status.setText("فشل إنشاء الملف: " + safeMessage(ex)));
            }
        });
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
                                lastAssistantAnswer = better;
                                correctButton.setText("✓ تم التصحيح");
                                correctButton.setEnabled(false);
                                status.setText("تم حفظ التصحيح • العناصر: " + count);
                                dialog.dismiss();
                            });
                        } catch (Exception ex) {
                            runOnUiThread(() -> {
                                dialog.getButton(AlertDialog.BUTTON_POSITIVE).setEnabled(true);
                                Toast.makeText(
                                        this,
                                        "فشل التصحيح: " + safeMessage(ex),
                                        Toast.LENGTH_LONG
                                ).show();
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
                LinearLayout.LayoutParams.WRAP_CONTENT, dp(40)
        );
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
        subtitle.setText("Fable-for-Qwen v1 • محلي • بحث ويب • إنشاء TXT");
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

        stopButton.setVisibility(isBusy && engine != null ? View.VISIBLE : View.GONE);
        stopButton.setEnabled(isBusy && engine != null);

        newChatButton.setEnabled(true);
        plusButton.setEnabled(true);
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
            try {
                engine.close();
            } catch (Exception ignored) {
            }
        }

        super.onDestroy();
    }
}
