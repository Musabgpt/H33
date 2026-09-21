package com.musab.aragpt2;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** H33 classic chat UI on top of the local-only native DeepSeek-Coder engine. */
public final class MainActivity extends Activity {
    private static final int IMPORT_REQUEST = 7001;
    private static final int MAX_NEW_TOKENS = 384;
    private static final int PURPLE = Color.rgb(177, 112, 255);
    private static final int USER_PURPLE = Color.rgb(91, 55, 150);
    private static final int BG = Color.rgb(15, 15, 15);
    private static final int PANEL = Color.rgb(43, 43, 43);

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());
    private final List<String> userTurns = new ArrayList<>();
    private final List<String> assistantTurns = new ArrayList<>();

    private LinearLayout messages;
    private ScrollView scroll;
    private EditText input;
    private Button sendButton;
    private Button stopButton;
    private Button plusButton;
    private Button modelButton;
    private TextView status;
    private NativeLlamaEngine engine;
    private boolean generating;
    private long generationId;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Window window = getWindow();
        window.setStatusBarColor(Color.BLACK);
        window.setNavigationBarColor(Color.BLACK);
        buildUi();
        loadModelOrImport();
    }

    private int dp(float value) {
        return (int) (value * getResources().getDisplayMetrics().density + 0.5f);
    }

    private GradientDrawable rounded(int color, float radiusDp) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(color);
        d.setCornerRadius(dp(radiusDp));
        return d;
    }

    private Button actionButton(String text, int widthDp, int heightDp) {
        Button b = new Button(this);
        b.setText(text);
        b.setTextColor(Color.WHITE);
        b.setTextSize(15f);
        b.setAllCaps(false);
        b.setMinWidth(0);
        b.setMinHeight(0);
        b.setPadding(dp(8), 0, dp(8), 0);
        b.setBackground(rounded(PURPLE, 15));
        b.setLayoutParams(new LinearLayout.LayoutParams(dp(widthDp), dp(heightDp)));
        return b;
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);
        root.setPadding(dp(12), dp(8), dp(12), dp(7));

        // Classic H33 header: the New button stays on the left and never covers the title.
        FrameLayout header = new FrameLayout(this);
        header.setLayoutParams(new LinearLayout.LayoutParams(-1, dp(67)));

        TextView title = new TextView(this);
        title.setText("H33 DeepSeek Coder");
        title.setTextColor(Color.WHITE);
        title.setTextSize(23f);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        title.setGravity(Gravity.CENTER);
        title.setPadding(dp(143), 0, 0, 0);
        header.addView(title, new FrameLayout.LayoutParams(-1, -1));

        Button newChat = actionButton("＋ جديد", 143, 59);
        FrameLayout.LayoutParams newLp = new FrameLayout.LayoutParams(dp(143), dp(59), Gravity.START | Gravity.CENTER_VERTICAL);
        header.addView(newChat, newLp);
        newChat.setOnClickListener(v -> newChat());
        root.addView(header);

        status = new TextView(this);
        status.setText("جاري تجهيز DeepSeek-Coder…");
        status.setTextColor(Color.LTGRAY);
        status.setTextSize(13f);
        status.setAlpha(0.86f);
        status.setGravity(Gravity.CENTER);
        status.setPadding(0, 0, 0, dp(5));
        root.addView(status, new LinearLayout.LayoutParams(-1, dp(30)));

        modelButton = actionButton("تحميل أوزان GGUF", 190, 48);
        modelButton.setTextSize(15f);
        modelButton.setVisibility(View.GONE);
        modelButton.setOnClickListener(v -> startModelImport());
        LinearLayout.LayoutParams modelLp = new LinearLayout.LayoutParams(dp(190), dp(48));
        modelLp.gravity = Gravity.CENTER_HORIZONTAL;
        modelLp.setMargins(0, 0, 0, dp(4));
        root.addView(modelButton, modelLp);

        scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setClipToPadding(false);
        scroll.setPadding(0, 0, 0, dp(8));
        messages = new LinearLayout(this);
        messages.setOrientation(LinearLayout.VERTICAL);
        messages.setPadding(dp(2), dp(5), dp(2), dp(12));
        scroll.addView(messages, new ScrollView.LayoutParams(-1, -2));
        root.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1f));

        // Classic H33 composer: send / stop on the left, message field in the middle, + on the right.
        LinearLayout composer = new LinearLayout(this);
        composer.setOrientation(LinearLayout.HORIZONTAL);
        composer.setGravity(Gravity.CENTER_VERTICAL);
        composer.setPadding(dp(8), dp(7), dp(8), dp(7));
        composer.setMinimumHeight(dp(76));
        composer.setBackground(rounded(PANEL, 34));

        sendButton = actionButton("↑", 58, 58);
        sendButton.setTextSize(24f);
        sendButton.setContentDescription("إرسال");
        sendButton.setEnabled(false);
        sendButton.setOnClickListener(v -> submit());

        stopButton = actionButton("■", 58, 58);
        stopButton.setTextSize(15f);
        stopButton.setContentDescription("إيقاف");
        stopButton.setVisibility(View.GONE);
        stopButton.setOnClickListener(v -> stopGeneration());

        input = new EditText(this);
        input.setTextColor(Color.WHITE);
        input.setHintTextColor(Color.rgb(155, 155, 155));
        input.setHint("اكتب رسالة...");
        input.setTextSize(17f);
        input.setSingleLine(false);
        input.setMaxLines(6);
        input.setGravity(Gravity.CENTER_VERTICAL | Gravity.START);
        input.setInputType(android.text.InputType.TYPE_CLASS_TEXT | android.text.InputType.TYPE_TEXT_FLAG_MULTI_LINE | android.text.InputType.TYPE_TEXT_FLAG_CAP_SENTENCES);
        input.setBackgroundColor(Color.TRANSPARENT);
        input.setPadding(dp(10), dp(6), dp(10), dp(6));
        input.setTextDirection(View.TEXT_DIRECTION_ANY_RTL);
        input.setImeOptions(android.view.inputmethod.EditorInfo.IME_ACTION_SEND);
        input.addTextChangedListener(new TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            @Override public void onTextChanged(CharSequence s, int start, int before, int count) { updateSendState(); }
            @Override public void afterTextChanged(Editable s) {}
        });
        input.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == android.view.inputmethod.EditorInfo.IME_ACTION_SEND) {
                submit();
                return true;
            }
            return false;
        });

        plusButton = actionButton("+", 58, 58);
        plusButton.setTextSize(26f);
        plusButton.setContentDescription("تحميل أو تغيير نموذج GGUF");
        plusButton.setOnClickListener(v -> startModelImport());

        composer.addView(sendButton);
        composer.addView(stopButton);
        composer.addView(input, new LinearLayout.LayoutParams(0, -2, 1f));
        composer.addView(plusButton);
        LinearLayout.LayoutParams composerLp = new LinearLayout.LayoutParams(-1, -2);
        composerLp.setMargins(0, dp(5), 0, dp(1));
        root.addView(composer, composerLp);

        setContentView(root);
    }

    private void startModelImport() {
        if (generating) return;
        startActivityForResult(new Intent(this, ModelImportActivity.class), IMPORT_REQUEST);
    }

    private Uri savedModelUri() {
        String value = getSharedPreferences(ModelImportActivity.PREFS, MODE_PRIVATE)
                .getString(ModelImportActivity.KEY_URI, null);
        if (value == null || value.trim().isEmpty()) return null;
        try {
            return Uri.parse(value);
        } catch (Exception ignored) {
            return null;
        }
    }

    private void loadModelOrImport() {
        Uri uri = savedModelUri();
        if (uri == null) {
            status.setText("لا يوجد نموذج GGUF — حمّل الأوزان من زر تحميل GGUF");
            modelButton.setVisibility(View.VISIBLE);
            setComposerEnabled(false);
            return;
        }

        modelButton.setVisibility(View.GONE);
        status.setText("جاري تحميل DeepSeek-Coder محليًا…");
        setComposerEnabled(false);
        executor.execute(() -> {
            try {
                NativeLlamaEngine local = new NativeLlamaEngine(getContentResolver(), uri);
                main.post(() -> {
                    engine = local;
                    status.setText("جاهز • DeepSeek-Coder محلي فقط");
                    setComposerEnabled(true);
                });
            } catch (Exception e) {
                main.post(() -> {
                    status.setText("فشل فتح النموذج المحفوظ: " + safeMessage(e));
                    modelButton.setText("إعادة تحميل أوزان GGUF");
                    modelButton.setVisibility(View.VISIBLE);
                    setComposerEnabled(false);
                });
            }
        });
    }

    private void submit() {
        if (generating || engine == null) return;
        String question = input.getText().toString().trim();
        if (question.isEmpty()) return;
        input.setText("");
        addBubble(question, true);
        userTurns.add(question);
        assistantTurns.add("");
        final int answerIndex = assistantTurns.size() - 1;
        final TextView answerBubble = addBubble("يكتب…", false);
        final String prompt = buildPrompt();
        final long runId = ++generationId;

        generating = true;
        setComposerEnabled(false);
        sendButton.setVisibility(View.GONE);
        stopButton.setVisibility(View.VISIBLE);
        status.setText("DeepSeek-Coder يكتب…");

        executor.execute(() -> {
            try {
                String answer = engine.generate(prompt, MAX_NEW_TOKENS);
                if (answer == null || answer.trim().isEmpty()) throw new IllegalStateException("النموذج لم يُرجع نصًا");
                String cleaned = cleanAnswer(answer);
                assistantTurns.set(answerIndex, cleaned);
                main.post(() -> animateAnswer(answerBubble, cleaned, runId));
            } catch (Exception e) {
                main.post(() -> {
                    if (runId != generationId) return;
                    answerBubble.setText("حدث خطأ أثناء التوليد: " + safeMessage(e));
                    finishGeneration();
                });
            }
        });
    }

    private void animateAnswer(TextView bubble, String text, long runId) {
        if (runId != generationId) return;
        bubble.setText("");
        final int[] index = {0};
        Runnable typer = new Runnable() {
            @Override public void run() {
                if (runId != generationId) return;
                if (index[0] >= text.length()) { finishGeneration(); return; }
                int next = Math.min(text.length(), index[0] + 3);
                bubble.setText(text.substring(0, next));
                index[0] = next;
                scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
                main.postDelayed(this, 8);
            }
        };
        main.post(typer);
    }

    private void stopGeneration() {
        if (!generating) return;
        generationId++;
        if (engine != null) engine.cancel();
        generating = false;
        status.setText("تم إيقاف التوليد");
        sendButton.setVisibility(View.VISIBLE);
        stopButton.setVisibility(View.GONE);
        setComposerEnabled(true);
    }

    private String buildPrompt() {
        StringBuilder p = new StringBuilder();
        p.append("You are H33, a helpful local AI assistant using DeepSeek Coder. ");
        p.append("Answer directly and naturally. You are especially strong at Python and programming. ");
        p.append("Give clear explanations and complete code when requested.\n");
        int start = Math.max(0, userTurns.size() - 6);
        for (int i = start; i < userTurns.size(); i++) {
            p.append("### Instruction:\n").append(userTurns.get(i)).append("\n");
            if (i < assistantTurns.size() - 1 && !assistantTurns.get(i).isEmpty()) {
                p.append("### Response:\n").append(assistantTurns.get(i)).append("\n<|EOT|>\n");
            }
        }
        p.append("### Response:");
        return p.toString();
    }

    private String cleanAnswer(String answer) {
        String s = answer.trim();
        int eot = s.indexOf("<|EOT|>");
        if (eot >= 0) s = s.substring(0, eot).trim();
        int marker = s.indexOf("### Response:");
        if (marker >= 0) s = s.substring(marker + "### Response:".length()).trim();
        return s;
    }

    private TextView addBubble(String text, boolean user) {
        FrameLayout row = new FrameLayout(this);
        row.setLayoutParams(new LinearLayout.LayoutParams(-1, -2));
        row.setPadding(dp(4), dp(4), dp(4), dp(4));
        TextView bubble = new TextView(this);
        bubble.setText(text);
        bubble.setTextSize(16f);
        bubble.setTextColor(Color.WHITE);
        bubble.setGravity(Gravity.CENTER_VERTICAL | Gravity.START);
        bubble.setPadding(dp(18), dp(13), dp(18), dp(13));
        bubble.setTextIsSelectable(true);
        bubble.setBackground(rounded(user ? USER_PURPLE : PANEL, 22));
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(-2, -2);
        lp.gravity = user ? Gravity.END : Gravity.START;
        lp.width = (int) (getResources().getDisplayMetrics().widthPixels * (user ? 0.78f : 0.90f));
        row.addView(bubble, lp);
        messages.addView(row);
        scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
        return bubble;
    }

    private void newChat() {
        generationId++;
        if (generating && engine != null) engine.cancel();
        generating = false;
        userTurns.clear();
        assistantTurns.clear();
        messages.removeAllViews();
        input.setText("");
        status.setText(engine == null ? "لا يوجد نموذج GGUF — حمّل الأوزان" : "جاهز • DeepSeek-Coder محلي فقط");
        modelButton.setVisibility(engine == null ? View.VISIBLE : View.GONE);
        setComposerEnabled(engine != null);
        sendButton.setVisibility(View.VISIBLE);
        stopButton.setVisibility(View.GONE);
    }

    private void finishGeneration() {
        generating = false;
        sendButton.setVisibility(View.VISIBLE);
        stopButton.setVisibility(View.GONE);
        status.setText("جاهز • DeepSeek-Coder محلي فقط");
        setComposerEnabled(true);
    }

    private void updateSendState() {
        if (!generating && engine != null && input != null) sendButton.setEnabled(!input.getText().toString().trim().isEmpty());
    }

    private void setComposerEnabled(boolean enabled) {
        if (input != null) input.setEnabled(enabled);
        if (plusButton != null) plusButton.setEnabled(true);
        updateSendState();
    }

    private static String safeMessage(Exception e) {
        return e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == IMPORT_REQUEST && resultCode == RESULT_OK) loadModelOrImport();
    }

    @Override protected void onDestroy() {
        generationId++;
        generating = false;
        if (engine != null) {
            try { engine.cancel(); } catch (Exception ignored) {}
            try { engine.close(); } catch (Exception ignored) {}
        }
        executor.shutdownNow();
        super.onDestroy();
    }
}
