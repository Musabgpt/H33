package com.musab.aragpt2;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int IMPORT_REQUEST = 7001;
    private static final String MODEL_NAME = "deepseek-coder-1.3b-instruct.Q4_K_M.gguf";
    private static final int MAX_NEW_TOKENS = 384;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());
    private final List<String> turns = new ArrayList<>();
    private LinearLayout messages;
    private ScrollView scroll;
    private EditText input;
    private Button send;
    private TextView status;
    private ProgressBar progress;
    private NativeLlamaEngine engine;
    private boolean generating;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildUi();
        loadModelOrImport();
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.rgb(18,18,18));
        TextView header = new TextView(this);
        header.setText("H33  •  DeepSeek Coder");
        header.setTextColor(Color.WHITE); header.setTextSize(22f);
        header.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        header.setPadding(24,28,24,12);
        root.addView(header, new LinearLayout.LayoutParams(-1,-2));
        status = new TextView(this);
        status.setText("جاري تجهيز النموذج المحلي…"); status.setTextColor(Color.LTGRAY);
        status.setTextSize(14f); status.setPadding(24,0,24,12);
        root.addView(status, new LinearLayout.LayoutParams(-1,-2));
        scroll = new ScrollView(this); scroll.setFillViewport(true);
        messages = new LinearLayout(this); messages.setOrientation(LinearLayout.VERTICAL);
        messages.setPadding(16,8,16,16); scroll.addView(messages, new ScrollView.LayoutParams(-1,-1));
        root.addView(scroll, new LinearLayout.LayoutParams(-1,0,1f));
        LinearLayout composer = new LinearLayout(this);
        composer.setOrientation(LinearLayout.HORIZONTAL); composer.setGravity(Gravity.CENTER_VERTICAL);
        composer.setPadding(12,8,12,12);
        input = new EditText(this); input.setHint("اكتب رسالتك…"); input.setTextColor(Color.WHITE);
        input.setHintTextColor(Color.GRAY); input.setTextSize(16f); input.setSingleLine(false); input.setMaxLines(5);
        input.setPadding(18,12,18,12); composer.addView(input,new LinearLayout.LayoutParams(0,-2,1f));
        send = new Button(this); send.setText("إرسال"); send.setOnClickListener(v -> submit());
        composer.addView(send,new LinearLayout.LayoutParams(-2,-2));
        progress = new ProgressBar(this); progress.setVisibility(View.GONE);
        composer.addView(progress,new LinearLayout.LayoutParams(-2,-2));
        root.addView(composer,new LinearLayout.LayoutParams(-1,-2));
        setContentView(root);
    }

    private File modelFile() { return new File(new File(getFilesDir(),"models"),MODEL_NAME); }

    private void loadModelOrImport() {
        File model = modelFile();
        if (!model.isFile() || model.length() < 1024) {
            status.setText("لا يوجد نموذج GGUF. اختر الملف أولًا.");
            startActivityForResult(new Intent(this,ModelImportActivity.class),IMPORT_REQUEST); return;
        }
        status.setText("جاري تحميل DeepSeek-Coder محليًا…"); setComposerEnabled(false);
        executor.execute(() -> {
            try {
                NativeLlamaEngine local = new NativeLlamaEngine(model);
                main.post(() -> { engine=local; status.setText("جاهز • DeepSeek-Coder محلي فقط"); setComposerEnabled(true); });
            } catch(Exception e) { main.post(() -> showError("فشل تحميل النموذج: "+safeMessage(e))); }
        });
    }

    private void submit() {
        if (generating || engine==null) return;
        String question=input.getText().toString().trim(); if(question.isEmpty()) return;
        input.setText(""); addBubble(question,true); turns.add("User: "+question);
        generating=true; setComposerEnabled(false); status.setText("DeepSeek-Coder يكتب…"); progress.setVisibility(View.VISIBLE);
        TextView answerBubble=addBubble("…",false); String prompt=buildPrompt();
        executor.execute(() -> {
            try {
                String answer=engine.generate(prompt,MAX_NEW_TOKENS);
                if(answer==null || answer.trim().isEmpty()) throw new IllegalStateException("النموذج لم يُرجع نصًا");
                String cleaned=cleanAnswer(answer); turns.add("Assistant: "+cleaned);
                main.post(() -> { answerBubble.setText(cleaned); finishGeneration(); });
            } catch(Exception e) {
                main.post(() -> { answerBubble.setText("حدث خطأ أثناء التوليد: "+safeMessage(e)); finishGeneration(); });
            }
        });
    }

    private String buildPrompt() {
        StringBuilder p=new StringBuilder();
        p.append("You are H33, a helpful local AI assistant running DeepSeek-Coder 1.3B. ");
        p.append("Answer directly and naturally. You are especially strong at Python and programming. ");
        p.append("Use clear explanations and complete code when requested. Do not mention web search or external tools.\n\n");
        int start=Math.max(0,turns.size()-12); for(int i=start;i<turns.size();i++) p.append(turns.get(i)).append("\n");
        p.append("Assistant:"); return p.toString();
    }

    private String cleanAnswer(String answer) {
        String s=answer.trim(); int marker=s.indexOf("Assistant:");
        if(marker>=0) s=s.substring(marker+"Assistant:".length()).trim(); return s;
    }

    private TextView addBubble(String text,boolean user) {
        TextView bubble=new TextView(this); bubble.setText(text); bubble.setTextSize(16f); bubble.setTextColor(Color.WHITE);
        bubble.setPadding(18,14,18,14); bubble.setBackgroundColor(user?Color.rgb(92,55,150):Color.rgb(40,40,40));
        LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(-1,-2); lp.setMargins(user?72:8,8,user?8:72,8);
        messages.addView(bubble,lp); scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN)); return bubble;
    }

    private void finishGeneration(){ generating=false; progress.setVisibility(View.GONE); status.setText("جاهز • DeepSeek-Coder محلي فقط"); setComposerEnabled(true); }
    private void setComposerEnabled(boolean enabled){ if(input!=null) input.setEnabled(enabled); if(send!=null) send.setEnabled(enabled); }
    private void showError(String message){ status.setText(message); setComposerEnabled(false); }
    private static String safeMessage(Exception e){ return e.getMessage()==null?e.getClass().getSimpleName():e.getMessage(); }

    @Override protected void onActivityResult(int requestCode,int resultCode,Intent data){ super.onActivityResult(requestCode,resultCode,data); if(requestCode==IMPORT_REQUEST) loadModelOrImport(); }
    @Override protected void onDestroy(){ generating=false; if(engine!=null) engine.close(); executor.shutdownNow(); super.onDestroy(); }
}
