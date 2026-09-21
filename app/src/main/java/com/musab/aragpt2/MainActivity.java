package com.musab.aragpt2;

import android.app.AlertDialog;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.res.Configuration;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.method.LinkMovementMethod;
import android.text.util.Linkify;
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
import android.webkit.CookieManager;
import android.webkit.WebResourceRequest;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private CodeModelEngine engine;
    private LocalInferenceEngine localInference;
    private CandidateCoordinator candidateCoordinator;
    private PreferenceStore preferenceStore;
    private UserDecisionService userDecisionService;
    private LinearLayout rootLayout, messagesContainer, composerBar;
    private ScrollView chatScroll;
    private TextView status;
    private EditText inputBox;
    private Button sendButton, stopButton, newChatButton, plusButton;
    private volatile long generationId = 0L;
    private boolean busy = false;
    private String lastAssistantAnswer = "";

    @Override protected void onCreate(Bundle savedInstanceState) {
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
        setBusy(true, "جاري تحميل DeepSeek-Coder عبر llama.cpp…");
        executor.execute(() -> {
            try {
                engine = new CodeModelEngine(this);
                localInference = new NativeLocalInferenceEngine(engine);
                preferenceStore = new PreferenceStore(this);
                userDecisionService = new UserDecisionService(preferenceStore, new UserDecisionService.CanonicalMemory() {
                    @Override public String get(String turnId) { return engine.getCanonicalAnswer(turnId); }
                    @Override public void commit(String turnId, String q, String a) throws Exception { engine.commitCanonicalTurn(turnId, q, a); }
                    @Override public boolean replace(String turnId, String a) throws Exception { return engine.replaceCanonicalAnswer(turnId, a); }
                    @Override public boolean remove(String turnId) throws Exception { return engine.removeCanonicalTurn(turnId); }
                });
                candidateCoordinator = new CandidateCoordinator(
                        new DeepSeekCoderLocalAnswerProvider(localInference, 320),
                        new ExtractiveWebEvidenceAnswerProvider(new WebEvidenceRetriever(5)),
                        new GoogleAiOverviewProvider(this));
                List<CodeModelEngine.ChatTurn> turns = engine.getConversationSnapshot();
                int preferences = preferenceStore.eventCount();
                runOnUiThread(() -> {
                    messagesContainer.removeAllViews();
                    if (turns.isEmpty()) addWelcomeMessage(); else renderConversation(turns);
                    setBusy(false, "جاهز • DeepSeek Python • ذاكرة مستخدم: " + preferences);
                });
            } catch (Exception ex) {
                runOnUiThread(() -> setBusy(true, "تعذر تحميل DeepSeek: " + safeMessage(ex)));
            }
        });
        sendButton.setOnClickListener(v -> sendMessage());
        stopButton.setOnClickListener(v -> stopGeneration());
        newChatButton.setOnClickListener(v -> startNewChat());
        plusButton.setOnClickListener(v -> showComposerMenu());
        inputBox.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_SEND) { sendMessage(); return true; }
            return false;
        });
    }

    private void installInsetsHandling() {
        ViewCompat.setOnApplyWindowInsetsListener(rootLayout, (v, wi) -> {
            Insets bars = wi.getInsets(WindowInsetsCompat.Type.systemBars());
            Insets ime = wi.getInsets(WindowInsetsCompat.Type.ime());
            boolean visible = wi.isVisible(WindowInsetsCompat.Type.ime());
            rootLayout.setPadding(dp(12) + bars.left, dp(10) + bars.top, dp(12) + bars.right, 0);
            ViewGroup.MarginLayoutParams lp = (ViewGroup.MarginLayoutParams) composerBar.getLayoutParams();
            lp.bottomMargin = (visible ? ime.bottom : bars.bottom) + dp(8);
            composerBar.setLayoutParams(lp);
            if (visible) scrollToBottom();
            return wi;
        });
    }

    private void styleComposer() {
        boolean night = (getResources().getConfiguration().uiMode & Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES;
        GradientDrawable composer = new GradientDrawable();
        composer.setColor(night ? Color.rgb(47,47,47) : Color.rgb(242,242,242));
        composer.setCornerRadius(dp(28));
        composerBar.setBackground(composer);
        inputBox.setTextColor(night ? Color.WHITE : Color.rgb(25,25,25));
        inputBox.setHintTextColor(night ? Color.rgb(175,175,175) : Color.rgb(110,110,110));
        styleRoundButton(plusButton, night ? Color.rgb(73,73,73) : Color.rgb(225,225,225), Color.WHITE);
        styleRoundButton(stopButton, night ? Color.rgb(73,73,73) : Color.rgb(225,225,225), Color.WHITE);
        styleRoundButton(sendButton, night ? Color.WHITE : Color.rgb(25,25,25), night ? Color.BLACK : Color.WHITE);
    }

    private void styleRoundButton(Button b, int bg, int fg) {
        GradientDrawable d = new GradientDrawable(); d.setShape(GradientDrawable.OVAL); d.setColor(bg);
        b.setBackground(d); b.setTextColor(fg); b.setElevation(dp(2));
    }

    private void showComposerMenu() {
        PopupMenu menu = new PopupMenu(this, plusButton);
        menu.getMenu().add("استيراد/تبديل نموذج GGUF");
        menu.getMenu().add("محادثة جديدة");
        menu.getMenu().add("مسح النص");
        menu.setOnMenuItemClickListener(item -> {
            String t = item.getTitle().toString();
            if (t.startsWith("استيراد")) {
                startActivity(new Intent(this, ModelGateActivity.class));
            } else if (t.equals("محادثة جديدة")) startNewChat();
            else inputBox.setText("");
            return true;
        });
        menu.show();
    }

    private void sendMessage() {
        if (busy || engine == null || candidateCoordinator == null) return;
        String question = inputBox.getText().toString().trim();
        if (question.isEmpty()) return;
        inputBox.setText(""); addUserBubble(question);
        LinearLayout block = new LinearLayout(this); block.setOrientation(LinearLayout.VERTICAL); block.setPadding(dp(2),dp(4),dp(2),dp(10));
        TextView loading = new TextView(this); loading.setText("H33 المحلي"); loading.setTypeface(null,1);
        TextView bubble = createBubble("…", false); block.addView(loading); block.addView(bubble); messagesContainer.addView(block); scrollToBottom();
        final long id = ++generationId; final long[] lastUi = {0}; setBusy(true,"H33 يجهز المرشحين…");
        executor.execute(() -> {
            try {
                CandidateSet set = candidateCoordinator.create(question, full -> {
                    if (id != generationId) return;
                    long now = SystemClock.uptimeMillis(); if (now - lastUi[0] < 35) return; lastUi[0] = now;
                    runOnUiThread(() -> { if (id == generationId) { bubble.setText(full); scrollToBottom(); } });
                });
                runOnUiThread(() -> { if (id == generationId) { renderCandidateComparison(block,set); setBusy(false,"جاهز • اختر أفضل جواب"); scrollToBottom(); } });
            } catch (Exception ex) {
                runOnUiThread(() -> { block.removeAllViews(); block.addView(createBubble("تعذر إنشاء المرشحين: " + safeMessage(ex),false)); setBusy(false,"جاهز"); });
            }
        });
    }

    private void renderCandidateComparison(LinearLayout block, CandidateSet set) {
        block.removeAllViews();
        TextView heading = new TextView(this); heading.setText("قارن الإجابات واختر الأنسب لك"); heading.setTypeface(null,1); heading.setPadding(dp(6),dp(2),dp(6),dp(8)); block.addView(heading);
        CandidateSelectionState state = new CandidateSelectionState(set.turnId); ArrayList<Button> buttons = new ArrayList<>();
        TextView selected = new TextView(this); selected.setPadding(dp(8),dp(8),dp(8),dp(4));
        boolean[] decisionBusy = {false};
        addCandidateCard(block,set,set.local,"H33 المحلي",state,buttons,selected,decisionBusy);
        addCandidateCard(block,set,set.web,"جواب البحث",state,buttons,selected,decisionBusy);
        addCandidateCard(block,set,set.hosted,"Google AI Overview",state,buttons,selected,decisionBusy);
        Button correction = smallButton("كلهم خطأ — سأكتب التصحيح");
        correction.setOnClickListener(v -> showCorrection(set,state,buttons,selected,decisionBusy)); block.addView(correction); block.addView(selected);
    }

    private void addCandidateCard(LinearLayout parent,CandidateSet set,AnswerCandidate candidate,String label,CandidateSelectionState state,List<Button> chooseButtons,TextView selected,boolean[] decisionBusy) {
        LinearLayout card = new LinearLayout(this); card.setOrientation(LinearLayout.VERTICAL); card.setPadding(dp(12),dp(10),dp(12),dp(10));
        GradientDrawable bg = new GradientDrawable(); bg.setColor(Color.rgb(38,38,38)); bg.setCornerRadius(dp(16)); card.setBackground(bg);
        TextView title = new TextView(this); title.setText(label); title.setTypeface(null,1); card.addView(title);
        TextView prov = new TextView(this); prov.setText((candidate.provider.isEmpty()?"غير محدد":candidate.provider) + (candidate.status.isEmpty()?"":" • "+candidate.status)); prov.setTextSize(11); prov.setAlpha(.72f); card.addView(prov);
        if (!candidate.available) { card.addView(createBubble(candidate.status.isEmpty()?"غير متاح":candidate.status,false)); parent.addView(card); return; }
        card.addView(createBubble(candidate.answer,false));
        if (!candidate.sources.isEmpty()) { TextView s=new TextView(this); StringBuilder x=new StringBuilder("المصادر"); int i=1; for(SearchResult r:candidate.sources){x.append("\n[").append(i++).append("] ").append(r.title.isEmpty()?r.host:r.title); if(!r.url.isEmpty())x.append("\n").append(r.url);} s.setText(x.toString()); s.setTextSize(12); s.setAutoLinkMask(Linkify.WEB_URLS); s.setMovementMethod(LinkMovementMethod.getInstance()); card.addView(s); }
        LinearLayout actions=new LinearLayout(this); Button choose=smallButton("اختر هذا الجواب"), copy=smallButton("نسخ"); chooseButtons.add(choose);
        choose.setOnClickListener(v -> { if(decisionBusy[0]||state.isFinalized())return; decisionBusy[0]=true; setButtonsEnabled(chooseButtons,false); executor.execute(()->{try{userDecisionService.select(set,candidate.id); runOnUiThread(()->{decisionBusy[0]=false; state.select(candidate.id); selected.setText("✓ اختيارك محفوظ كذاكرة معتمدة: "+label); lastAssistantAnswer=candidate.answer;});}catch(Exception e){runOnUiThread(()->{decisionBusy[0]=false;setButtonsEnabled(chooseButtons,true);selected.setText("فشل الحفظ: "+safeMessage(e));});}});});
        copy.setOnClickListener(v -> copyText(candidate.answer)); actions.addView(choose); actions.addView(copy); card.addView(actions); parent.addView(card);
    }

    private void showCorrection(CandidateSet set,CandidateSelectionState state,List<Button> buttons,TextView selected,boolean[] decisionBusy) {
        EditText input=new EditText(this); input.setHint("اكتب الجواب الصحيح"); input.setMinLines(3);
        new AlertDialog.Builder(this).setTitle("تصحيح الإجابة").setView(input).setNegativeButton("إلغاء",null).setPositiveButton("حفظ",(d,w)->{String a=input.getText().toString().trim();if(a.isEmpty()||decisionBusy[0])return;decisionBusy[0]=true;executor.execute(()->{try{userDecisionService.correct(set,a);runOnUiThread(()->{decisionBusy[0]=false;state.correct(a);selected.setText("✓ تصحيحك محفوظ كذاكرة معتمدة");lastAssistantAnswer=a;});}catch(Exception e){runOnUiThread(()->selected.setText("فشل التصحيح: "+safeMessage(e)));}});}).show();
    }

    private void setButtonsEnabled(List<Button> bs,boolean enabled){for(Button b:bs)b.setEnabled(enabled);}
    private void copyText(String text){ClipboardManager cm=(ClipboardManager)getSystemService(Context.CLIPBOARD_SERVICE);cm.setPrimaryClip(ClipData.newPlainText("H33",text));Toast.makeText(this,"تم النسخ",Toast.LENGTH_SHORT).show();}
    private void stopGeneration(){if(localInference!=null)localInference.cancel();generationId++;setBusy(false,"تم إيقاف التوليد");}
    private void startNewChat(){generationId++;if(localInference!=null)localInference.cancel();messagesContainer.removeAllViews();addWelcomeMessage();inputBox.setText("");if(engine!=null)executor.execute(()->{try{engine.newConversation();}catch(Exception ignored){}});}
    private void renderConversation(List<CodeModelEngine.ChatTurn> turns){for(CodeModelEngine.ChatTurn t:turns){if("user".equals(t.role))addUserBubble(t.content);else{messagesContainer.addView(createBubble(t.content,false));lastAssistantAnswer=t.content;}}scrollToBottom();}
    private void addUserBubble(String text){LinearLayout w=new LinearLayout(this);w.setGravity(Gravity.END);w.addView(createBubble(text,true));messagesContainer.addView(w);}
    private TextView createBubble(String text,boolean user){TextView b=new TextView(this);b.setText(text);b.setTextSize(16);b.setTextIsSelectable(true);b.setPadding(dp(14),dp(10),dp(14),dp(10));GradientDrawable d=new GradientDrawable();d.setColor(user?Color.rgb(92,64,165):Color.rgb(38,38,38));d.setCornerRadius(dp(18));b.setBackground(d);b.setTextColor(Color.WHITE);return b;}
    private Button smallButton(String text){Button b=new Button(this);b.setText(text);b.setTextSize(12);b.setAllCaps(false);b.setMinWidth(0);b.setMinHeight(0);return b;}
    private void addWelcomeMessage(){LinearLayout w=new LinearLayout(this);w.setOrientation(LinearLayout.VERTICAL);w.setGravity(Gravity.CENTER_HORIZONTAL);w.setPadding(dp(20),dp(28),dp(20),dp(18));TextView t=new TextView(this);t.setText("H33 DeepSeek Coder");t.setTextSize(25);TextView s=new TextView(this);s.setText("DeepSeek-Coder 1.3B • Python • Web Evidence • Google");s.setAlpha(.75f);w.addView(t);w.addView(s);messagesContainer.addView(w);}
    private void setBusy(boolean value,String message){busy=value;status.setText(message);sendButton.setEnabled(!value&&engine!=null);inputBox.setEnabled(!value&&engine!=null);stopButton.setVisibility(value&&engine!=null?View.VISIBLE:View.GONE);plusButton.setEnabled(true);newChatButton.setEnabled(true);}
    private void scrollToBottom(){chatScroll.post(()->chatScroll.fullScroll(ScrollView.FOCUS_DOWN));}
    private int dp(int v){return (int)(v*getResources().getDisplayMetrics().density+.5f);}
    private static String safeMessage(Throwable t){String m=t==null?null:t.getMessage();return m==null||m.trim().isEmpty()?t.getClass().getSimpleName():m;}
    @Override protected void onDestroy(){generationId++;if(localInference!=null)localInference.cancel();executor.shutdownNow();if(engine!=null)engine.close();super.onDestroy();}
}
