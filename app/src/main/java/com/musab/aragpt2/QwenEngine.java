package com.musab.aragpt2;

import android.content.Context;
import android.content.res.AssetManager;

import org.json.JSONArray;
import org.json.JSONObject;

import ai.onnxruntime.genai.Generator;
import ai.onnxruntime.genai.GeneratorParams;
import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.Sequences;
import ai.onnxruntime.genai.Tokenizer;
import ai.onnxruntime.genai.TokenizerStream;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class QwenEngine implements AutoCloseable {
    public interface StreamListener {
        void onUpdate(String fullText);
    }

    public static final class ChatTurn {
        public final String turnId;
        public final String role;
        public final String content;

        ChatTurn(String role, String content) {
            this("", role, content);
        }

        ChatTurn(String turnId, String role, String content) {
            this.turnId = turnId == null ? "" : turnId;
            this.role = role == null ? "" : role;
            this.content = content == null ? "" : content;
        }
    }

    private static final String FALLBACK_SYSTEM_PROMPT =
            "أنت H33، مساعد محلي مبني على Qwen2.5. أجب مباشرة وبدقة، لا تخمّن، " +
            "واستخدم نتائج الويب المعطاة لك عندما تكون موجودة.";

    private static final String MODEL_ASSET_DIR = "model";
    private static final String MODEL_LOCAL_DIR = "qwen2_5_0_5b_int4";
    private static final String CHAT_FILE = "current_chat.jsonl";
    private static final int MAX_HISTORY_MESSAGES = 16;
    private static final int MAX_PROMPT_TOKENS = 1750;

    private final Context context;
    private final CorrectionMemory memory;
    private final Model model;
    private final Tokenizer tokenizer;
    private final File chatFile;
    private final String fablePrompt;
    private final String chatTemplate;
    private final ConversationHistory conversation =
            new ConversationHistory(MAX_HISTORY_MESSAGES);

    private volatile boolean cancelRequested = false;

    public QwenEngine(Context context) throws Exception {
        this.context = context.getApplicationContext();
        this.memory = new CorrectionMemory(this.context);
        this.chatFile = new File(this.context.getFilesDir(), CHAT_FILE);
        this.fablePrompt = readTextAsset(this.context.getAssets(), "fable_for_qwen_v1.txt",
                FALLBACK_SYSTEM_PROMPT);

        File modelDir = new File(this.context.getFilesDir(), MODEL_LOCAL_DIR);
        prepareModelDirectory(this.context.getAssets(), MODEL_ASSET_DIR, modelDir);

        this.model = new Model(modelDir.getAbsolutePath());
        this.tokenizer = new Tokenizer(model);
        this.chatTemplate = readTextAsset(
                this.context.getAssets(),
                "model/chat_template.jinja",
                ""
        );
        loadConversation();
    }

    public String generateStream(String question, int maxNewTokens, String webContext,
                                 StreamListener listener) throws Exception {
        String answer = generateCandidate(
                question,
                maxNewTokens,
                webContext,
                webContext != null && !webContext.trim().isEmpty(),
                listener
        );

        String q = question == null ? "" : question.trim();
        if (!q.isEmpty() && !answer.isEmpty()) {
            commitCanonicalTurn(java.util.UUID.randomUUID().toString(), q, answer);
        }
        return answer;
    }

    public String generateCandidate(String question, int maxNewTokens, String evidenceContext,
                                    boolean evidenceOnly, StreamListener listener) throws Exception {
        String q = question == null ? "" : question.trim();
        if (q.isEmpty()) return "";

        cancelRequested = false;

        String fittedEvidence =
                evidenceContext == null ? "" : evidenceContext.trim();
        boolean hasEvidence = !fittedEvidence.isEmpty();

        if (evidenceOnly && hasEvidence) {
            fittedEvidence =
                    "EVIDENCE_ONLY: أجب فقط مما تدعمه النتائج التالية. " +
                    "إذا لم تكفِ النتائج، قل بوضوح إن الأدلة غير كافية. " +
                    "لا تستخدم معرفتك الداخلية لتعويض نقص الأدلة.\n" +
                    fittedEvidence;
        }

        String exact = memory.exactAnswer(q);
        if (!hasEvidence && exact != null && !exact.isEmpty()) {
            if (listener != null) listener.onUpdate(exact);
            return exact;
        }

        List<ChatTurn> snapshot = getConversationSnapshot();
        StringBuilder raw = new StringBuilder();

        try (Sequences encoded = encodeTrimmedPrompt(snapshot, q, fittedEvidence)) {
            int[] inputIds = encoded.getSequence(0);
            int totalMaxLength = Math.min(2048, inputIds.length + Math.max(8, maxNewTokens));

            try (GeneratorParams params = new GeneratorParams(model)) {
                params.setSearchOption("max_length", (double) totalMaxLength);
                params.setSearchOption("do_sample", !hasEvidence);
                if (!hasEvidence) {
                    params.setSearchOption("temperature", 0.70);
                    params.setSearchOption("top_k", 20.0);
                    params.setSearchOption("top_p", 0.80);
                }
                params.setSearchOption("repetition_penalty", 1.10);

                try (Generator generator = new Generator(model, params);
                     TokenizerStream stream = tokenizer.createStream()) {
                    generator.appendTokenSequences(encoded);

                    while (!generator.isDone() && !cancelRequested) {
                        generator.generateNextToken();
                        int token = generator.getLastTokenInSequence(0);
                        raw.append(stream.decode(token));

                        String visible = cleanup(raw.toString());
                        if (listener != null && !visible.isEmpty()) {
                            listener.onUpdate(visible);
                        }

                        if (containsStopMarker(raw)) break;
                    }
                }
            }
        }

        return cleanup(raw.toString());
    }

    public void cancelGeneration() {
        cancelRequested = true;
    }

    public void newConversation() throws Exception {
        cancelRequested = true;
        conversation.clear();
        saveConversationLocked();
    }

    public List<ChatTurn> getConversationSnapshot() {
        ArrayList<ChatTurn> out = new ArrayList<>();
        for (ConversationHistory.Turn turn : conversation.snapshot()) {
            out.add(new ChatTurn(turn.turnId, turn.role, turn.content));
        }
        return Collections.unmodifiableList(out);
    }

    public void commitCanonicalTurn(String turnId, String question, String answer)
            throws Exception {
        String q = question == null ? "" : question.trim();
        String a = answer == null ? "" : answer.trim();
        if (q.isEmpty() || a.isEmpty()) return;

        String id = turnId == null ? "" : turnId.trim();
        if (id.isEmpty()) id = java.util.UUID.randomUUID().toString();

        conversation.appendTurn(id, q, a);
        saveConversationLocked();
    }

    public boolean replaceCanonicalAnswer(String turnId, String answer) throws Exception {
        boolean replaced = conversation.replaceAnswer(turnId, answer);
        if (replaced) saveConversationLocked();
        return replaced;
    }

    public String rememberCorrect(String question, String answer) throws Exception {
        memory.remember(question, answer, false);
        return "تم حفظ الإجابة الصحيحة";
    }

    public String learnCorrection(String question, String wrongAnswer, String correction) throws Exception {
        memory.remember(question, correction, true);
        replaceConversationAnswer(question, wrongAnswer, correction);
        return "تم حفظ التصحيح";
    }

    public int memoryCount() {
        return memory.count();
    }

    private void replaceConversationAnswer(
            String question, String wrongAnswer, String correction) throws Exception {
        String q = question == null ? "" : question.trim();
        String old = wrongAnswer == null ? "" : wrongAnswer.trim();
        String corrected = correction == null ? "" : correction.trim();
        if (q.isEmpty() || corrected.isEmpty()) return;

        if (conversation.replaceLatestAnswer(q, old, corrected)) {
            saveConversationLocked();
        }
    }

    private Sequences encodeTrimmedPrompt(List<ChatTurn> snapshot,
                                           String question, String webContext)
            throws Exception {
        ArrayList<ChatTurn> recent = new ArrayList<>(snapshot);

        // Keep the most recent two full user/assistant exchanges whenever possible.
        final int minRecentMessages = Math.min(4, recent.size());

        String fittedWeb = webContext == null ? "" : webContext.trim();

        while (true) {
            String prompt = buildPrompt(recent, question, fittedWeb);
            Sequences encoded = tokenizer.encode(prompt);
            if (encoded.getSequence(0).length <= MAX_PROMPT_TOKENS) {
                return encoded;
            }
            encoded.close();

            // First sacrifice old history, never the newest two exchanges.
            if (recent.size() > minRecentMessages) {
                int remove = Math.min(2, recent.size() - minRecentMessages);
                for (int i = 0; i < remove; i++) recent.remove(0);
                continue;
            }

            // Then shrink web evidence before sacrificing recent conversation context.
            if (!fittedWeb.isEmpty()) {
                if (fittedWeb.length() > 2200) {
                    fittedWeb = fittedWeb.substring(0, 2200);
                    continue;
                }
                if (fittedWeb.length() > 1200) {
                    fittedWeb = fittedWeb.substring(0, 1200);
                    continue;
                }
                if (fittedWeb.length() > 600) {
                    fittedWeb = fittedWeb.substring(0, 600);
                    continue;
                }
                fittedWeb = "";
                continue;
            }

            // As a last resort keep only the immediately previous exchange.
            if (recent.size() > 2) {
                recent = new ArrayList<>(recent.subList(recent.size() - 2, recent.size()));
                continue;
            }

            // Current question must always be preserved.
            return tokenizer.encode(buildPrompt(recent, question, ""));
        }
    }

    private String buildPrompt(List<ChatTurn> history, String question, String webContext) {
        if (chatTemplate != null && !chatTemplate.trim().isEmpty()) {
            try {
                JSONArray messages = new JSONArray();

                StringBuilder system = new StringBuilder(fablePrompt);

                String sessionContext = buildRecentSessionContext(history);
                if (!sessionContext.isEmpty()) {
                    system.append("\n\n").append(sessionContext);
                }

                if (webContext != null && !webContext.trim().isEmpty()) {
                    system.append("\n\nWEB_RESULTS حديثة. استخدم فقط ما يفيد السؤال، ")
                            .append("ولا تختلق مصادر. عند الاستشهاد استخدم [رقم النتيجة].\n")
                            .append(webContext.trim());
                }
                addMessage(messages, "system", system.toString());

                if (webContext == null || webContext.trim().isEmpty()) {
                    List<CorrectionMemory.Entry> examples = memory.bestExamples(question, 1);
                    for (CorrectionMemory.Entry e : examples) {
                        addMessage(messages, "user", e.question);
                        addMessage(messages, "assistant", e.answer);
                    }
                }

                int start = Math.max(0, history.size() - MAX_HISTORY_MESSAGES);
                for (int i = start; i < history.size(); i++) {
                    ChatTurn turn = history.get(i);
                    if (!"user".equals(turn.role) && !"assistant".equals(turn.role)) continue;
                    addMessage(messages, turn.role, turn.content);
                }

                addMessage(messages, "user", question);

                return tokenizer.applyChatTemplate(
                        chatTemplate,
                        messages.toString(),
                        null,
                        true
                );
            } catch (Exception ignored) {
                // Fall through to the conservative manual Qwen ChatML formatter.
            }
        }

        return buildPromptFallback(history, question, webContext);
    }

    private String buildPromptFallback(List<ChatTurn> history, String question, String webContext) {
        StringBuilder p = new StringBuilder();

        String fallbackSystem = fablePrompt;
        String sessionContext = buildRecentSessionContext(history);
        if (!sessionContext.isEmpty()) {
            fallbackSystem += "\n\n" + sessionContext;
        }

        p.append("<|im_start|>system\n")
                .append(fallbackSystem)
                .append("<|im_end|>\n");

        if (webContext != null && !webContext.trim().isEmpty()) {
            p.append("<|im_start|>system\n")
                    .append("WEB_RESULTS حديثة. استخدم فقط ما يفيد السؤال، ولا تختلق مصادر. ")
                    .append("عند الاستشهاد استخدم [رقم النتيجة].\n")
                    .append(webContext.trim())
                    .append("<|im_end|>\n");
        }

        if (webContext == null || webContext.trim().isEmpty()) {
            List<CorrectionMemory.Entry> examples = memory.bestExamples(question, 1);
            for (CorrectionMemory.Entry e : examples) {
                p.append("<|im_start|>user\n")
                        .append(e.question)
                        .append("<|im_end|>\n")
                        .append("<|im_start|>assistant\n")
                        .append(e.answer)
                        .append("<|im_end|>\n");
            }
        }

        int start = Math.max(0, history.size() - MAX_HISTORY_MESSAGES);
        for (int i = start; i < history.size(); i++) {
            ChatTurn turn = history.get(i);
            if (!"user".equals(turn.role) && !"assistant".equals(turn.role)) continue;
            p.append("<|im_start|>")
                    .append(turn.role)
                    .append("\n")
                    .append(turn.content)
                    .append("<|im_end|>\n");
        }

        p.append("<|im_start|>user\n")
                .append(question)
                .append("<|im_end|>\n")
                .append("<|im_start|>assistant\n");

        return p.toString();
    }

    private static void addMessage(JSONArray messages, String role, String content) throws Exception {
        JSONObject obj = new JSONObject();
        obj.put("role", role);
        obj.put("content", content == null ? "" : content);
        messages.put(obj);
    }

    private String buildRecentSessionContext(List<ChatTurn> history) {
        if (history == null || history.isEmpty()) return "";

        int start = Math.max(0, history.size() - 4);
        StringBuilder out = new StringBuilder();
        out.append("SESSION_CONTEXT من نفس المحادثة. هذا السياق موثوق. ")
                .append("إذا أشار المستخدم إلى شيء قاله أو قلته قبل قليل، استخدمه مباشرة ولا تدّعِ أنه غير موجود.\n");

        for (int i = start; i < history.size(); i++) {
            ChatTurn turn = history.get(i);
            if (!"user".equals(turn.role) && !"assistant".equals(turn.role)) continue;

            String text = turn.content == null ? "" : turn.content.trim();
            if (text.length() > 360) text = text.substring(0, 360) + "…";

            out.append("user".equals(turn.role) ? "المستخدم سابقًا: " : "المساعد سابقًا: ")
                    .append(text)
                    .append("\n");
        }

        return out.toString().trim();
    }

    private void loadConversation() {
        if (!chatFile.exists()) return;

        ArrayList<ConversationHistory.Turn> loaded = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(chatFile), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                JSONObject obj = new JSONObject(line);
                String turnId = obj.optString("turn_id", "");
                String role = obj.optString("role", "");
                String content = obj.optString("content", "");

                if (("user".equals(role) || "assistant".equals(role)) &&
                        !content.trim().isEmpty()) {
                    loaded.add(new ConversationHistory.Turn(turnId, role, content));
                }
            }
            conversation.replaceAll(loaded);
        } catch (Exception ignored) {
            conversation.clear();
        }
    }

    private void saveConversationLocked() throws Exception {
        File tmp = new File(chatFile.getParentFile(), chatFile.getName() + ".tmp");
        try (FileOutputStream out = new FileOutputStream(tmp, false)) {
            for (ConversationHistory.Turn turn : conversation.snapshot()) {
                JSONObject obj = new JSONObject();
                obj.put("turn_id", turn.turnId);
                obj.put("role", turn.role);
                obj.put("content", turn.content);
                out.write((obj.toString() + "\n").getBytes(StandardCharsets.UTF_8));
            }
            out.getFD().sync();
        }

        if (chatFile.exists() && !chatFile.delete()) {
            throw new IllegalStateException("تعذر تحديث سجل المحادثة");
        }
        if (!tmp.renameTo(chatFile)) {
            throw new IllegalStateException("تعذر حفظ سجل المحادثة");
        }
    }

    private static boolean containsStopMarker(StringBuilder raw) {
        String s = raw.toString();
        return s.contains("<|im_end|>") || s.contains("<|endoftext|>");
    }

    private static String cleanup(String text) {
        if (text == null) return "";
        String out = text;
        int marker = out.indexOf("<|im_end|>");
        if (marker >= 0) out = out.substring(0, marker);
        marker = out.indexOf("<|endoftext|>");
        if (marker >= 0) out = out.substring(0, marker);
        return out.trim();
    }

    private static String readTextAsset(AssetManager assets, String name, String fallback) {
        try (InputStream in = assets.open(name);
             BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            StringBuilder out = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                out.append(line).append('\n');
                if (out.length() > 6000) break;
            }
            String value = out.toString().trim();
            return value.isEmpty() ? fallback : value;
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private static void prepareModelDirectory(AssetManager assets, String assetPath, File target)
            throws Exception {
        File marker = new File(target, ".ready_v1");
        if (marker.exists() && new File(target, "genai_config.json").exists()) return;

        deleteRecursive(target);
        if (!target.mkdirs() && !target.isDirectory()) {
            throw new IllegalStateException("تعذر إنشاء مجلد النموذج");
        }

        copyAssetTree(assets, assetPath, target);
        if (!new File(target, "genai_config.json").exists()) {
            throw new IllegalStateException("genai_config.json غير موجود داخل النموذج");
        }
        if (!marker.createNewFile() && !marker.exists()) {
            throw new IllegalStateException("تعذر تثبيت علامة جاهزية النموذج");
        }
    }

    private static void copyAssetTree(AssetManager assets, String assetPath, File target)
            throws Exception {
        String[] children = assets.list(assetPath);
        if (children == null) {
            throw new IllegalStateException("تعذر قراءة أصول النموذج: " + assetPath);
        }

        if (children.length == 0) {
            File parent = target.getParentFile();
            if (parent != null && !parent.exists() && !parent.mkdirs()) {
                throw new IllegalStateException("تعذر إنشاء: " + parent);
            }
            try (InputStream in = assets.open(assetPath);
                 FileOutputStream out = new FileOutputStream(target)) {
                byte[] buffer = new byte[1024 * 1024];
                int n;
                while ((n = in.read(buffer)) != -1) out.write(buffer, 0, n);
                out.getFD().sync();
            }
            return;
        }

        if (!target.exists() && !target.mkdirs()) {
            throw new IllegalStateException("تعذر إنشاء: " + target);
        }
        for (String child : children) {
            copyAssetTree(assets, assetPath + "/" + child, new File(target, child));
        }
    }

    private static void deleteRecursive(File file) {
        if (!file.exists()) return;
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) for (File child : children) deleteRecursive(child);
        }
        file.delete();
    }

    @Override
    public void close() {
        cancelRequested = true;
        try { tokenizer.close(); } catch (Exception ignored) {}
        try { model.close(); } catch (Exception ignored) {}
    }
}
