package com.musab.aragpt2;

import android.content.Context;
import android.content.res.AssetManager;

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
        public final String role;
        public final String content;

        ChatTurn(String role, String content) {
            this.role = role;
            this.content = content;
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
    private final ArrayList<ChatTurn> conversation = new ArrayList<>();

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
        loadConversation();
    }

    public String generateStream(String question, int maxNewTokens, String webContext,
                                 StreamListener listener) throws Exception {
        String q = question == null ? "" : question.trim();
        if (q.isEmpty()) return "";

        cancelRequested = false;

        boolean hasWeb = webContext != null && !webContext.trim().isEmpty();

        String exact = memory.exactAnswer(q);
        if (!hasWeb && exact != null && !exact.isEmpty()) {
            synchronized (conversation) {
                conversation.add(new ChatTurn("user", q));
                conversation.add(new ChatTurn("assistant", exact));
                trimConversationInMemory();
                saveConversationLocked();
            }
            if (listener != null) listener.onUpdate(exact);
            return exact;
        }

        List<ChatTurn> snapshot;
        synchronized (conversation) {
            snapshot = new ArrayList<>(conversation);
        }

        StringBuilder raw = new StringBuilder();

        try (Sequences encoded = encodeTrimmedPrompt(snapshot, q, webContext)) {
            int[] inputIds = encoded.getSequence(0);
            int totalMaxLength = Math.min(2048, inputIds.length + Math.max(8, maxNewTokens));

            try (GeneratorParams params = new GeneratorParams(model)) {
                params.setSearchOption("max_length", (double) totalMaxLength);
                params.setSearchOption("do_sample", !hasWeb);
                if (!hasWeb) {
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
                        if (listener != null && !visible.isEmpty()) listener.onUpdate(visible);

                        if (containsStopMarker(raw)) break;
                    }
                }
            }
        }

        String answer = cleanup(raw.toString());
        if (!answer.isEmpty()) {
            synchronized (conversation) {
                conversation.add(new ChatTurn("user", q));
                conversation.add(new ChatTurn("assistant", answer));
                trimConversationInMemory();
                saveConversationLocked();
            }
        }
        return answer;
    }

    public void cancelGeneration() {
        cancelRequested = true;
    }

    public void newConversation() throws Exception {
        cancelRequested = true;
        synchronized (conversation) {
            conversation.clear();
            saveConversationLocked();
        }
    }

    public List<ChatTurn> getConversationSnapshot() {
        synchronized (conversation) {
            return Collections.unmodifiableList(new ArrayList<>(conversation));
        }
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

    private void replaceConversationAnswer(String question, String wrongAnswer, String correction) throws Exception {
        String q = question == null ? "" : question.trim();
        String old = wrongAnswer == null ? "" : wrongAnswer.trim();
        String corrected = correction == null ? "" : correction.trim();
        if (q.isEmpty() || corrected.isEmpty()) return;

        synchronized (conversation) {
            for (int i = conversation.size() - 1; i >= 1; i--) {
                ChatTurn assistant = conversation.get(i);
                ChatTurn user = conversation.get(i - 1);
                if (!"assistant".equals(assistant.role) || !"user".equals(user.role)) continue;
                if (!user.content.trim().equals(q)) continue;
                if (!old.isEmpty() && !assistant.content.trim().equals(old)) continue;

                conversation.set(i, new ChatTurn("assistant", corrected));
                saveConversationLocked();
                return;
            }
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
        StringBuilder p = new StringBuilder();
        p.append("<|im_start|>system\n")
                .append(fablePrompt)
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

    private void trimConversationInMemory() {
        while (conversation.size() > MAX_HISTORY_MESSAGES) {
            if (conversation.size() >= 2) {
                conversation.remove(0);
                conversation.remove(0);
            } else {
                conversation.remove(0);
            }
        }
    }

    private void loadConversation() {
        if (!chatFile.exists()) return;
        synchronized (conversation) {
            conversation.clear();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                    new FileInputStream(chatFile), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;
                    JSONObject obj = new JSONObject(line);
                    String role = obj.optString("role", "");
                    String content = obj.optString("content", "");
                    if (("user".equals(role) || "assistant".equals(role)) && !content.isEmpty()) {
                        conversation.add(new ChatTurn(role, content));
                    }
                }
                trimConversationInMemory();
            } catch (Exception ignored) {
                conversation.clear();
            }
        }
    }

    private void saveConversationLocked() throws Exception {
        File tmp = new File(chatFile.getParentFile(), chatFile.getName() + ".tmp");
        try (FileOutputStream out = new FileOutputStream(tmp, false)) {
            for (ChatTurn turn : conversation) {
                JSONObject obj = new JSONObject();
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
