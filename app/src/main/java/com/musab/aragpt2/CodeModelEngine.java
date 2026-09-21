package com.musab.aragpt2;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.SystemClock;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * DeepSeek-Coder conversation layer backed by an external GGUF model.
 * No ONNX Runtime, tokenizer regex engine, or model asset is used here.
 */
public final class CodeModelEngine implements AutoCloseable {
    public static final class ChatTurn {
        public final String turnId;
        public final String role;
        public final String content;

        ChatTurn(String turnId, String role, String content) {
            this.turnId = turnId == null ? "" : turnId;
            this.role = role == null ? "" : role;
            this.content = content == null ? "" : content;
        }
    }

    private static final String CHAT_FILE = "current_chat.jsonl";
    private static final String MODEL_DIR = "models";
    private static final String MODEL_FILE = "deepseek-coder-1.3b-instruct.Q4_K_M.gguf";
    private static final int MAX_HISTORY_MESSAGES = 8;
    private static final int MAX_PROMPT_CHARS = 18000;
    private static final long MODEL_WAIT_MS = 15L * 60L * 1000L;

    private static final String SYSTEM_PROMPT =
            "You are H33 Python Coder, a local programming assistant. " +
            "Answer Python programming questions only. Prefer correct, runnable Python 3 code " +
            "with concise explanations. Do not invent package APIs. Answer in English. " +
            "Never claim web access.\n";

    private final File modelFile;
    private final ConversationFileStore chatStore;
    private final ConversationHistory conversation =
            new ConversationHistory(MAX_HISTORY_MESSAGES);
    private NativeLlamaEngine nativeEngine;
    private volatile boolean cancelRequested;

    public CodeModelEngine(Context context) throws Exception {
        Context app = context.getApplicationContext();
        File modelDir = new File(app.getFilesDir(), MODEL_DIR);
        if (!modelDir.exists() && !modelDir.mkdirs()) {
            throw new IllegalStateException("Cannot create model directory");
        }
        modelFile = new File(modelDir, MODEL_FILE);
        chatStore = new ConversationFileStore(
                new File(app.getFilesDir(), CHAT_FILE));
        loadConversation();

        if (!modelFile.isFile()) {
            launchModelPicker(context);
            waitForImportedModel();
        }

        if (!modelFile.isFile()) {
            throw new IllegalStateException("GGUF model was not imported");
        }
        nativeEngine = new NativeLlamaEngine(modelFile);
    }

    public String generateCandidate(
            String question,
            int maxNewTokens,
            LocalInferenceEngine.StreamListener listener) throws Exception {
        String q = clean(question);
        if (q.isEmpty()) return "";
        cancelRequested = false;

        String prompt = buildPrompt(q);
        String answer = nativeEngine.generate(prompt, maxNewTokens);
        if (cancelRequested) return answer == null ? "" : answer.trim();

        answer = cleanup(answer);
        if (listener != null && !answer.isEmpty()) listener.onUpdate(answer);
        return answer;
    }

    public String generate(
            String question,
            int maxNewTokens,
            String evidenceContext,
            boolean evidenceOnly,
            LocalInferenceEngine.StreamListener listener) throws Exception {
        return generateCandidate(question, maxNewTokens, listener);
    }

    public void cancelGeneration() {
        cancelRequested = true;
        if (nativeEngine != null) nativeEngine.cancel();
    }

    public void newConversation() throws Exception {
        cancelGeneration();
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
        String q = clean(question);
        String a = clean(answer);
        if (q.isEmpty() || a.isEmpty()) return;
        String id = clean(turnId);
        if (id.isEmpty()) id = UUID.randomUUID().toString();
        conversation.appendTurn(id, q, a);
        saveConversationLocked();
    }

    public boolean replaceCanonicalAnswer(String turnId, String answer)
            throws Exception {
        boolean replaced = conversation.replaceAnswer(turnId, answer);
        if (replaced) saveConversationLocked();
        return replaced;
    }

    public String getCanonicalAnswer(String turnId) {
        return conversation.answerForTurn(turnId);
    }

    public boolean removeCanonicalTurn(String turnId) throws Exception {
        boolean removed = conversation.removeTurn(turnId);
        if (removed) saveConversationLocked();
        return removed;
    }

    private String buildPrompt(String question) {
        StringBuilder out = new StringBuilder();
        out.append("<｜begin▁of▁sentence｜>");
        out.append(SYSTEM_PROMPT);

        List<ConversationHistory.Turn> turns = conversation.snapshot();
        int start = Math.max(0, turns.size() - MAX_HISTORY_MESSAGES);
        for (int i = start; i < turns.size(); i++) {
            ConversationHistory.Turn turn = turns.get(i);
            if ("user".equals(turn.role)) {
                out.append("### Instruction:\n").append(turn.content).append("\n");
            } else if ("assistant".equals(turn.role)) {
                out.append("### Response:\n")
                        .append(turn.content)
                        .append("\n<|EOT|>\n");
            }
        }

        out.append("### Instruction:\n").append(question).append("\n### Response:");
        String prompt = out.toString();
        if (prompt.length() <= MAX_PROMPT_CHARS) return prompt;

        int keepFrom = Math.max(0, prompt.length() - MAX_PROMPT_CHARS);
        return SYSTEM_PROMPT + "\n### Instruction:\n" +
                prompt.substring(keepFrom) + "\n### Response:";
    }

    private void waitForImportedModel() throws Exception {
        long deadline = SystemClock.uptimeMillis() + MODEL_WAIT_MS;
        while (SystemClock.uptimeMillis() < deadline) {
            if (modelFile.isFile() && modelFile.length() > 16) return;
            Thread.sleep(250L);
        }
    }

    private void launchModelPicker(Context context) {
        if (!(context instanceof Activity)) return;
        Activity activity = (Activity) context;
        activity.runOnUiThread(() -> {
            try {
                Intent intent = new Intent(activity, ModelImportActivity.class);
                activity.startActivity(intent);
            } catch (Exception ignored) {
            }
        });
    }

    private void loadConversation() {
        conversation.replaceAll(chatStore.load());
    }

    private void saveConversationLocked() throws Exception {
        chatStore.save(conversation.snapshot());
    }

    private static String cleanup(String text) {
        String out = text == null ? "" : text;
        int marker = out.indexOf("<|EOT|>");
        if (marker >= 0) out = out.substring(0, marker);
        marker = out.indexOf("<｜end▁of▁sentence｜>");
        if (marker >= 0) out = out.substring(0, marker);
        return out.trim();
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }

    @Override
    public void close() {
        cancelGeneration();
        if (nativeEngine != null) {
            nativeEngine.close();
            nativeEngine = null;
        }
    }
}
