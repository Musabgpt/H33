package com.musab.aragpt2;

import android.content.Context;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/** DeepSeek-Coder engine backed by llama.cpp. GGUF stays outside the APK. */
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
    private static final int MAX_HISTORY_MESSAGES = 8;
    private static final int CONTEXT_SIZE = 4096;
    private static final String SYSTEM_PROMPT =
            "You are H33 Python Coder, a local programming assistant. "
            + "Answer Python programming questions. Prefer correct runnable Python 3 code "
            + "with concise explanations. Do not invent package APIs. Never claim web access.";

    private final File chatFile;
    private final ConversationFileStore chatStore;
    private final ConversationHistory conversation = new ConversationHistory(MAX_HISTORY_MESSAGES);
    private final NativeCodeModel nativeModel;

    public CodeModelEngine(Context context) throws Exception {
        Context app = context.getApplicationContext();
        chatFile = new File(app.getFilesDir(), CHAT_FILE);
        chatStore = new ConversationFileStore(chatFile);
        if (!ModelManager.isInstalled(app)) {
            throw new IllegalStateException("لم يتم تثبيت نموذج GGUF.");
        }
        nativeModel = new NativeCodeModel();
        nativeModel.load(ModelManager.modelFile(app), CONTEXT_SIZE);
        loadConversation();
    }

    public String generateCandidate(String question, int maxNewTokens,
                                    LocalInferenceEngine.StreamListener listener) throws Exception {
        String q = clean(question);
        if (q.isEmpty()) return "";
        String prompt = buildPrompt(getConversationSnapshot(), q);
        String answer = cleanup(nativeModel.generate(prompt, Math.max(16, maxNewTokens), CONTEXT_SIZE));
        if (listener != null && !answer.isEmpty()) listener.onUpdate(answer);
        return answer;
    }

    public void cancelGeneration() {}

    public void newConversation() throws Exception {
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

    public void commitCanonicalTurn(String turnId, String question, String answer) throws Exception {
        String q = clean(question), a = clean(answer);
        if (q.isEmpty() || a.isEmpty()) return;
        String id = clean(turnId);
        if (id.isEmpty()) id = UUID.randomUUID().toString();
        conversation.appendTurn(id, q, a);
        saveConversationLocked();
    }

    public boolean replaceCanonicalAnswer(String turnId, String answer) throws Exception {
        boolean replaced = conversation.replaceAnswer(turnId, answer);
        if (replaced) saveConversationLocked();
        return replaced;
    }

    public String getCanonicalAnswer(String turnId) { return conversation.answerForTurn(turnId); }

    public boolean removeCanonicalTurn(String turnId) throws Exception {
        boolean removed = conversation.removeTurn(turnId);
        if (removed) saveConversationLocked();
        return removed;
    }

    private String buildPrompt(List<ChatTurn> history, String question) {
        StringBuilder out = new StringBuilder(4096);
        out.append("### System:\n").append(SYSTEM_PROMPT).append("\n\n");
        int start = Math.max(0, history.size() - MAX_HISTORY_MESSAGES);
        for (int i = start; i < history.size(); i++) {
            ChatTurn turn = history.get(i);
            if ("user".equals(turn.role)) {
                out.append("### Instruction:\n").append(turn.content).append("\n");
            } else if ("assistant".equals(turn.role)) {
                out.append("### Response:\n").append(turn.content).append("\n<|EOT|>\n");
            }
        }
        out.append("### Instruction:\n").append(question).append("\n### Response:\n");
        return out.toString();
    }

    private void loadConversation() { conversation.replaceAll(chatStore.load()); }
    private void saveConversationLocked() throws Exception { chatStore.save(conversation.snapshot()); }

    private static String cleanup(String text) {
        String out = text == null ? "" : text;
        int marker = out.indexOf("<|EOT|>");
        if (marker >= 0) out = out.substring(0, marker);
        marker = out.indexOf("<｜end▁of▁sentence｜>");
        if (marker >= 0) out = out.substring(0, marker);
        return out.trim();
    }

    private static String clean(String value) { return value == null ? "" : value.trim(); }

    @Override public void close() { nativeModel.close(); }
}
