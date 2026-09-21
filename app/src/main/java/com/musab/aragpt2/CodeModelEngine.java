package com.musab.aragpt2;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.StatFs;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** DeepSeek Coder GGUF engine tuned for 4 GB Android devices. */
public final class CodeModelEngine implements AutoCloseable {
    public interface StreamListener { void onUpdate(String text); }

    public static final class ChatTurn {
        public final String turnId, role, content;
        ChatTurn(String id, String role, String content) {
            this.turnId = safe(id); this.role = safe(role); this.content = safe(content);
        }
    }

    private static final String MODEL_ASSET = "model/deepseek-coder-1.3b-q4_k_m.gguf";
    private static final String MODEL_FILE = "deepseek-coder-1.3b-q4_k_m.gguf";
    private static final String CHAT_FILE = "current_chat.jsonl";
    private static final int MAX_HISTORY_MESSAGES = 4;
    private static final int MAX_PROMPT_CHARS = 7000;
    private static final long MIN_MODEL_BYTES = 450L * 1024L * 1024L;
    private static final Pattern TOOL_CALL = Pattern.compile(
            "<tool_call\\s+name=\\\"([a-zA-Z0-9_.-]{1,48})\\\">([\\s\\S]*?)</tool_call>");
    private static final String SYSTEM =
            "You are H33 DeepSeek Coder, an offline Python programming assistant. " +
            "Write correct runnable Python 3. Reply in the user's language. " +
            "Be concise. Never claim internet access.";

    private final Context context;
    private final ToolRegistry tools;
    private final ConversationHistory conversation = new ConversationHistory(MAX_HISTORY_MESSAGES);
    private final ConversationFileStore store;
    private boolean closed;

    public CodeModelEngine(Context context, ToolRegistry tools) throws Exception {
        this.context = context.getApplicationContext();
        this.tools = tools == null ? new ToolRegistry() : tools;
        this.store = new ConversationFileStore(new File(this.context.getFilesDir(), CHAT_FILE));
        conversation.replaceAll(store.load());

        File model = prepareModelFile();
        LlamaNative.load();
        String error = LlamaNative.open(model.getAbsolutePath(), 512, 4);
        if (error != null && !error.isEmpty()) throw new IllegalStateException(error);
    }

    public String generateCandidate(String question, int maxNewTokens,
            StreamListener listener) throws Exception {
        ensureOpen();
        String q = safe(question).trim();
        if (q.isEmpty()) return "";
        StringBuilder streamed = new StringBuilder();
        String answer = LlamaNative.generate(
                buildPrompt(q),
                Math.min(160, Math.max(16, maxNewTokens)),
                token -> {
                    streamed.append(token);
                    String visible = cleanup(streamed.toString());
                    if (listener != null && !visible.isEmpty()) listener.onUpdate(visible);
                });
        if (answer == null) answer = "";
        answer = cleanup(answer);
        if (listener != null && streamed.length() == 0 && !answer.isEmpty()) listener.onUpdate(answer);

        Matcher call = TOOL_CALL.matcher(answer);
        if (!call.find()) return answer;
        LocalTool tool = tools.get(call.group(1));
        if (tool == null) return answer;
        String result;
        try { result = tool.execute(call.group(2).trim()); }
        catch (Exception e) { result = "Tool error: " + e.getMessage(); }
        StringBuilder finalStream = new StringBuilder();
        String finalAnswer = cleanup(LlamaNative.generate(
                buildPrompt(q + "\n\nTool " + call.group(1) + " returned:\n" + result
                        + "\nGive the final answer without tool markup."), 160,
                token -> {
                    finalStream.append(token);
                    String visible = cleanup(finalStream.toString());
                    if (listener != null && !visible.isEmpty()) listener.onUpdate(visible);
                }));
        if (listener != null && finalStream.length() == 0 && !finalAnswer.isEmpty()) listener.onUpdate(finalAnswer);
        return finalAnswer;
    }

    public void cancelGeneration() { LlamaNative.cancel(); }

    public synchronized void newConversation() throws Exception {
        conversation.clear(); store.save(conversation.snapshot());
    }

    public synchronized List<ChatTurn> getConversationSnapshot() {
        ArrayList<ChatTurn> out = new ArrayList<>();
        for (ConversationHistory.Turn t : conversation.snapshot()) {
            out.add(new ChatTurn(t.turnId, t.role, t.content));
        }
        return Collections.unmodifiableList(out);
    }

    public synchronized void commitCanonicalTurn(String id, String q, String a) throws Exception {
        if (safe(q).trim().isEmpty() || safe(a).trim().isEmpty()) return;
        conversation.appendTurn(safe(id).isEmpty() ? UUID.randomUUID().toString() : id, q, a);
        store.save(conversation.snapshot());
    }

    private synchronized String buildPrompt(String question) {
        StringBuilder p = new StringBuilder(SYSTEM).append(tools.promptDescription()).append("\n\n");
        for (ConversationHistory.Turn t : conversation.snapshot()) {
            if ("user".equals(t.role)) p.append("### Instruction:\n").append(t.content).append('\n');
            else if ("assistant".equals(t.role)) p.append("### Response:\n").append(t.content).append("\n<|EOT|>\n");
        }
        p.append("### Instruction:\n").append(question).append("\n### Response:\n");
        if (p.length() > MAX_PROMPT_CHARS) return p.substring(p.length() - MAX_PROMPT_CHARS);
        return p.toString();
    }

    private File prepareModelFile() throws Exception {
        File target = new File(context.getFilesDir(), MODEL_FILE);
        if (isValidModel(target)) return target; // Small updates reuse the installed model.

        AssetManager assets = context.getAssets();
        long free = new StatFs(context.getFilesDir().getAbsolutePath()).getAvailableBytes();
        if (free < 950L * 1024L * 1024L) {
            throw new IllegalStateException("تحتاج مساحة فارغة تقارب 1GB لتحضير النموذج");
        }
        File temp = new File(context.getFilesDir(), MODEL_FILE + ".part");
        if (temp.exists() && !temp.delete()) throw new IllegalStateException("تعذر تنظيف ملف النموذج المؤقت");
        try (InputStream in = assets.open(MODEL_ASSET, AssetManager.ACCESS_STREAMING);
             FileOutputStream out = new FileOutputStream(temp)) {
            byte[] buffer = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buffer)) != -1) out.write(buffer, 0, n);
            out.getFD().sync();
        } catch (Exception e) {
            temp.delete();
            throw new IllegalStateException("هذه نسخة تحديث صغيرة ولا يوجد نموذج محفوظ. ثبّت النسخة الكاملة أولًا.", e);
        }
        if (!isValidModel(temp)) { temp.delete(); throw new IllegalStateException("ملف GGUF غير مكتمل"); }
        if (target.exists()) target.delete();
        if (!temp.renameTo(target)) throw new IllegalStateException("تعذر تثبيت ملف النموذج");
        return target;
    }

    private static boolean isValidModel(File file) {
        if (!file.isFile() || file.length() < MIN_MODEL_BYTES) return false;
        try (InputStream in = new java.io.FileInputStream(file)) {
            return in.read() == 'G' && in.read() == 'G' && in.read() == 'U' && in.read() == 'F';
        } catch (Exception ignored) { return false; }
    }

    private static String cleanup(String text) {
        String out = safe(text);
        int eot = out.indexOf("<|EOT|>"); if (eot >= 0) out = out.substring(0, eot);
        int next = out.indexOf("### Instruction:"); if (next >= 0) out = out.substring(0, next);
        return out.trim();
    }
    private static String safe(String value) { return value == null ? "" : value; }
    private void ensureOpen() { if (closed) throw new IllegalStateException("المحرك مغلق"); }

    @Override public synchronized void close() {
        if (!closed) { closed = true; LlamaNative.close(); }
    }
}
