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
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class CodeModelEngine implements AutoCloseable {
    public interface StreamListener { void onUpdate(String text); }
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

    private static final String MODEL_ASSET_DIR = "model";
    private static final String MODEL_LOCAL_DIR = "deepseek_coder_1_3b_int4";
    private static final String CHAT_FILE = "current_chat.jsonl";
    private static final int MAX_HISTORY_MESSAGES = 8;
    private static final int MAX_PROMPT_TOKENS = 2600;
    private static final String SYSTEM_PROMPT =
            "You are H33 Python Coder, a local programming assistant. " +
            "Answer Python programming questions only. " +
            "Prefer correct, runnable Python 3 code with concise explanations. " +
            "Do not invent package APIs. If the request is not about Python programming, " +
            "say that the local Python model is not intended for that task. " +
            "Reply in the same language as the user. Never claim web access.";
    private static final Pattern TOOL_CALL = Pattern.compile(
            "<tool_call\\s+name=\\\"([a-zA-Z0-9_.-]{1,48})\\\">([\\s\\S]*?)</tool_call>");

    private final Context context;
    private final Model model;
    private final Tokenizer tokenizer;
    private final File chatFile;
    private final ConversationFileStore chatStore;
    private final ToolRegistry tools;
    private final String chatTemplate;
    private final ConversationHistory conversation =
            new ConversationHistory(MAX_HISTORY_MESSAGES);

    private volatile boolean cancelRequested = false;

    public CodeModelEngine(Context context, ToolRegistry tools) throws Exception {
        this.context = context.getApplicationContext();
        this.tools = tools == null ? new ToolRegistry() : tools;
        this.chatFile = new File(this.context.getFilesDir(), CHAT_FILE);
        this.chatStore = new ConversationFileStore(this.chatFile);

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

    public String generateCandidate(
            String question,
            int maxNewTokens,
            StreamListener listener) throws Exception {
        String q = clean(question);
        if (q.isEmpty()) return "";

        cancelRequested = false;
        List<ChatTurn> snapshot = getConversationSnapshot();
        StringBuilder raw = new StringBuilder();

        try (Sequences encoded = encodeTrimmedPrompt(snapshot, q)) {
            int[] inputIds = encoded.getSequence(0);
            int totalMaxLength =
                    Math.min(4096, inputIds.length + Math.max(16, maxNewTokens));

            try (GeneratorParams params = new GeneratorParams(model)) {
                params.setSearchOption("max_length", (double) totalMaxLength);
                params.setSearchOption("do_sample", false);
                params.setSearchOption("top_k", 50.0);
                params.setSearchOption("top_p", 0.95);
                params.setSearchOption("repetition_penalty", 1.08);

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

        String first = cleanup(raw.toString());
        Matcher call = TOOL_CALL.matcher(first);
        if (!call.find()) return first;
        LocalTool tool = tools.get(call.group(1));
        if (tool == null) return first;
        String result;
        try { result = tool.execute(call.group(2).trim()); }
        catch (Exception error) { result = "Tool error: " + error.getMessage(); }
        return generateAfterTool(snapshot, q, call.group(1), result, maxNewTokens, listener);
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
        String q = clean(question);
        String a = clean(answer);
        if (q.isEmpty() || a.isEmpty()) return;

        String id = clean(turnId);
        if (id.isEmpty()) id = java.util.UUID.randomUUID().toString();

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

    private Sequences encodeTrimmedPrompt(
            List<ChatTurn> snapshot,
            String question) throws Exception {
        ArrayList<ChatTurn> recent = new ArrayList<>(snapshot);
        while (true) {
            String prompt = buildPrompt(recent, question);
            Sequences encoded = tokenizer.encode(prompt);
            if (encoded.getSequence(0).length <= MAX_PROMPT_TOKENS) {
                return encoded;
            }
            encoded.close();

            if (recent.size() >= 2) {
                recent.remove(0);
                recent.remove(0);
                continue;
            }
            return tokenizer.encode(buildPrompt(
                    Collections.emptyList(), question));
        }
    }

    private String buildPrompt(List<ChatTurn> history, String question) {
        if (!chatTemplate.trim().isEmpty()) {
            try {
                JSONArray messages = new JSONArray();
                addMessage(messages, "system", SYSTEM_PROMPT + tools.promptDescription());

                int start = Math.max(0, history.size() - MAX_HISTORY_MESSAGES);
                for (int i = start; i < history.size(); i++) {
                    ChatTurn turn = history.get(i);
                    if (!"user".equals(turn.role)
                            && !"assistant".equals(turn.role)) {
                        continue;
                    }
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
                // Fall back to DeepSeek Coder's documented instruction format.
            }
        }

        StringBuilder out = new StringBuilder();
        out.append(SYSTEM_PROMPT).append(tools.promptDescription()).append("\n\n");

        int start = Math.max(0, history.size() - MAX_HISTORY_MESSAGES);
        for (int i = start; i < history.size(); i++) {
            ChatTurn turn = history.get(i);
            if ("user".equals(turn.role)) {
                out.append("### Instruction:\n")
                        .append(turn.content)
                        .append("\n");
            } else if ("assistant".equals(turn.role)) {
                out.append("### Response:\n")
                        .append(turn.content)
                        .append("\n<|EOT|>\n");
            }
        }

        out.append("### Instruction:\n")
                .append(question)
                .append("\n### Response:\n");
        return out.toString();
    }

    private String generateAfterTool(List<ChatTurn> history, String question,
            String toolName, String toolResult, int maxNewTokens,
            StreamListener listener) throws Exception {
        String augmented = question + "\n\nLocal tool " + toolName
                + " returned:\n" + toolResult + "\nUse it in the final answer.";
        StringBuilder raw = new StringBuilder();
        try (Sequences encoded = encodeTrimmedPrompt(history, augmented);
             GeneratorParams params = new GeneratorParams(model)) {
            int[] ids = encoded.getSequence(0);
            params.setSearchOption("max_length", (double)Math.min(4096,
                    ids.length + Math.max(16, maxNewTokens)));
            params.setSearchOption("do_sample", false);
            params.setSearchOption("repetition_penalty", 1.08);
            try (Generator generator = new Generator(model, params);
                 TokenizerStream stream = tokenizer.createStream()) {
                generator.appendTokenSequences(encoded);
                while (!generator.isDone() && !cancelRequested) {
                    generator.generateNextToken();
                    raw.append(stream.decode(generator.getLastTokenInSequence(0)));
                    String visible = cleanup(raw.toString());
                    if (listener != null && !visible.isEmpty()) listener.onUpdate(visible);
                    if (containsStopMarker(raw)) break;
                }
            }
        }
        return cleanup(raw.toString());
    }

    private static void addMessage(
            JSONArray messages,
            String role,
            String content) throws Exception {
        JSONObject obj = new JSONObject();
        obj.put("role", role);
        obj.put("content", content == null ? "" : content);
        messages.put(obj);
    }

    private void loadConversation() {
        conversation.replaceAll(chatStore.load());
    }

    private void saveConversationLocked() throws Exception {
        chatStore.save(conversation.snapshot());
    }

    private static boolean containsStopMarker(StringBuilder raw) {
        String s = raw.toString();
        return s.contains("<|EOT|>")
                || s.contains("<｜end▁of▁sentence｜>");
    }

    private static String cleanup(String text) {
        String out = text == null ? "" : text;
        int marker = out.indexOf("<|EOT|>");
        if (marker >= 0) out = out.substring(0, marker);
        marker = out.indexOf("<｜end▁of▁sentence｜>");
        if (marker >= 0) out = out.substring(0, marker);
        return out.trim();
    }

    private static String readTextAsset(
            AssetManager assets,
            String name,
            String fallback) {
        try (InputStream in = assets.open(name);
             BufferedReader reader = new BufferedReader(
                     new InputStreamReader(in, StandardCharsets.UTF_8))) {
            StringBuilder out = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                out.append(line).append('\n');
                if (out.length() > 8000) break;
            }
            String value = out.toString().trim();
            return value.isEmpty() ? fallback : value;
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private static void prepareModelDirectory(
            AssetManager assets,
            String assetPath,
            File target) throws Exception {
        File marker = new File(target, ".ready_deepseek_coder_v1");
        if (marker.exists()
                && new File(target, "genai_config.json").exists()) {
            return;
        }

        deleteRecursive(target);
        if (!target.mkdirs() && !target.isDirectory()) {
            throw new IllegalStateException("Cannot create model directory");
        }

        copyAssetTree(assets, assetPath, target);
        if (!new File(target, "genai_config.json").exists()) {
            throw new IllegalStateException(
                    "genai_config.json missing from model package");
        }

        if (!marker.createNewFile() && !marker.exists()) {
            throw new IllegalStateException(
                    "Cannot create model readiness marker");
        }
    }

    private static void copyAssetTree(
            AssetManager assets,
            String assetPath,
            File target) throws Exception {
        String[] children = assets.list(assetPath);
        if (children == null) {
            throw new IllegalStateException(
                    "Cannot read model asset: " + assetPath);
        }

        if (children.length == 0) {
            File parent = target.getParentFile();
            if (parent != null && !parent.exists() && !parent.mkdirs()) {
                throw new IllegalStateException(
                        "Cannot create: " + parent);
            }
            try (InputStream in = assets.open(assetPath);
                 FileOutputStream out = new FileOutputStream(target)) {
                byte[] buffer = new byte[1024 * 1024];
                int n;
                while ((n = in.read(buffer)) != -1) {
                    out.write(buffer, 0, n);
                }
                out.getFD().sync();
            }
            return;
        }

        if (!target.exists() && !target.mkdirs()) {
            throw new IllegalStateException("Cannot create: " + target);
        }

        for (String child : children) {
            copyAssetTree(
                    assets,
                    assetPath + "/" + child,
                    new File(target, child));
        }
    }

    private static void deleteRecursive(File file) {
        if (!file.exists()) return;
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) deleteRecursive(child);
            }
        }
        file.delete();
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }

    @Override
    public void close() {
        cancelRequested = true;
        try { tokenizer.close(); } catch (Exception ignored) {}
        try { model.close(); } catch (Exception ignored) {}
    }
}
