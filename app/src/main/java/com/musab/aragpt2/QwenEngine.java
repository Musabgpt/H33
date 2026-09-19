package com.musab.aragpt2;

import android.content.Context;
import android.content.res.AssetManager;

import ai.onnxruntime.genai.Generator;
import ai.onnxruntime.genai.GeneratorParams;
import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.Sequences;
import ai.onnxruntime.genai.Tokenizer;
import ai.onnxruntime.genai.TokenizerStream;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.List;

public final class QwenEngine implements AutoCloseable {
    private static final String SYSTEM_PROMPT =
            "أنت مساعد عربي محلي. أجب مباشرة وبوضوح وبقدر السؤال. " +
            "لا ترفض سؤالاً عادياً لمجرد الحذر الزائد. إذا لم تعرف الإجابة فقل إنك لا تعرف، " +
            "ولا تختلق معلومات. استخدم العربية ما لم يطلب المستخدم لغة أخرى.";

    private static final String MODEL_ASSET_DIR = "model";
    private static final String MODEL_LOCAL_DIR = "qwen2_5_0_5b_int4";

    private final Context context;
    private final CorrectionMemory memory;
    private final Model model;
    private final Tokenizer tokenizer;

    public QwenEngine(Context context) throws Exception {
        this.context = context.getApplicationContext();
        this.memory = new CorrectionMemory(this.context);

        File modelDir = new File(this.context.getFilesDir(), MODEL_LOCAL_DIR);
        prepareModelDirectory(this.context.getAssets(), MODEL_ASSET_DIR, modelDir);

        this.model = new Model(modelDir.getAbsolutePath());
        this.tokenizer = new Tokenizer(model);
    }

    public String generate(String question, int maxNewTokens) throws Exception {
        String q = question == null ? "" : question.trim();
        if (q.isEmpty()) return "";

        String exact = memory.exactAnswer(q);
        if (exact != null && !exact.isEmpty()) return exact;

        String prompt = buildPrompt(q);
        try (Sequences encoded = tokenizer.encode(prompt)) {
            int[] inputIds = encoded.getSequence(0);
            int totalMaxLength = Math.min(2048, inputIds.length + Math.max(8, maxNewTokens));

            try (GeneratorParams params = new GeneratorParams(model)) {
                params.setSearchOption("max_length", (double) totalMaxLength);
                params.setSearchOption("do_sample", true);
                params.setSearchOption("temperature", 0.65);
                params.setSearchOption("top_k", 40.0);
                params.setSearchOption("top_p", 0.90);
                params.setSearchOption("repetition_penalty", 1.08);

                StringBuilder answer = new StringBuilder();
                try (Generator generator = new Generator(model, params);
                     TokenizerStream stream = tokenizer.createStream()) {
                    generator.appendTokenSequences(encoded);

                    while (!generator.isDone()) {
                        generator.generateNextToken();
                        int token = generator.getLastTokenInSequence(0);
                        answer.append(stream.decode(token));
                    }
                }
                return cleanup(answer.toString());
            }
        }
    }

    public String rememberCorrect(String question, String answer) throws Exception {
        memory.remember(question, answer, false);
        return "تم حفظ الإجابة الصحيحة في ذاكرة التعلم المحلية";
    }

    public String learnCorrection(String question, String wrongAnswer, String correction) throws Exception {
        memory.remember(question, correction, true);
        return "تم حفظ التصحيح وسيُستخدم مباشرةً عند تكرار السؤال وكمرجع للأسئلة القريبة";
    }

    public int memoryCount() {
        return memory.count();
    }

    private String buildPrompt(String question) {
        StringBuilder p = new StringBuilder();
        p.append("<|im_start|>system\n")
                .append(SYSTEM_PROMPT)
                .append("<|im_end|>\n");

        List<CorrectionMemory.Entry> examples = memory.bestExamples(question, 3);
        for (CorrectionMemory.Entry e : examples) {
            p.append("<|im_start|>user\n")
                    .append(e.question)
                    .append("<|im_end|>\n")
                    .append("<|im_start|>assistant\n")
                    .append(e.answer)
                    .append("<|im_end|>\n");
        }

        p.append("<|im_start|>user\n")
                .append(question)
                .append("<|im_end|>\n")
                .append("<|im_start|>assistant\n");

        return p.toString();
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
        try { tokenizer.close(); } catch (Exception ignored) {}
        try { model.close(); } catch (Exception ignored) {}
    }
}
