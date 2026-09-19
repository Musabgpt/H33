package com.musab.aragpt2;

import android.content.Context;
import android.content.res.AssetManager;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public final class AraGpt2Engine implements AutoCloseable {
    private static final int EOS_ID = 0;
    private static final int MAX_CONTEXT = 192;
    private static final int ADAPTER_RANK = 8;
    private static final int POLICY_TOP_K = 32;
    private static final int MAX_TRAIN_TOKENS = 48;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Gpt2Tokenizer tokenizer;
    private final Random random = new Random(42);
    private final Context context;
    private final FeedbackStore feedbackStore;

    private RewardAdapter adapter;

    private static final class StepOutput {
        final float[] logits;
        final float[] hidden;
        StepOutput(float[] logits, float[] hidden) {
            this.logits = logits;
            this.hidden = hidden;
        }
    }

    public AraGpt2Engine(Context context) throws Exception {
        this.context = context.getApplicationContext();
        this.feedbackStore = new FeedbackStore(this.context);
        tokenizer = new Gpt2Tokenizer(this.context);
        String modelAsset = findModelAsset(this.context.getAssets());
        if (modelAsset == null) {
            throw new IllegalStateException("ملف النموذج غير موجود. ابنِ APK من GitHub Actions حتى يتم تضمين AraGPT2.");
        }
        File modelFile = ensureAssetCopied(this.context, modelAsset);
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(Math.max(2, Math.min(4, Runtime.getRuntime().availableProcessors())));
        options.setInterOpNumThreads(1);
        session = env.createSession(modelFile.getAbsolutePath(), options);
    }

    public String generate(String question, int maxNewTokens, float temperature, int topK) throws Exception {
        String prompt = formatPrompt(question);
        List<Integer> ids = new ArrayList<>(tokenizer.encode(prompt));
        if (ids.isEmpty()) ids.add(EOS_ID);
        trimLeft(ids, MAX_CONTEXT);

        List<Integer> generated = new ArrayList<>();
        for (int step = 0; step < maxNewTokens && ids.size() < MAX_CONTEXT; step++) {
            StepOutput out = runBase(ids);
            ensureAdapter(out);
            adapter.addToLogits(out.logits, out.hidden);
            int next = sampleTopK(out.logits, temperature, topK);
            if (next == EOS_ID) break;
            ids.add(next);
            generated.add(next);
            if (generated.size() >= 2) {
                String partial = tokenizer.decode(generated);
                if (partial.contains("\nسؤال:") || partial.contains("\n\n")) break;
            }
        }
        return tokenizer.decode(generated).trim();
    }

    public String reinforceCorrect(String question, String answer) throws Exception {
        int steps = trainAnswer(question, answer, +1.0f, 1);
        feedbackStore.append(question, answer, answer, +1.0f);
        return "تم تعزيز الإجابة الصحيحة عبر " + steps + " خطوة تعلم";
    }

    public String learnCorrection(String question, String wrongAnswer, String correction) throws Exception {
        int down = 0;
        if (wrongAnswer != null && !wrongAnswer.trim().isEmpty()) {
            down = trainAnswer(question, wrongAnswer, -0.20f, 1);
        }
        int up = trainAnswer(question, correction, +1.0f, 2);
        feedbackStore.append(question, wrongAnswer, correction, +1.0f);
        return "تم التعلم: " + up + " خطوة للتصحيح" + (down > 0 ? " و" + down + " خطوة لتقليل الجواب الخاطئ" : "");
    }

    public long learnedWeightCount() {
        return adapter == null ? 0 : adapter.nonZeroWeightCount();
    }

    private int trainAnswer(String question, String answer, float reward, int epochs) throws Exception {
        if (answer == null || answer.trim().isEmpty()) return 0;
        List<Integer> prompt = new ArrayList<>(tokenizer.encode(formatPrompt(question)));
        List<Integer> targets = new ArrayList<>(tokenizer.encode(answer.trim()));
        if (targets.isEmpty()) return 0;
        if (targets.size() > MAX_TRAIN_TOKENS) targets = new ArrayList<>(targets.subList(0, MAX_TRAIN_TOKENS));

        int steps = 0;
        for (int epoch = 0; epoch < epochs; epoch++) {
            List<Integer> contextIds = new ArrayList<>(prompt);
            trimLeft(contextIds, MAX_CONTEXT);
            for (int target : targets) {
                StepOutput out = runBase(contextIds);
                ensureAdapter(out);
                adapter.reinforce(out.logits, out.hidden, target, reward, POLICY_TOP_K);
                contextIds.add(target);
                trimLeft(contextIds, MAX_CONTEXT);
                steps++;
            }
        }
        if (adapter != null) adapter.save();
        return steps;
    }

    private void ensureAdapter(StepOutput out) {
        if (adapter == null) {
            adapter = new RewardAdapter(context, out.hidden.length, out.logits.length, ADAPTER_RANK);
        }
    }

    private StepOutput runBase(List<Integer> ids) throws Exception {
        if (ids.isEmpty()) throw new IllegalArgumentException("السياق فارغ");
        long[] shape = new long[]{1, ids.size()};
        long[] input = new long[ids.size()];
        long[] mask = new long[ids.size()];
        for (int i = 0; i < ids.size(); i++) {
            input[i] = ids.get(i);
            mask[i] = 1L;
        }

        try (OnnxTensor inputIds = OnnxTensor.createTensor(env, LongBuffer.wrap(input), shape);
             OnnxTensor attentionMask = OnnxTensor.createTensor(env, LongBuffer.wrap(mask), shape)) {
            Map<String, OnnxTensor> feed = new HashMap<>();
            feed.put("input_ids", inputIds);
            feed.put("attention_mask", attentionMask);
            try (OrtSession.Result result = session.run(feed)) {
                float[] logits = flatten2d(result.get(0).getValue(), "logits");
                float[] hidden = flatten2d(result.get(1).getValue(), "hidden");
                return new StepOutput(logits, hidden);
            }
        }
    }

    private static float[] flatten2d(Object value, String name) {
        if (value instanceof float[][]) return ((float[][]) value)[0].clone();
        if (value instanceof float[]) return ((float[]) value).clone();
        throw new IllegalStateException("Unexpected ONNX output for " + name + ": " + value.getClass());
    }

    private int sampleTopK(float[] logits, float temperature, int topK) {
        int k = Math.max(1, Math.min(topK, logits.length));
        int[] idx = new int[k];
        float[] vals = new float[k];
        for (int i = 0; i < k; i++) { idx[i] = -1; vals[i] = -Float.MAX_VALUE; }
        for (int i = 0; i < logits.length; i++) {
            float v = logits[i];
            if (v <= vals[k - 1]) continue;
            int pos = k - 1;
            while (pos > 0 && v > vals[pos - 1]) {
                vals[pos] = vals[pos - 1]; idx[pos] = idx[pos - 1]; pos--;
            }
            vals[pos] = v; idx[pos] = i;
        }
        if (temperature <= 0.01f) return idx[0];
        float max = vals[0] / temperature;
        double sum = 0.0;
        double[] probs = new double[k];
        for (int i = 0; i < k; i++) {
            probs[i] = Math.exp((vals[i] / temperature) - max);
            sum += probs[i];
        }
        double r = random.nextDouble() * sum;
        for (int i = 0; i < k; i++) {
            r -= probs[i];
            if (r <= 0) return idx[i];
        }
        return idx[0];
    }

    private static void trimLeft(List<Integer> ids, int max) {
        if (ids.size() <= max) return;
        int remove = ids.size() - max;
        ids.subList(0, remove).clear();
    }

    private static String formatPrompt(String question) {
        return "سؤال: " + question.trim() + "\nجواب:\n";
    }

    private static String findModelAsset(AssetManager am) throws Exception {
        String[] files = am.list("");
        if (files == null) return null;
        for (String f : files) if (f.equals("aragpt2.int8.onnx")) return f;
        for (String f : files) if (f.equals("aragpt2.fp32.onnx")) return f;
        return null;
    }

    private static File ensureAssetCopied(Context context, String assetName) throws Exception {
        File dst = new File(context.getFilesDir(), assetName);
        if (dst.exists() && dst.length() > 1024 * 1024) return dst;
        try (InputStream in = context.getAssets().open(assetName);
             FileOutputStream out = new FileOutputStream(dst)) {
            byte[] buf = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
        }
        return dst;
    }

    @Override public void close() throws Exception {
        session.close();
        env.close();
    }
}
