package com.musab.aragpt2;

import android.content.Context;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.Arrays;
import java.util.List;

public final class RewardAdapter {
    private static final int MAGIC = 0x41524132;
    private static final int VERSION = 2;
    private static final String FILE_NAME = "reward_adapter_v2.bin";
    private static final int CONTEXT_TOKENS = 64;

    private final int vocabSize;
    private final int rank;
    private final float[] weights;
    private final File saveFile;
    private final float learningRate = 0.10f;
    private final float maxAbsWeight = 2.5f;

    public RewardAdapter(Context context, int vocabSize, int rank) {
        this.vocabSize = vocabSize;
        this.rank = rank;
        this.weights = new float[rank * vocabSize];
        this.saveFile = new File(context.getFilesDir(), FILE_NAME);
        loadIfCompatible();
    }

    public synchronized void addToLogits(float[] logits, List<Integer> contextIds) {
        float[] z = contextFeatures(contextIds);
        for (int v = 0; v < vocabSize; v++) {
            float delta = 0f;
            int offset = v;
            for (int r = 0; r < rank; r++, offset += vocabSize) {
                delta += z[r] * weights[offset];
            }
            logits[v] += delta;
        }
    }

    public synchronized float reinforce(float[] baseLogits, List<Integer> contextIds, int targetId,
                                        float reward, int policyTopK) {
        if (targetId < 0 || targetId >= vocabSize || reward == 0f) return 0f;
        float[] z = contextFeatures(contextIds);
        int k = Math.max(2, Math.min(policyTopK, vocabSize));
        int[] ids = topKWithTarget(baseLogits, z, targetId, k);
        float[] scores = new float[ids.length];

        float max = -Float.MAX_VALUE;
        for (int i = 0; i < ids.length; i++) {
            int id = ids[i];
            float s = baseLogits[id];
            int offset = id;
            for (int r = 0; r < rank; r++, offset += vocabSize) {
                s += z[r] * weights[offset];
            }
            scores[i] = s;
            if (s > max) max = s;
        }

        double sum = 0.0;
        for (int i = 0; i < scores.length; i++) {
            scores[i] = (float) Math.exp(Math.max(-40f, Math.min(40f, scores[i] - max)));
            sum += scores[i];
        }
        if (sum <= 0.0 || Double.isNaN(sum)) return 0f;

        float targetProb = 0f;
        for (int i = 0; i < ids.length; i++) {
            float p = (float) (scores[i] / sum);
            int id = ids[i];
            if (id == targetId) targetProb = p;
            float advantage = ((id == targetId) ? 1f : 0f) - p;
            float coeff = learningRate * reward * advantage;
            int offset = id;
            for (int r = 0; r < rank; r++, offset += vocabSize) {
                float w = weights[offset] + coeff * z[r];
                if (w > maxAbsWeight) w = maxAbsWeight;
                else if (w < -maxAbsWeight) w = -maxAbsWeight;
                weights[offset] = w;
            }
        }
        return targetProb;
    }

    public synchronized void save() throws Exception {
        File tmp = new File(saveFile.getParentFile(), saveFile.getName() + ".tmp");
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(tmp)))) {
            out.writeInt(MAGIC);
            out.writeInt(VERSION);
            out.writeInt(vocabSize);
            out.writeInt(rank);
            out.writeInt(weights.length);
            for (float w : weights) out.writeFloat(w);
        }
        if (saveFile.exists() && !saveFile.delete()) {
            throw new IllegalStateException("تعذر استبدال ملف التعلم القديم");
        }
        if (!tmp.renameTo(saveFile)) {
            throw new IllegalStateException("تعذر حفظ أوزان التعلم");
        }
    }

    public synchronized long nonZeroWeightCount() {
        long count = 0;
        for (float w : weights) if (Math.abs(w) > 1e-8f) count++;
        return count;
    }

    private void loadIfCompatible() {
        if (!saveFile.exists()) return;
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(saveFile)))) {
            int magic = in.readInt();
            int version = in.readInt();
            int v = in.readInt();
            int r = in.readInt();
            int n = in.readInt();
            if (magic != MAGIC || version != VERSION || v != vocabSize || r != rank || n != weights.length) {
                return;
            }
            for (int i = 0; i < weights.length; i++) weights[i] = in.readFloat();
        } catch (Exception ignored) {
            Arrays.fill(weights, 0f);
        }
    }

    private float[] contextFeatures(List<Integer> contextIds) {
        float[] z = new float[rank];
        if (contextIds == null || contextIds.isEmpty()) {
            z[0] = 1f;
            return z;
        }

        int start = Math.max(0, contextIds.size() - CONTEXT_TOKENS);
        int count = contextIds.size() - start;
        for (int i = start; i < contextIds.size(); i++) {
            int token = contextIds.get(i);
            int relative = i - start;
            float recency = 0.30f + 0.70f * ((relative + 1f) / count);
            long base = mix64(((long) token << 32) ^ (relative * 0x9E3779B97F4A7C15L));
            for (int r = 0; r < rank; r++) {
                long h = mix64(base + r * 0xD1B54A32D192ED03L);
                z[r] += ((h & 1L) == 0L ? recency : -recency);
            }
        }

        float norm = 1e-6f;
        for (float x : z) norm += x * x;
        norm = (float) Math.sqrt(norm);
        for (int r = 0; r < rank; r++) z[r] /= norm;
        return z;
    }

    private int[] topKWithTarget(float[] baseLogits, float[] z, int targetId, int k) {
        int[] ids = new int[k];
        float[] vals = new float[k];
        Arrays.fill(ids, -1);
        Arrays.fill(vals, -Float.MAX_VALUE);

        for (int v = 0; v < vocabSize; v++) {
            float s = baseLogits[v];
            int offset = v;
            for (int r = 0; r < rank; r++, offset += vocabSize) {
                s += z[r] * weights[offset];
            }
            if (s <= vals[k - 1]) continue;
            int pos = k - 1;
            while (pos > 0 && s > vals[pos - 1]) {
                vals[pos] = vals[pos - 1];
                ids[pos] = ids[pos - 1];
                pos--;
            }
            vals[pos] = s;
            ids[pos] = v;
        }

        boolean found = false;
        for (int id : ids) if (id == targetId) { found = true; break; }
        if (!found) ids[k - 1] = targetId;
        return ids;
    }

    private static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }
}
