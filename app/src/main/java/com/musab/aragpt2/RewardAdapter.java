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
import java.util.Random;

public final class RewardAdapter {
    private static final int MAGIC = 0x41524131;
    private static final int VERSION = 1;
    private static final String FILE_NAME = "reward_adapter_v1.bin";

    private final int hiddenSize;
    private final int vocabSize;
    private final int rank;
    private final float[] projection;
    private final float[] weights;
    private final File saveFile;

    private final float learningRate;
    private final float maxAbsWeight;

    public RewardAdapter(Context context, int hiddenSize, int vocabSize, int rank) {
        this.hiddenSize = hiddenSize;
        this.vocabSize = vocabSize;
        this.rank = rank;
        this.learningRate = 0.10f;
        this.maxAbsWeight = 2.5f;
        this.projection = makeProjection(hiddenSize, rank);
        this.weights = new float[rank * vocabSize];
        this.saveFile = new File(context.getFilesDir(), FILE_NAME);
        loadIfCompatible();
    }

    public synchronized void addToLogits(float[] logits, float[] hidden) {
        float[] z = project(hidden);
        for (int v = 0; v < vocabSize; v++) {
            float delta = 0f;
            for (int r = 0; r < rank; r++) {
                delta += z[r] * weights[r * vocabSize + v];
            }
            logits[v] += delta;
        }
    }

    public synchronized float reinforce(float[] baseLogits, float[] hidden, int targetId,
                                        float reward, int policyTopK) {
        if (targetId < 0 || targetId >= vocabSize || reward == 0f) return 0f;
        float[] z = project(hidden);
        int k = Math.max(2, Math.min(policyTopK, vocabSize));
        int[] ids = topKWithTarget(baseLogits, z, targetId, k);
        float[] scores = new float[ids.length];

        float max = -Float.MAX_VALUE;
        for (int i = 0; i < ids.length; i++) {
            int id = ids[i];
            float s = baseLogits[id];
            for (int r = 0; r < rank; r++) s += z[r] * weights[r * vocabSize + id];
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
            for (int r = 0; r < rank; r++) {
                int idx = r * vocabSize + id;
                float w = weights[idx] + coeff * z[r];
                if (w > maxAbsWeight) w = maxAbsWeight;
                else if (w < -maxAbsWeight) w = -maxAbsWeight;
                weights[idx] = w;
            }
        }
        return targetProb;
    }

    public synchronized void save() throws Exception {
        File tmp = new File(saveFile.getParentFile(), saveFile.getName() + ".tmp");
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(tmp)))) {
            out.writeInt(MAGIC);
            out.writeInt(VERSION);
            out.writeInt(hiddenSize);
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

    public synchronized void reset() {
        Arrays.fill(weights, 0f);
        if (saveFile.exists()) saveFile.delete();
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
            int h = in.readInt();
            int v = in.readInt();
            int r = in.readInt();
            int n = in.readInt();
            if (magic != MAGIC || version != VERSION || h != hiddenSize || v != vocabSize || r != rank || n != weights.length) {
                return;
            }
            for (int i = 0; i < weights.length; i++) weights[i] = in.readFloat();
        } catch (Exception ignored) {
            Arrays.fill(weights, 0f);
        }
    }

    private float[] project(float[] hidden) {
        float[] z = new float[rank];
        for (int h = 0; h < hiddenSize; h++) {
            float x = hidden[h];
            int base = h * rank;
            for (int r = 0; r < rank; r++) z[r] += x * projection[base + r];
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
            for (int r = 0; r < rank; r++) s += z[r] * weights[r * vocabSize + v];
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

    private static float[] makeProjection(int hiddenSize, int rank) {
        float[] p = new float[hiddenSize * rank];
        Random rnd = new Random(0xA2A62026L + hiddenSize * 31L + rank);
        float scale = (float) (1.0 / Math.sqrt(hiddenSize));
        for (int i = 0; i < p.length; i++) p[i] = (float) rnd.nextGaussian() * scale;
        return p;
    }
}
