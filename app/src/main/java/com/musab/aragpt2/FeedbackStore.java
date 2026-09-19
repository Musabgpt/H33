package com.musab.aragpt2;

import android.content.Context;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;

public final class FeedbackStore {
    private final File file;

    public FeedbackStore(Context context) {
        this.file = new File(context.getFilesDir(), "feedback.jsonl");
    }

    public synchronized void append(String question, String modelAnswer, String taughtAnswer, float reward) {
        try {
            JSONObject obj = new JSONObject();
            obj.put("time_ms", System.currentTimeMillis());
            obj.put("question", question);
            obj.put("model_answer", modelAnswer == null ? "" : modelAnswer);
            obj.put("taught_answer", taughtAnswer == null ? "" : taughtAnswer);
            obj.put("reward", reward);
            byte[] bytes = (obj.toString() + "\n").getBytes(StandardCharsets.UTF_8);
            try (FileOutputStream out = new FileOutputStream(file, true)) {
                out.write(bytes);
            }
        } catch (Exception ignored) {
        }
    }
}
