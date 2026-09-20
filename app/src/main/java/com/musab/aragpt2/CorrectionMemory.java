package com.musab.aragpt2;

import android.content.Context;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public final class CorrectionMemory {
    public static final class Entry {
        public final String question;
        public final String answer;
        public final boolean corrected;
        public final long timeMs;
        double score;

        Entry(String question, String answer, boolean corrected, long timeMs) {
            this.question = question;
            this.answer = answer;
            this.corrected = corrected;
            this.timeMs = timeMs;
        }
    }

    private final File file;

    public CorrectionMemory(Context context) {
        file = new File(context.getFilesDir(), "legacy_learning_memory.jsonl");
    }

    public synchronized void remember(String question, String answer, boolean corrected) throws Exception {
        String q = clean(question);
        String a = clean(answer);
        if (q.isEmpty() || a.isEmpty()) return;

        JSONObject obj = new JSONObject();
        obj.put("time_ms", System.currentTimeMillis());
        obj.put("question", q);
        obj.put("answer", a);
        obj.put("corrected", corrected);

        try (FileOutputStream out = new FileOutputStream(file, true)) {
            out.write((obj.toString() + "\n").getBytes(StandardCharsets.UTF_8));
            out.getFD().sync();
        }
    }

    public synchronized String exactAnswer(String question) {
        String target = normalize(question);
        if (target.isEmpty()) return null;

        List<Entry> all = load();
        for (int i = all.size() - 1; i >= 0; i--) {
            Entry e = all.get(i);
            if (normalize(e.question).equals(target)) return e.answer;
        }
        return null;
    }

    public synchronized List<Entry> bestExamples(String question, int limit) {
        if (limit <= 0) return Collections.emptyList();
        String target = normalize(question);
        Set<String> targetWords = words(target);
        List<Entry> all = load();

        for (Entry e : all) {
            String q = normalize(e.question);
            if (q.equals(target)) {
                e.score = 2.0;
            } else {
                e.score = jaccard(targetWords, words(q));
                if (e.corrected) e.score += 0.08;
            }
        }

        all.removeIf(e -> e.score < 0.16);
        all.sort(Comparator
                .comparingDouble((Entry e) -> e.score).reversed()
                .thenComparingLong(e -> -e.timeMs));

        ArrayList<Entry> unique = new ArrayList<>();
        HashSet<String> seen = new HashSet<>();
        for (Entry e : all) {
            String key = normalize(e.question);
            if (seen.add(key)) unique.add(e);
            if (unique.size() >= limit) break;
        }
        Collections.reverse(unique);
        return unique;
    }

    public synchronized int count() {
        return load().size();
    }

    private List<Entry> load() {
        ArrayList<Entry> out = new ArrayList<>();
        if (!file.exists()) return out;

        try (BufferedReader r = new BufferedReader(new InputStreamReader(
                new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                try {
                    JSONObject obj = new JSONObject(line);
                    String q = obj.optString("question", "");
                    String a = obj.optString("answer", "");
                    if (q.isEmpty() || a.isEmpty()) continue;
                    out.add(new Entry(
                            q,
                            a,
                            obj.optBoolean("corrected", false),
                            obj.optLong("time_ms", 0L)
                    ));
                } catch (Exception ignored) {
                }
            }
        } catch (Exception ignored) {
        }
        return out;
    }

    private static double jaccard(Set<String> a, Set<String> b) {
        if (a.isEmpty() || b.isEmpty()) return 0.0;
        int intersection = 0;
        for (String s : a) if (b.contains(s)) intersection++;
        int union = a.size() + b.size() - intersection;
        return union == 0 ? 0.0 : ((double) intersection / union);
    }

    private static Set<String> words(String s) {
        HashSet<String> out = new HashSet<>();
        for (String w : s.split("\\s+")) {
            if (w.length() > 1) out.add(w);
        }
        return out;
    }

    private static String normalize(String s) {
        String n = clean(s).toLowerCase(Locale.ROOT);
        n = n.replaceAll("[\\p{Punct}\\p{S}،؛؟]+", " ");
        n = n.replaceAll("[ًٌٍَُِّْـ]", "");
        n = n.replaceAll("\\s+", " ").trim();
        return n;
    }

    private static String clean(String s) {
        return s == null ? "" : s.trim();
    }
}
