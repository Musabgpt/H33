package com.musab.aragpt2;

import android.content.Context;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class PreferenceStore {
    private static final String FILE_NAME = "h33_preferences_v1.jsonl";

    private final File file;

    public PreferenceStore(Context context) {
        this(new File(context.getFilesDir(), FILE_NAME));
    }

    PreferenceStore(File file) {
        if (file == null) throw new IllegalArgumentException("file required");
        this.file = file;
    }

    public synchronized PreferenceRecord recordSelection(
            CandidateSet set, String candidateId) throws Exception {
        PreferenceRecord record = PreferenceRecord.selection(
                set, candidateId, System.currentTimeMillis());
        append(record);
        return record;
    }

    public synchronized PreferenceRecord recordCorrection(
            CandidateSet set, String correction) throws Exception {
        PreferenceRecord record = PreferenceRecord.correction(
                set, correction, System.currentTimeMillis());
        append(record);
        return record;
    }

    public synchronized void append(PreferenceRecord record) throws Exception {
        if (record == null) throw new IllegalArgumentException("record required");

        File parent = file.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IllegalStateException("تعذر إنشاء مجلد بيانات التعلم");
        }

        byte[] line = (record.toJson() + "\n").getBytes(StandardCharsets.UTF_8);
        try (FileOutputStream out = new FileOutputStream(file, true)) {
            out.write(line);
            out.flush();
            out.getFD().sync();
        }
    }

    public synchronized List<PreferenceRecord> loadAll() {
        ArrayList<PreferenceRecord> out = new ArrayList<>();
        if (!file.exists()) return out;

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String value = line.trim();
                if (value.isEmpty()) continue;
                try {
                    out.add(PreferenceRecord.fromJson(value));
                } catch (Exception ignored) {
                    // Keep earlier durable events even if one line is incomplete/corrupt.
                }
            }
        } catch (Exception ignored) {
            // Caller receives every record successfully parsed before the read failed.
        }
        return out;
    }

    public synchronized PreferenceRecord latestDecision(String turnId) {
        String target = turnId == null ? "" : turnId.trim();
        if (target.isEmpty()) return null;

        List<PreferenceRecord> all = loadAll();
        for (int i = all.size() - 1; i >= 0; i--) {
            PreferenceRecord record = all.get(i);
            if (target.equals(record.turnId)) return record;
        }
        return null;
    }

    public synchronized List<PreferenceRecord> latestDecisions() {
        LinkedHashMap<String, PreferenceRecord> latest = new LinkedHashMap<>();
        for (PreferenceRecord record : loadAll()) {
            if (record.turnId.isEmpty()) continue;
            if (latest.containsKey(record.turnId)) {
                latest.remove(record.turnId);
            }
            latest.put(record.turnId, record);
        }
        return new ArrayList<>(latest.values());
    }

    public synchronized int eventCount() {
        return loadAll().size();
    }
}
