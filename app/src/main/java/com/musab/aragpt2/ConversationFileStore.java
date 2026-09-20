package com.musab.aragpt2;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class ConversationFileStore {
    private final File file;

    public ConversationFileStore(File file) {
        if (file == null) throw new IllegalArgumentException("file required");
        this.file = file;
    }

    public synchronized List<ConversationHistory.Turn> load() {
        if (!file.exists()) return Collections.emptyList();

        ArrayList<ConversationHistory.Turn> loaded = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String value = line.trim();
                if (value.isEmpty()) continue;

                try {
                    JSONObject obj = new JSONObject(value);
                    String turnId = obj.optString("turn_id", "");
                    String role = obj.optString("role", "");
                    String content = obj.optString("content", "");

                    if (("user".equals(role) || "assistant".equals(role))
                            && !content.trim().isEmpty()) {
                        loaded.add(new ConversationHistory.Turn(
                                turnId, role, content));
                    }
                } catch (Exception ignoredMalformedLine) {
                    // Keep valid durable turns even if one JSONL line is incomplete.
                }
            }
        } catch (Exception ignoredReadFailure) {
            // Return every complete turn parsed before the read failure.
        }
        return loaded;
    }

    public synchronized void save(List<ConversationHistory.Turn> turns)
            throws Exception {
        File parent = file.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IllegalStateException("تعذر إنشاء مجلد سجل المحادثة");
        }

        File tmp = new File(parent, file.getName() + ".tmp");
        try {
            try (FileOutputStream out = new FileOutputStream(tmp, false)) {
                if (turns != null) {
                    for (ConversationHistory.Turn turn : turns) {
                        if (turn == null) continue;
                        if (!"user".equals(turn.role)
                                && !"assistant".equals(turn.role)) {
                            continue;
                        }
                        if (turn.content == null || turn.content.trim().isEmpty()) {
                            continue;
                        }

                        JSONObject obj = new JSONObject();
                        obj.put("turn_id", turn.turnId);
                        obj.put("role", turn.role);
                        obj.put("content", turn.content);
                        out.write((obj.toString() + "\n")
                                .getBytes(StandardCharsets.UTF_8));
                    }
                }
                out.flush();
                out.getFD().sync();
            }

            try {
                Files.move(
                        tmp.toPath(),
                        file.toPath(),
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE
                );
            } catch (AtomicMoveNotSupportedException unsupported) {
                Files.move(
                        tmp.toPath(),
                        file.toPath(),
                        StandardCopyOption.REPLACE_EXISTING
                );
            }
        } finally {
            if (tmp.exists() && !tmp.equals(file)) {
                // Only a failed move can leave this behind.
                tmp.delete();
            }
        }
    }
}
