package com.musab.aragpt2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class ConversationHistory {
    public static final class Turn {
        public final String turnId;
        public final String role;
        public final String content;

        public Turn(String turnId, String role, String content) {
            this.turnId = clean(turnId);
            this.role = clean(role);
            this.content = clean(content);
        }
    }

    private final int maxMessages;
    private final ArrayList<Turn> turns = new ArrayList<>();

    public ConversationHistory(int maxMessages) {
        this.maxMessages = Math.max(2, maxMessages);
    }

    public synchronized void appendTurn(String turnId, String question, String answer) {
        String q = clean(question);
        String a = clean(answer);
        if (q.isEmpty() || a.isEmpty()) return;

        String id = clean(turnId);
        turns.add(new Turn(id, "user", q));
        turns.add(new Turn(id, "assistant", a));
        trimPairs();
    }

    public synchronized boolean replaceAnswer(String turnId, String answer) {
        String id = clean(turnId);
        String a = clean(answer);
        if (id.isEmpty() || a.isEmpty()) return false;

        for (int i = turns.size() - 1; i >= 0; i--) {
            Turn t = turns.get(i);
            if ("assistant".equals(t.role) && id.equals(t.turnId)) {
                turns.set(i, new Turn(t.turnId, "assistant", a));
                return true;
            }
        }
        return false;
    }

    public synchronized String answerForTurn(String turnId) {
        String id = clean(turnId);
        if (id.isEmpty()) return "";

        for (int i = turns.size() - 1; i >= 0; i--) {
            Turn t = turns.get(i);
            if ("assistant".equals(t.role) && id.equals(t.turnId)) {
                return t.content;
            }
        }
        return "";
    }

    public synchronized boolean removeTurn(String turnId) {
        String id = clean(turnId);
        if (id.isEmpty()) return false;

        boolean removed = false;
        for (int i = turns.size() - 1; i >= 0; i--) {
            if (id.equals(turns.get(i).turnId)) {
                turns.remove(i);
                removed = true;
            }
        }
        return removed;
    }

    public synchronized boolean replaceLatestAnswer(
            String question, String oldAnswer, String correctedAnswer) {
        String q = clean(question);
        String old = clean(oldAnswer);
        String corrected = clean(correctedAnswer);
        if (q.isEmpty() || corrected.isEmpty()) return false;

        for (int i = turns.size() - 1; i >= 1; i--) {
            Turn assistant = turns.get(i);
            Turn user = turns.get(i - 1);
            if (!"assistant".equals(assistant.role) || !"user".equals(user.role)) continue;
            if (!user.content.equals(q)) continue;
            if (!old.isEmpty() && !assistant.content.equals(old)) continue;

            turns.set(i, new Turn(assistant.turnId, "assistant", corrected));
            return true;
        }
        return false;
    }

    public synchronized void replaceAll(List<Turn> loaded) {
        turns.clear();
        if (loaded != null) {
            for (Turn t : loaded) {
                if (t == null) continue;
                if (!"user".equals(t.role) && !"assistant".equals(t.role)) continue;
                if (t.content.isEmpty()) continue;
                turns.add(new Turn(t.turnId, t.role, t.content));
            }
        }
        trimPairs();
    }

    public synchronized void clear() {
        turns.clear();
    }

    public synchronized List<Turn> snapshot() {
        return Collections.unmodifiableList(new ArrayList<>(turns));
    }

    private void trimPairs() {
        while (turns.size() > maxMessages) {
            if (turns.size() >= 2) {
                turns.remove(0);
                turns.remove(0);
            } else {
                turns.remove(0);
            }
        }

        while (!turns.isEmpty() && !"user".equals(turns.get(0).role)) {
            turns.remove(0);
        }

        if ((turns.size() & 1) != 0) {
            turns.remove(0);
        }
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
