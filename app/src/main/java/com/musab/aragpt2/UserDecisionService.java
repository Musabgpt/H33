package com.musab.aragpt2;

import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class UserDecisionService {
    public interface CanonicalMemory {
        String get(String turnId);
        void commit(String turnId, String question, String answer) throws Exception;
        boolean replace(String turnId, String answer) throws Exception;
        boolean remove(String turnId) throws Exception;
    }

    public static final class Result {
        public final PreferenceRecord record;
        public final MemoryConflictStatus conflictStatus;

        Result(PreferenceRecord record, MemoryConflictStatus conflictStatus) {
            this.record = record;
            this.conflictStatus = conflictStatus == null
                    ? MemoryConflictStatus.NONE : conflictStatus;
        }

        public boolean hasFreshEvidenceConflict() {
            return conflictStatus == MemoryConflictStatus.CURRENT_EVIDENCE_CONFLICT;
        }
    }

    private static final Pattern NUMBER =
            Pattern.compile("(?<!\\p{L})[-+]?\\d+(?:[.,]\\d+)?");

    private final PreferenceStore preferenceStore;
    private final CanonicalMemory canonicalMemory;

    public UserDecisionService(
            PreferenceStore preferenceStore,
            CanonicalMemory canonicalMemory) {
        if (preferenceStore == null) {
            throw new IllegalArgumentException("preferenceStore required");
        }
        if (canonicalMemory == null) {
            throw new IllegalArgumentException("canonicalMemory required");
        }
        this.preferenceStore = preferenceStore;
        this.canonicalMemory = canonicalMemory;
    }

    public Result select(CandidateSet set, String candidateId) throws Exception {
        if (set == null) throw new IllegalArgumentException("candidate set required");
        AnswerCandidate selected = set.byId(candidateId);
        if (selected == null || !selected.available) {
            throw new IllegalArgumentException("selected candidate unavailable");
        }

        MemoryConflictStatus conflict =
                detectConflict(set, selected.answer);

        PreferenceRecord[] saved = new PreferenceRecord[1];
        commitWithRollback(
                set,
                selected.answer,
                () -> saved[0] = preferenceStore.recordSelection(
                        set, selected.id, conflict)
        );
        return new Result(saved[0], conflict);
    }

    public Result correct(CandidateSet set, String correction) throws Exception {
        if (set == null) throw new IllegalArgumentException("candidate set required");
        String answer = clean(correction);
        if (answer.isEmpty()) throw new IllegalArgumentException("correction required");

        MemoryConflictStatus conflict = detectConflict(set, answer);

        PreferenceRecord[] saved = new PreferenceRecord[1];
        commitWithRollback(
                set,
                answer,
                () -> saved[0] = preferenceStore.recordCorrection(
                        set, answer, conflict)
        );
        return new Result(saved[0], conflict);
    }

    private void commitWithRollback(
            CandidateSet set,
            String answer,
            PreferenceWrite preferenceWrite) throws Exception {
        String previous = canonicalMemory.get(set.turnId);
        boolean hadPrevious = !clean(previous).isEmpty();
        boolean changed = false;

        try {
            if (hadPrevious) {
                if (!canonicalMemory.replace(set.turnId, answer)) {
                    throw new IllegalStateException(
                            "تعذر العثور على turn لاعتماد الجواب");
                }
            } else {
                canonicalMemory.commit(set.turnId, set.question, answer);
            }
            changed = true;
            preferenceWrite.write();
        } catch (Exception ex) {
            if (changed) {
                try {
                    if (hadPrevious) {
                        if (!canonicalMemory.replace(set.turnId, previous)) {
                            throw new IllegalStateException(
                                    "تعذر استعادة الجواب السابق");
                        }
                    } else {
                        canonicalMemory.remove(set.turnId);
                    }
                } catch (Exception rollbackError) {
                    ex.addSuppressed(rollbackError);
                }
            }
            throw ex;
        }
    }

    static MemoryConflictStatus detectConflict(
            CandidateSet set,
            String userApprovedAnswer) {
        if (set == null
                || !FreshnessPolicy.requiresFreshEvidence(set.question)
                || set.web == null
                || !set.web.available
                || set.web.sources.isEmpty()) {
            return MemoryConflictStatus.NONE;
        }

        String approved = normalize(userApprovedAnswer);
        String evidence = normalize(set.web.answer);
        if (approved.isEmpty() || evidence.isEmpty() || approved.equals(evidence)) {
            return MemoryConflictStatus.NONE;
        }

        Set<String> approvedNumbers = numbers(approved);
        Set<String> evidenceNumbers = numbers(evidence);
        if (!approvedNumbers.isEmpty()
                && !evidenceNumbers.isEmpty()
                && disjoint(approvedNumbers, evidenceNumbers)) {
            return MemoryConflictStatus.CURRENT_EVIDENCE_CONFLICT;
        }

        if (booleanOpposite(approved, evidence)) {
            return MemoryConflictStatus.CURRENT_EVIDENCE_CONFLICT;
        }

        return MemoryConflictStatus.NONE;
    }

    private static boolean booleanOpposite(String a, String b) {
        boolean aYes = token(a, "yes") || token(a, "نعم");
        boolean aNo = token(a, "no") || token(a, "لا");
        boolean bYes = token(b, "yes") || token(b, "نعم");
        boolean bNo = token(b, "no") || token(b, "لا");
        return (aYes && bNo) || (aNo && bYes);
    }

    private static boolean token(String text, String token) {
        for (String part : text.split("\\s+")) {
            if (part.equals(token)) return true;
        }
        return false;
    }

    private static Set<String> numbers(String text) {
        HashSet<String> out = new HashSet<>();
        Matcher matcher = NUMBER.matcher(text);
        while (matcher.find()) {
            out.add(matcher.group().replace(',', '.'));
        }
        return out;
    }

    private static boolean disjoint(Set<String> a, Set<String> b) {
        for (String value : a) {
            if (b.contains(value)) return false;
        }
        return true;
    }

    private static String normalize(String value) {
        return clean(value)
                .toLowerCase(java.util.Locale.ROOT)
                .replaceAll("[ًٌٍَُِّْـ]", "")
                .replaceAll("[\\p{Punct}\\p{S}،؛؟]+", " ")
                .replaceAll("\\s+", " ")
                .trim();
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }

    @FunctionalInterface
    private interface PreferenceWrite {
        void write() throws Exception;
    }
}
