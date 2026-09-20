package com.musab.aragpt2;

import java.util.UUID;

public final class CandidateCoordinator {
    @FunctionalInterface
    public interface CanonicalTurnStore {
        void commit(String turnId, String question, String answer) throws Exception;
    }

    private final LocalAnswerProvider localProvider;
    private final WebEvidenceAnswerProvider webProvider;
    private final HostedAnswerProvider hostedProvider;
    private final CanonicalTurnStore turnStore;

    public CandidateCoordinator(
            LocalAnswerProvider localProvider,
            WebEvidenceAnswerProvider webProvider,
            HostedAnswerProvider hostedProvider,
            CanonicalTurnStore turnStore) {
        this.localProvider = localProvider;
        this.webProvider = webProvider;
        this.hostedProvider = hostedProvider;
        this.turnStore = turnStore;
    }

    public CandidateSet create(
            String question,
            QwenEngine.StreamListener localListener) throws Exception {
        String q = question == null ? "" : question.trim();
        if (q.isEmpty()) {
            throw new IllegalArgumentException("السؤال فارغ");
        }

        String turnId = UUID.randomUUID().toString();

        AnswerCandidate local;
        try {
            local = localProvider.answer(q, localListener);
            if (local == null) {
                local = unavailableLocal("تعذر إنشاء الجواب المحلي");
            }
        } catch (Exception ex) {
            local = unavailableLocal(statusFrom(ex, "فشل الجواب المحلي"));
        }

        AnswerCandidate web;
        try {
            web = webProvider.answer(q);
            if (web == null) {
                web = AnswerCandidate.unavailable(
                        "web", AnswerCandidate.Kind.WEB,
                        "web-evidence", "تعذر إنشاء جواب البحث");
            }
        } catch (Exception ex) {
            web = AnswerCandidate.unavailable(
                    "web", AnswerCandidate.Kind.WEB,
                    "web-evidence", statusFrom(ex, "فشل البحث"));
        }

        AnswerCandidate hosted;
        try {
            hosted = hostedProvider.answer(q);
            if (hosted == null) {
                hosted = AnswerCandidate.unavailable(
                        "hosted", AnswerCandidate.Kind.HOSTED,
                        "none", "نموذج المتصفح غير متاح");
            }
        } catch (Exception ex) {
            hosted = AnswerCandidate.unavailable(
                    "hosted", AnswerCandidate.Kind.HOSTED,
                    "none", statusFrom(ex, "فشل نموذج المتصفح"));
        }

        if (local.available && turnStore != null) {
            turnStore.commit(turnId, q, local.answer);
        }

        return new CandidateSet(turnId, q, local, web, hosted);
    }

    private static AnswerCandidate unavailableLocal(String status) {
        return AnswerCandidate.unavailable(
                "local", AnswerCandidate.Kind.LOCAL,
                "qwen2.5-0.5b-int4", status);
    }

    private static String statusFrom(Throwable t, String fallback) {
        if (t == null || t.getMessage() == null ||
                t.getMessage().trim().isEmpty()) {
            return fallback;
        }
        return fallback + ": " + t.getMessage().trim();
    }
}
