package com.musab.aragpt2;

import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;

public final class CandidateCoordinator {
    private final LocalAnswerProvider localProvider;
    private final WebEvidenceAnswerProvider webProvider;
    private final HostedAnswerProvider hostedProvider;

    public CandidateCoordinator(
            LocalAnswerProvider localProvider,
            WebEvidenceAnswerProvider webProvider,
            HostedAnswerProvider hostedProvider) {
        this.localProvider = localProvider;
        this.webProvider = webProvider;
        this.hostedProvider = hostedProvider;
    }

    public CandidateSet create(
            String question,
            LocalInferenceEngine.StreamListener localListener) throws Exception {
        String q = question == null ? "" : question.trim();
        if (q.isEmpty()) {
            throw new IllegalArgumentException("السؤال فارغ");
        }

        String turnId = UUID.randomUUID().toString();

        FutureTask<AnswerCandidate> hostedTask =
                new FutureTask<>(() -> resolveHosted(q));
        Thread hostedThread =
                new Thread(hostedTask, "h33-hosted-candidate");
        hostedThread.setDaemon(true);
        hostedThread.start();

        AnswerCandidate local;
        try {
            local = localProvider.answer(q, localListener);
            if (local == null) {
                local = unavailableLocal("تعذر إنشاء الجواب المحلي");
            }
        } catch (Exception ex) {
            local = unavailableLocal(statusFrom(ex, "فشل الجواب المحلي"));
        }

        AnswerCandidate web = resolveWeb(q);

        AnswerCandidate hosted;
        try {
            hosted = hostedTask.get();
        } catch (InterruptedException ex) {
            hostedThread.interrupt();
            Thread.currentThread().interrupt();
            hosted = AnswerCandidate.unavailable(
                    "hosted", AnswerCandidate.Kind.HOSTED,
                    "none", "تم إيقاف جواب Google");
        } catch (ExecutionException ex) {
            hosted = AnswerCandidate.unavailable(
                    "hosted", AnswerCandidate.Kind.HOSTED,
                    "none", statusFrom(ex.getCause(), "فشل نموذج المتصفح"));
        }


        return new CandidateSet(turnId, q, local, web, hosted);
    }

    private AnswerCandidate resolveWeb(String question) {
        try {
            AnswerCandidate web = webProvider.answer(question);
            if (web != null) return web;
            return AnswerCandidate.unavailable(
                    "web", AnswerCandidate.Kind.WEB,
                    "web-evidence", "تعذر إنشاء جواب البحث");
        } catch (Exception ex) {
            return AnswerCandidate.unavailable(
                    "web", AnswerCandidate.Kind.WEB,
                    "web-evidence", statusFrom(ex, "فشل البحث"));
        }
    }

    private AnswerCandidate resolveHosted(String question) {
        try {
            AnswerCandidate hosted = hostedProvider.answer(question);
            if (hosted != null) return hosted;
            return AnswerCandidate.unavailable(
                    "hosted", AnswerCandidate.Kind.HOSTED,
                    "none", "نموذج المتصفح غير متاح");
        } catch (Exception ex) {
            return AnswerCandidate.unavailable(
                    "hosted", AnswerCandidate.Kind.HOSTED,
                    "none", statusFrom(ex, "فشل نموذج المتصفح"));
        }
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
