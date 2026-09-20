package com.musab.aragpt2;

public final class UnavailableHostedAnswerProvider
        implements HostedAnswerProvider {
    private final String status;

    public UnavailableHostedAnswerProvider() {
        this("نموذج المتصفح غير مهيأ");
    }

    public UnavailableHostedAnswerProvider(String status) {
        this.status = status == null ? "غير متاح" : status.trim();
    }

    @Override
    public AnswerCandidate answer(String question) {
        return AnswerCandidate.unavailable(
                "hosted",
                AnswerCandidate.Kind.HOSTED,
                "none",
                status
        );
    }
}
