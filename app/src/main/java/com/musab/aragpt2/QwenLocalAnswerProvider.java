package com.musab.aragpt2;

import java.util.Collections;

public final class QwenLocalAnswerProvider implements LocalAnswerProvider {
    private final LocalInferenceEngine inference;
    private final int maxNewTokens;

    public QwenLocalAnswerProvider(LocalInferenceEngine inference, int maxNewTokens) {
        if (inference == null) throw new IllegalArgumentException("inference required");
        this.inference = inference;
        this.maxNewTokens = Math.max(16, maxNewTokens);
    }

    @Override
    public AnswerCandidate answer(
            String question, LocalInferenceEngine.StreamListener listener) throws Exception {
        String answer = inference.generate(
                question, maxNewTokens, "", false, listener);
        if (answer == null || answer.trim().isEmpty()) {
            return AnswerCandidate.unavailable(
                    "local", AnswerCandidate.Kind.LOCAL,
                    "qwen2.5-0.5b-int4", "لم ينتج Qwen جوابًا");
        }
        return AnswerCandidate.available(
                "local", AnswerCandidate.Kind.LOCAL,
                "qwen2.5-0.5b-int4", answer,
                Collections.emptyList(), "محلي");
    }
}
