package com.musab.aragpt2;

import java.util.Collections;

public final class QwenLocalAnswerProvider implements LocalAnswerProvider {
    private final QwenEngine engine;
    private final int maxNewTokens;

    public QwenLocalAnswerProvider(QwenEngine engine, int maxNewTokens) {
        this.engine = engine;
        this.maxNewTokens = Math.max(16, maxNewTokens);
    }

    @Override
    public AnswerCandidate answer(
            String question, QwenEngine.StreamListener listener) throws Exception {
        String answer = engine.generateCandidate(
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
