package com.musab.aragpt2;

import java.util.Collections;

public final class DeepSeekCoderLocalAnswerProvider
        implements LocalAnswerProvider {
    private final LocalInferenceEngine inference;
    private final int maxNewTokens;

    public DeepSeekCoderLocalAnswerProvider(
            LocalInferenceEngine inference,
            int maxNewTokens) {
        if (inference == null) {
            throw new IllegalArgumentException("inference required");
        }
        this.inference = inference;
        this.maxNewTokens = Math.max(32, maxNewTokens);
    }

    @Override
    public AnswerCandidate answer(
            String question,
            LocalInferenceEngine.StreamListener listener) throws Exception {
        String q = question == null ? "" : question.trim();
        if (q.isEmpty()) {
            return AnswerCandidate.unavailable(
                    "local",
                    AnswerCandidate.Kind.LOCAL,
                    "deepseek-coder-1.3b-int4",
                    "السؤال فارغ");
        }

        String answer = inference.generate(
                q,
                maxNewTokens,
                "",
                false,
                listener
        );

        if (answer == null || answer.trim().isEmpty()) {
            return AnswerCandidate.unavailable(
                    "local",
                    AnswerCandidate.Kind.LOCAL,
                    "deepseek-coder-1.3b-int4",
                    "لم ينتج نموذج البرمجة جوابًا");
        }

        return AnswerCandidate.available(
                "local",
                AnswerCandidate.Kind.LOCAL,
                "deepseek-coder-1.3b-int4",
                answer,
                Collections.emptyList(),
                "محلي • Python coder • English output"
        );
    }
}
