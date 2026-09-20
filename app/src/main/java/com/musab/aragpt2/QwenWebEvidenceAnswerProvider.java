package com.musab.aragpt2;

public final class QwenWebEvidenceAnswerProvider
        implements WebEvidenceAnswerProvider {
    private final QwenEngine engine;
    private final int maxResults;
    private final int maxNewTokens;

    public QwenWebEvidenceAnswerProvider(
            QwenEngine engine, int maxResults, int maxNewTokens) {
        this.engine = engine;
        this.maxResults = Math.max(1, maxResults);
        this.maxNewTokens = Math.max(16, maxNewTokens);
    }

    @Override
    public AnswerCandidate answer(String question) throws Exception {
        WebSearchClient.WebPayload payload =
                WebSearchClient.resolve(question, maxResults);

        if (payload == null || !payload.isUsable()) {
            return AnswerCandidate.unavailable(
                    "web", AnswerCandidate.Kind.WEB,
                    "web-evidence", "لم أجد أدلة ويب مرتبطة بالسؤال");
        }

        String answer = engine.generateCandidate(
                question,
                maxNewTokens,
                payload.context,
                true,
                null
        );

        if (answer == null || answer.trim().isEmpty()) {
            return AnswerCandidate.unavailable(
                    "web", AnswerCandidate.Kind.WEB,
                    "web-evidence", "تعذر تكوين جواب من أدلة الويب");
        }

        return AnswerCandidate.available(
                "web",
                AnswerCandidate.Kind.WEB,
                "web-evidence",
                answer,
                payload.results,
                "مبني على " + payload.sourceCount + " مصدر"
        );
    }
}
