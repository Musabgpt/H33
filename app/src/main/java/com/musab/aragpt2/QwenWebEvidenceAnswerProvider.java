package com.musab.aragpt2;

public final class QwenWebEvidenceAnswerProvider
        implements WebEvidenceAnswerProvider {
    private final LocalInferenceEngine inference;
    private final WebEvidenceRetriever retriever;
    private final TranslationBridge translation;
    private final int maxNewTokens;

    public QwenWebEvidenceAnswerProvider(
            LocalInferenceEngine inference,
            WebEvidenceRetriever retriever,
            TranslationBridge translation,
            int maxNewTokens) {
        if (inference == null) throw new IllegalArgumentException("inference required");
        if (retriever == null) throw new IllegalArgumentException("retriever required");
        if (translation == null) throw new IllegalArgumentException("translation required");
        this.inference = inference;
        this.retriever = retriever;
        this.translation = translation;
        this.maxNewTokens = Math.max(16, maxNewTokens);
    }

    @Override
    public AnswerCandidate answer(String question) throws Exception {
        String original = question == null ? "" : question.trim();
        if (original.isEmpty()) {
            return unavailable("السؤال فارغ");
        }

        String englishQuery = "";
        if (PivotingLocalAnswerProvider.containsArabic(original)) {
            TranslationBridge.Result translated =
                    translation.arabicToEnglish(original);
            if (translated.translated) {
                englishQuery = translated.text;
            }
        }

        WebSearchClient.WebPayload payload =
                retriever.retrieve(original, englishQuery);

        if (payload == null || !payload.isUsable()
                || payload.results.isEmpty()) {
            return unavailable("لم أجد أدلة ويب مرتبطة بالسؤال");
        }

        String answer = inference.generate(
                original,
                maxNewTokens,
                payload.context,
                true,
                null
        );

        if (answer == null || answer.trim().isEmpty()) {
            return unavailable("تعذر تكوين جواب من أدلة الويب");
        }

        boolean fresh = FreshnessPolicy.requiresFreshEvidence(original);
        String status = "مبني على " + payload.sourceCount + " مصدر";
        if (fresh) status += " • سؤال حديث/متغير";

        return AnswerCandidate.available(
                "web",
                AnswerCandidate.Kind.WEB,
                "web-evidence",
                answer,
                payload.results,
                status
        );
    }

    private static AnswerCandidate unavailable(String status) {
        return AnswerCandidate.unavailable(
                "web",
                AnswerCandidate.Kind.WEB,
                "web-evidence",
                status
        );
    }
}
