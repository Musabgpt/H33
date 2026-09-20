package com.musab.aragpt2;

public final class ExtractiveWebEvidenceAnswerProvider
        implements WebEvidenceAnswerProvider {
    private final WebEvidenceRetriever retriever;

    public ExtractiveWebEvidenceAnswerProvider(
            WebEvidenceRetriever retriever) {
        if (retriever == null) {
            throw new IllegalArgumentException("retriever required");
        }
        this.retriever = retriever;
    }

    @Override
    public AnswerCandidate answer(String question) throws Exception {
        WebSearchClient.WebPayload payload =
                retriever.retrieve(question);

        if (payload == null
                || !payload.isUsable()
                || payload.results.isEmpty()) {
            return AnswerCandidate.unavailable(
                    "web",
                    AnswerCandidate.Kind.WEB,
                    "web-evidence",
                    "لم أجد أدلة ويب مرتبطة بالسؤال");
        }

        StringBuilder answer = new StringBuilder();
        int index = 1;
        for (SearchResult result : payload.results) {
            if (answer.length() > 0) answer.append("\n\n");
            answer.append("[").append(index++).append("] ");
            if (!result.title.isEmpty()) {
                answer.append(result.title);
            } else if (!result.host.isEmpty()) {
                answer.append(result.host);
            } else {
                answer.append("Web source");
            }

            if (!result.snippet.isEmpty()) {
                answer.append("\n").append(result.snippet);
            }
        }

        return AnswerCandidate.available(
                "web",
                AnswerCandidate.Kind.WEB,
                "web-evidence-extractive",
                answer.toString(),
                payload.results,
                "مقتطفات من " + payload.sourceCount + " مصدر • بلا توليد محلي"
        );
    }
}
