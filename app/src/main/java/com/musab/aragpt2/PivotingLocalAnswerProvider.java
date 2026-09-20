package com.musab.aragpt2;

import java.util.Collections;

public final class PivotingLocalAnswerProvider implements LocalAnswerProvider {
    private final LocalInferenceEngine inference;
    private final TranslationBridge translation;
    private final int maxNewTokens;

    public PivotingLocalAnswerProvider(
            LocalInferenceEngine inference,
            TranslationBridge translation,
            int maxNewTokens) {
        if (inference == null) throw new IllegalArgumentException("inference required");
        if (translation == null) throw new IllegalArgumentException("translation required");
        this.inference = inference;
        this.translation = translation;
        this.maxNewTokens = Math.max(16, maxNewTokens);
    }

    @Override
    public AnswerCandidate answer(
            String question,
            LocalInferenceEngine.StreamListener listener) throws Exception {
        String original = clean(question);
        if (original.isEmpty()) {
            return AnswerCandidate.unavailable(
                    "local", AnswerCandidate.Kind.LOCAL,
                    "qwen2.5-0.5b-int4", "السؤال فارغ"
            );
        }

        if (!containsArabic(original)) {
            String answer = inference.generate(
                    original, maxNewTokens, "", false, listener);
            return candidate(answer, "local-direct-en", "محلي • English direct");
        }

        TranslationBridge.Result toEnglish = translation.arabicToEnglish(original);
        if (!toEnglish.translated) {
            String answer = inference.generate(
                    original, maxNewTokens, "", false, listener);
            return candidate(
                    answer,
                    "local-direct-ar",
                    "محلي • Arabic direct • translation fallback"
            );
        }

        String englishAnswer = inference.generate(
                toEnglish.text,
                maxNewTokens,
                "",
                false,
                null
        );

        if (clean(englishAnswer).isEmpty()) {
            return AnswerCandidate.unavailable(
                    "local", AnswerCandidate.Kind.LOCAL,
                    "local-pivot-en", "لم ينتج Qwen جوابًا"
            );
        }

        TranslationBridge.Result toArabic = translation.englishToArabic(englishAnswer);
        if (toArabic.translated) {
            if (listener != null) listener.onUpdate(toArabic.text);
            return candidate(
                    toArabic.text,
                    "local-pivot-en",
                    "محلي • Arabic→English reasoning→Arabic"
            );
        }

        if (listener != null) listener.onUpdate(englishAnswer);
        return candidate(
                englishAnswer,
                "local-pivot-en",
                "محلي • English reasoning • Arabic display fallback"
        );
    }

    private static AnswerCandidate candidate(
            String answer, String provider, String status) {
        String clean = clean(answer);
        if (clean.isEmpty()) {
            return AnswerCandidate.unavailable(
                    "local", AnswerCandidate.Kind.LOCAL,
                    provider, "لم ينتج Qwen جوابًا"
            );
        }
        return AnswerCandidate.available(
                "local", AnswerCandidate.Kind.LOCAL,
                provider, clean, Collections.emptyList(), status
        );
    }

    static boolean containsArabic(String text) {
        if (text == null) return false;
        for (int i = 0; i < text.length(); i++) {
            Character.UnicodeBlock block = Character.UnicodeBlock.of(text.charAt(i));
            if (block == Character.UnicodeBlock.ARABIC
                    || block == Character.UnicodeBlock.ARABIC_PRESENTATION_FORMS_A
                    || block == Character.UnicodeBlock.ARABIC_PRESENTATION_FORMS_B) {
                return true;
            }
        }
        return false;
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
