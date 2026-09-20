package com.musab.aragpt2;

public interface TranslationBridge extends AutoCloseable {
    final class Result {
        public final String text;
        public final boolean translated;
        public final String status;

        private Result(String text, boolean translated, String status) {
            this.text = text == null ? "" : text.trim();
            this.translated = translated && !this.text.isEmpty();
            this.status = status == null ? "" : status.trim();
        }

        public static Result success(String text) {
            return new Result(text, true, "translated");
        }

        public static Result fallback(String original, String status) {
            return new Result(original, false, status);
        }
    }

    Result arabicToEnglish(String text);

    Result englishToArabic(String text);

    @Override
    void close();
}
