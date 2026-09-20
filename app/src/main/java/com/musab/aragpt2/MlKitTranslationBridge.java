package com.musab.aragpt2;

import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.common.model.DownloadConditions;
import com.google.mlkit.nl.translate.TranslateLanguage;
import com.google.mlkit.nl.translate.Translation;
import com.google.mlkit.nl.translate.Translator;
import com.google.mlkit.nl.translate.TranslatorOptions;

import java.util.concurrent.TimeUnit;

public final class MlKitTranslationBridge implements TranslationBridge {
    private static final long MODEL_TIMEOUT_SECONDS = 45L;
    private static final long TRANSLATE_TIMEOUT_SECONDS = 20L;

    private final Translator arToEn;
    private final Translator enToAr;
    private final DownloadConditions downloadConditions;

    public MlKitTranslationBridge() {
        this.arToEn = Translation.getClient(
                new TranslatorOptions.Builder()
                        .setSourceLanguage(TranslateLanguage.ARABIC)
                        .setTargetLanguage(TranslateLanguage.ENGLISH)
                        .build()
        );
        this.enToAr = Translation.getClient(
                new TranslatorOptions.Builder()
                        .setSourceLanguage(TranslateLanguage.ENGLISH)
                        .setTargetLanguage(TranslateLanguage.ARABIC)
                        .build()
        );
        this.downloadConditions = new DownloadConditions.Builder()
                .requireWifi()
                .build();
    }

    @Override
    public Result arabicToEnglish(String text) {
        return translate(arToEn, text, "ar-en");
    }

    @Override
    public Result englishToArabic(String text) {
        return translate(enToAr, text, "en-ar");
    }

    private Result translate(Translator translator, String text, String direction) {
        String input = clean(text);
        if (input.isEmpty()) return Result.fallback("", direction + ":empty");

        try {
            Tasks.await(
                    translator.downloadModelIfNeeded(downloadConditions),
                    MODEL_TIMEOUT_SECONDS,
                    TimeUnit.SECONDS
            );
            String translated = Tasks.await(
                    translator.translate(input),
                    TRANSLATE_TIMEOUT_SECONDS,
                    TimeUnit.SECONDS
            );
            if (clean(translated).isEmpty()) {
                return Result.fallback(input, direction + ":empty-result");
            }
            return Result.success(translated);
        } catch (Exception ex) {
            return Result.fallback(input, direction + ":" + safeType(ex));
        }
    }

    @Override
    public void close() {
        try { arToEn.close(); } catch (Exception ignored) {}
        try { enToAr.close(); } catch (Exception ignored) {}
    }

    private static String safeType(Throwable t) {
        if (t == null) return "failed";
        Throwable cause = t.getCause() == null ? t : t.getCause();
        String name = cause.getClass().getSimpleName();
        return name == null || name.trim().isEmpty() ? "failed" : name;
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
