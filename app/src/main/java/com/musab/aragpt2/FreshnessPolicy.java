package com.musab.aragpt2;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

public final class FreshnessPolicy {
    public enum Classification {
        STATIC_OR_GENERAL,
        CURRENT_OR_FRESH
    }

    private static final Set<String> EN_FRESH = new HashSet<>(Arrays.asList(
            "current", "currently", "today", "tonight", "now", "latest",
            "recent", "newest", "news", "price", "prices", "live",
            "president", "prime minister", "ceo", "weather", "score"
    ));

    private static final Set<String> AR_FRESH = new HashSet<>(Arrays.asList(
            "الحالي", "حالي", "حالياً", "حاليا", "اليوم", "الليلة",
            "الآن", "الان", "أحدث", "احدث", "آخر", "اخر", "جديد",
            "الأخبار", "الاخبار", "سعر", "أسعار", "اسعار", "مباشر",
            "الرئيس", "رئيس", "الطقس", "نتيجة"
    ));

    private FreshnessPolicy() {}

    public static Classification classify(String query) {
        String q = normalize(query);
        if (q.isEmpty()) return Classification.STATIC_OR_GENERAL;

        for (String cue : EN_FRESH) {
            if (containsPhrase(q, cue)) return Classification.CURRENT_OR_FRESH;
        }
        for (String cue : AR_FRESH) {
            if (containsPhrase(q, cue)) return Classification.CURRENT_OR_FRESH;
        }

        if (q.matches(".*\\b(20[2-9][0-9])\\b.*")
                && (q.contains("this year") || q.contains("هذه السنة")
                || q.contains("هالسنة") || q.contains("السنة الحالية"))) {
            return Classification.CURRENT_OR_FRESH;
        }

        return Classification.STATIC_OR_GENERAL;
    }

    public static boolean requiresFreshEvidence(String query) {
        return classify(query) == Classification.CURRENT_OR_FRESH;
    }

    private static boolean containsPhrase(String normalized, String cue) {
        String c = normalize(cue);
        if (c.indexOf(' ') >= 0) return normalized.contains(c);
        for (String token : normalized.split("\\s+")) {
            if (token.equals(c)) return true;
        }
        return false;
    }

    private static String normalize(String value) {
        if (value == null) return "";
        return value.toLowerCase(Locale.ROOT)
                .replaceAll("[ًٌٍَُِّْـ]", "")
                .replaceAll("[\\p{Punct}\\p{S}،؛؟]+", " ")
                .replaceAll("\\s+", " ")
                .trim();
    }
}
