package com.musab.aragpt2;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

public final class SearchQueryPlanner {
    private static final Set<String> EN_NOISE = new HashSet<>(Arrays.asList(
            "how", "many", "much", "what", "which", "who", "whom", "whose",
            "where", "when", "why", "are", "is", "was", "were", "be", "been",
            "there", "the", "a", "an", "and", "or", "do", "does", "did",
            "can", "could", "would", "should", "their", "its", "of", "in",
            "on", "at", "to", "for", "from", "with", "by", "please", "tell"
    ));

    private static final Set<String> AR_NOISE = new HashSet<>(Arrays.asList(
            "كم", "عدد", "ما", "ماذا", "ماهي", "وماهي", "ماهو", "وماهو",
            "من", "هو", "هي", "هل", "وين", "أين", "اين", "متى", "ليش", "لماذا",
            "في", "على", "الى", "إلى", "عن", "مع", "لو", "اذا", "إذا",
            "الحالي", "حالي", "حاليا", "حالياً", "اليوم", "الآن", "الان"
    ));

    private SearchQueryPlanner() {}

    public static SearchQueryPlan plan(String original, String englishTranslation) {
        String originalClean = clean(original);
        String englishClean = clean(englishTranslation);

        if (englishClean.equalsIgnoreCase(originalClean)) englishClean = "";

        return new SearchQueryPlan(
                originalClean,
                focused(originalClean),
                englishClean,
                focused(englishClean)
        );
    }

    static String focused(String query) {
        String clean = clean(query);
        if (clean.isEmpty()) return "";

        String normalized = clean
                .replaceAll("[ًٌٍَُِّْـ]", "")
                .replaceAll("[\\p{Punct}\\p{S}،؛؟]+", " ")
                .replaceAll("\\s+", " ")
                .trim();

        StringBuilder out = new StringBuilder();
        for (String raw : normalized.split("\\s+")) {
            if (raw.isEmpty()) continue;
            String lower = raw.toLowerCase(Locale.ROOT);
            String check = stripArabicConjunction(lower);
            if (EN_NOISE.contains(lower) || AR_NOISE.contains(lower)
                    || AR_NOISE.contains(check)) {
                continue;
            }
            if (out.length() > 0) out.append(' ');
            out.append(raw);
        }
        return out.toString().trim();
    }

    private static String stripArabicConjunction(String token) {
        if (token != null && token.length() > 2 && token.charAt(0) == 'و') {
            return token.substring(1);
        }
        return token == null ? "" : token;
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
