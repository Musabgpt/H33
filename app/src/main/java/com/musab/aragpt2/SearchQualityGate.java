package com.musab.aragpt2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public final class SearchQualityGate {
    private static final double MIN_RELEVANCE = 0.18;

    private static final Set<String> DENIED_HOST_FRAGMENTS =
            new HashSet<>(Arrays.asList(
                    "xnxx", "xvideos", "pornhub", "redtube", "youporn", "xhamster"
            ));

    private static final Set<String> DENIED_CONTENT_TOKENS =
            new HashSet<>(Arrays.asList(
                    "porn", "porno", "xxx", "sex", "adult", "nudes", "nude"
            ));

    private static final Set<String> AR_STOP =
            new HashSet<>(Arrays.asList(
                    "من", "هو", "هي", "ما", "ماذا", "هل", "في", "على", "الى", "إلى",
                    "الحالي", "حاليا", "حالياً", "اليوم", "الآن", "الان"
            ));

    private static final Set<String> EN_STOP =
            new HashSet<>(Arrays.asList(
                    "who", "is", "the", "a", "an", "current", "today", "now", "of", "in"
            ));

    private SearchQualityGate() {}

    public static final class Result {
        public final List<SearchResult> accepted;
        public final List<SearchResult> rejected;

        Result(List<SearchResult> accepted, List<SearchResult> rejected) {
            this.accepted = Collections.unmodifiableList(accepted);
            this.rejected = Collections.unmodifiableList(rejected);
        }
    }

    public static Result filter(String query, List<SearchResult> raw, int maxResults) {
        if (maxResults <= 0 || raw == null || raw.isEmpty()) {
            return new Result(Collections.emptyList(), Collections.emptyList());
        }

        Set<String> queryTokens = significantTokens(query);
        List<SearchResult> rejected = new ArrayList<>();
        Map<String, SearchResult> bestByHost = new LinkedHashMap<>();

        for (SearchResult source : raw) {
            if (source == null || source.host.isEmpty() || isDenied(source)) {
                if (source != null) rejected.add(source);
                continue;
            }

            double titleCoverage = coverage(queryTokens, tokens(source.title));
            double bodyCoverage = coverage(queryTokens, tokens(source.snippet));
            Set<String> combined = tokens(source.title + " " + source.snippet);
            int matches = matchedCount(queryTokens, combined);
            double score = Math.min(1.0, titleCoverage * 0.65 + bodyCoverage * 0.35);

            if (queryTokens.isEmpty() || matches == 0 || score < MIN_RELEVANCE) {
                rejected.add(source);
                continue;
            }

            SearchResult scored = source.withRelevance(score);
            SearchResult previous = bestByHost.get(scored.host);
            if (previous == null || scored.relevance > previous.relevance) {
                if (previous != null) rejected.add(previous);
                bestByHost.put(scored.host, scored);
            } else {
                rejected.add(scored);
            }
        }

        ArrayList<SearchResult> accepted = new ArrayList<>(bestByHost.values());
        accepted.sort(Comparator.comparingDouble((SearchResult r) -> r.relevance).reversed());

        if (accepted.size() > maxResults) {
            rejected.addAll(accepted.subList(maxResults, accepted.size()));
            accepted = new ArrayList<>(accepted.subList(0, maxResults));
        }

        return new Result(accepted, rejected);
    }

    private static boolean isDenied(SearchResult r) {
        String host = r.host.toLowerCase(Locale.ROOT);
        for (String denied : DENIED_HOST_FRAGMENTS) {
            if (host.contains(denied)) return true;
        }

        String content = normalize(r.title + " " + r.snippet);
        Set<String> contentTokens = tokens(content);
        int bad = 0;
        for (String token : DENIED_CONTENT_TOKENS) {
            if (contentTokens.contains(token)) bad++;
        }
        return bad >= 2;
    }

    private static double coverage(Set<String> query, Set<String> candidate) {
        if (query.isEmpty() || candidate.isEmpty()) return 0.0;
        return (double) matchedCount(query, candidate) / query.size();
    }

    private static int matchedCount(Set<String> query, Set<String> candidate) {
        int count = 0;
        for (String q : query) {
            boolean found = false;
            for (String c : candidate) {
                if (equivalentToken(q, c)) {
                    found = true;
                    break;
                }
            }
            if (found) count++;
        }
        return count;
    }

    private static boolean equivalentToken(String a, String b) {
        if (a.equals(b)) return true;
        String x = stripArabicArticle(a);
        String y = stripArabicArticle(b);
        if (x.equals(y)) return true;

        if (isArabicWord(x) && isArabicWord(y) && x.length() >= 4 && y.length() >= 4) {
            if (Math.abs(x.length() - y.length()) <= 1 &&
                    x.regionMatches(0, y, 0, Math.min(x.length(), y.length()) - 1)) {
                return true;
            }
        }
        return false;
    }

    private static String stripArabicArticle(String token) {
        if (token.startsWith("ال") && token.length() > 4) return token.substring(2);
        return token;
    }

    private static boolean isArabicWord(String token) {
        return token.matches(".*[\\u0600-\\u06FF].*");
    }

    private static Set<String> significantTokens(String text) {
        Set<String> all = tokens(text);
        all.removeIf(t -> AR_STOP.contains(t) || EN_STOP.contains(t));
        return all;
    }

    private static Set<String> tokens(String text) {
        String normalized = normalize(text);
        if (normalized.isEmpty()) return new HashSet<>();

        HashSet<String> out = new HashSet<>();
        for (String token : normalized.split("\\s+")) {
            if (token.length() > 1) out.add(token);
        }
        return out;
    }

    private static String normalize(String value) {
        String n = value == null ? "" : value.toLowerCase(Locale.ROOT);
        n = n.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا');
        n = n.replace('ى', 'ي');
        n = n.replaceAll("[ًٌٍَُِّْـ]", "");
        n = n.replaceAll("[^\\p{L}\\p{N}]+", " ");
        return n.replaceAll("\\s+", " ").trim();
    }
}
