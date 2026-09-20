package com.musab.aragpt2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public final class SearchQualityGate {
    private static final Set<String> DENIED_HOST_FRAGMENTS = new HashSet<>(Arrays.asList(
            "xnxx", "xvideos", "pornhub", "redtube", "youporn", "xhamster"
    ));

    private static final Set<String> AR_STOP = new HashSet<>(Arrays.asList(
            "من", "هو", "هي", "ما", "ماذا", "هل", "في", "على", "الى", "إلى",
            "الحالي", "حاليا", "حالياً", "اليوم", "الآن", "الان"
    ));

    private static final Set<String> EN_STOP = new HashSet<>(Arrays.asList(
            "who", "is", "the", "a", "an", "current", "today", "now", "of", "in"
    ));

    private SearchQualityGate() {}

    public static final class Result {
        public final List<SearchResult> accepted;
        public final List<SearchResult> rejected;

        Result(List<SearchResult> accepted, List<SearchResult> rejected) {
            this.accepted = Collections.unmodifiableList(new ArrayList<>(accepted));
            this.rejected = Collections.unmodifiableList(new ArrayList<>(rejected));
        }
    }

    public static Result filter(String query, List<SearchResult> input, int maxResults) {
        if (input == null || input.isEmpty() || maxResults <= 0) {
            return new Result(Collections.emptyList(), Collections.emptyList());
        }

        Set<String> queryTokens = significantTokens(query);
        ArrayList<SearchResult> rejected = new ArrayList<>();
        Map<String, SearchResult> bestByHost = new HashMap<>();

        for (SearchResult raw : input) {
            if (raw == null || raw.host.isEmpty() || isDeniedHost(raw.host)) {
                if (raw != null) rejected.add(raw);
                continue;
            }

            Set<String> titleTokens = tokens(raw.title);
            Set<String> bodyTokens = tokens(raw.snippet);
            Set<String> combined = new HashSet<>(titleTokens);
            combined.addAll(bodyTokens);

            int matched = matchedCount(queryTokens, combined);
            double titleCoverage = coverage(queryTokens, titleTokens);
            double bodyCoverage = coverage(queryTokens, bodyTokens);
            double score = Math.min(1.0, titleCoverage * 0.65 + bodyCoverage * 0.35);

            if (queryTokens.isEmpty() || matched == 0 || score < 0.16) {
                rejected.add(raw.withRelevance(score));
                continue;
            }

            SearchResult scored = raw.withRelevance(score);
            SearchResult existing = bestByHost.get(scored.host);
            if (existing == null || scored.relevance > existing.relevance) {
                if (existing != null) rejected.add(existing);
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

    private static boolean isDeniedHost(String host) {
        String h = host.toLowerCase(Locale.ROOT);
        for (String fragment : DENIED_HOST_FRAGMENTS) {
            if (h.contains(fragment)) return true;
        }
        return false;
    }

    private static int matchedCount(Set<String> query, Set<String> candidate) {
        int n = 0;
        for (String token : query) {
            if (candidate.contains(token)) n++;
        }
        return n;
    }

    private static double coverage(Set<String> query, Set<String> candidate) {
        if (query.isEmpty()) return 0.0;
        return (double) matchedCount(query, candidate) / (double) query.size();
    }

    private static Set<String> significantTokens(String value) {
        Set<String> out = tokens(value);
        out.removeAll(AR_STOP);
        out.removeAll(EN_STOP);
        return out;
    }

    private static Set<String> tokens(String value) {
        HashSet<String> out = new HashSet<>();
        String normalized = normalize(value);
        if (normalized.isEmpty()) return out;

        for (String token : normalized.split("\\s+")) {
            if (token.length() > 1) out.add(token);
        }
        return out;
    }

    private static String normalize(String value) {
        if (value == null) return "";
        String s = value.toLowerCase(Locale.ROOT);
        s = s.replaceAll("[ًٌٍَُِّْـ]", "");
        s = s.replaceAll("[\\p{Punct}\\p{S}،؛؟]+", " ");
        s = s.replaceAll("\\s+", " ").trim();

        StringBuilder out = new StringBuilder();
        for (String token : s.split("\\s+")) {
            if (token.isEmpty()) continue;
            String t = token;
            if (t.startsWith("ال") && t.length() > 4) t = t.substring(2);
            if (t.endsWith("ية") && t.length() > 3) {
                t = t.substring(0, t.length() - 2) + "يا";
            }
            if (out.length() > 0) out.append(' ');
            out.append(t);
        }
        return out.toString();
    }
}
