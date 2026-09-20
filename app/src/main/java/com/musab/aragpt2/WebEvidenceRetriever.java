package com.musab.aragpt2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public final class WebEvidenceRetriever {
    private final int maxResults;

    public WebEvidenceRetriever(int maxResults) {
        this.maxResults = Math.max(1, maxResults);
    }

    public WebSearchClient.WebPayload retrieve(
            String originalQuestion) throws Exception {
        String original = clean(originalQuestion);
        if (original.isEmpty()) {
            return WebSearchClient.payloadFromAccepted(
                    Collections.emptyList(), false);
        }

        if (WebSearchClient.containsUrl(original)) {
            return WebSearchClient.resolve(original, maxResults);
        }

        SearchQueryPlan plan =
                SearchQueryPlanner.plan(original);
        int rawLimit = Math.max(maxResults * 3, 8);

        List<SearchResult> accepted = collect(
                plan, rawLimit, Provider.DDG_HTML);

        if (accepted.isEmpty()) {
            accepted = collect(plan, rawLimit, Provider.BING_RSS);
        }

        if (accepted.isEmpty()) {
            accepted = collect(plan, rawLimit, Provider.DDG_INSTANT);
        }

        if (accepted.isEmpty()) {
            return WebSearchClient.payloadFromAccepted(
                    Collections.emptyList(), false);
        }

        accepted.sort(
                Comparator.comparingDouble(
                        (SearchResult r) -> r.relevance).reversed());

        if (accepted.size() > maxResults) {
            accepted = new ArrayList<>(
                    accepted.subList(0, maxResults));
        }

        return WebSearchClient.payloadFromAccepted(accepted, false);
    }

    private List<SearchResult> collect(
            SearchQueryPlan plan,
            int rawLimit,
            Provider provider) {
        Map<String, SearchResult> merged = new LinkedHashMap<>();

        for (String query : plan.allQueries()) {
            List<SearchResult> raw;
            try {
                raw = fetch(provider, query, rawLimit);
            } catch (Exception ignored) {
                continue;
            }

            SearchQualityGate.Result quality =
                    SearchQualityGate.filter(
                            query, raw, Math.max(maxResults * 2, maxResults));

            for (SearchResult result : quality.accepted) {
                String key = dedupeKey(result);
                SearchResult existing = merged.get(key);
                if (existing == null
                        || result.relevance > existing.relevance) {
                    merged.put(key, result);
                }
            }
        }

        return new ArrayList<>(merged.values());
    }

    private static List<SearchResult> fetch(
            Provider provider,
            String query,
            int rawLimit) throws Exception {
        switch (provider) {
            case DDG_HTML:
                return WebSearchClient.searchDuckDuckGoHtml(
                        query, rawLimit);
            case BING_RSS:
                return WebSearchClient.searchBingRss(
                        query, rawLimit);
            case DDG_INSTANT:
                return WebSearchClient.searchDuckDuckGoInstant(
                        query, rawLimit);
            default:
                return Collections.emptyList();
        }
    }

    private static String dedupeKey(SearchResult result) {
        String host = clean(result.host).toLowerCase(Locale.ROOT);
        if (!host.isEmpty()) return "host:" + host;
        String url = clean(result.url).toLowerCase(Locale.ROOT);
        if (!url.isEmpty()) return "url:" + url;
        return "title:" + clean(result.title).toLowerCase(Locale.ROOT);
    }

    private enum Provider {
        DDG_HTML,
        BING_RSS,
        DDG_INSTANT
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
