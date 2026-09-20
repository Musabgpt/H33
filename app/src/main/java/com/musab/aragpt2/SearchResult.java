package com.musab.aragpt2;

import java.net.URI;
import java.util.Locale;

public final class SearchResult {
    public final String title;
    public final String url;
    public final String snippet;
    public final String host;
    public final double relevance;

    public SearchResult(String title, String url, String snippet) {
        this(title, url, snippet, safeHost(url), 0.0);
    }

    public SearchResult(String title, String url, String snippet,
                        String host, double relevance) {
        this.title = clean(title);
        this.url = clean(url);
        this.snippet = clean(snippet);
        this.host = clean(host).toLowerCase(Locale.ROOT);
        this.relevance = relevance;
    }

    public SearchResult withRelevance(double score) {
        return new SearchResult(title, url, snippet, host, score);
    }

    static String safeHost(String value) {
        String u = clean(value);
        if (u.isEmpty()) return "";
        try {
            URI uri = URI.create(u);
            String scheme = uri.getScheme();
            if (scheme == null ||
                    (!"http".equalsIgnoreCase(scheme) && !"https".equalsIgnoreCase(scheme))) {
                return "";
            }
            String h = uri.getHost();
            if (h == null) return "";
            h = h.toLowerCase(Locale.ROOT);
            if (h.startsWith("www.")) h = h.substring(4);
            return h;
        } catch (Exception ignored) {
            return "";
        }
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
