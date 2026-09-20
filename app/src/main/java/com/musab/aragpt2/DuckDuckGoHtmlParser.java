package com.musab.aragpt2;

import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class DuckDuckGoHtmlParser {
    private static final Pattern RESULT_PATTERN = Pattern.compile(
            "(?is)<a[^>]*class=[\\\"'][^\\\"']*result__a[^\\\"']*[\\\"'][^>]*href=[\\\"']([^\\\"']+)[\\\"'][^>]*>(.*?)</a>"
    );
    private static final Pattern SNIPPET_PATTERN = Pattern.compile(
            "(?is)<(?:a|div)[^>]*class=[\\\"'][^\\\"']*result__snippet[^\\\"']*[\\\"'][^>]*>(.*?)</(?:a|div)>"
    );

    private DuckDuckGoHtmlParser() {}

    public static List<SearchResult> parse(String html, int maxResults) {
        if (html == null || html.trim().isEmpty() || maxResults <= 0) {
            return Collections.emptyList();
        }

        ArrayList<String> snippets = new ArrayList<>();
        Matcher snippetMatcher = SNIPPET_PATTERN.matcher(html);
        while (snippetMatcher.find()) {
            snippets.add(cleanHtml(snippetMatcher.group(1)));
        }

        ArrayList<SearchResult> out = new ArrayList<>();
        Matcher resultMatcher = RESULT_PATTERN.matcher(html);
        int index = 0;

        while (resultMatcher.find() && out.size() < maxResults) {
            String originalUrl = unwrapResultUrl(resultMatcher.group(1));
            String title = cleanHtml(resultMatcher.group(2));
            String snippet = index < snippets.size() ? snippets.get(index) : "";
            index++;

            if (title.isEmpty() || originalUrl.isEmpty()) continue;
            out.add(new SearchResult(title, originalUrl, snippet));
        }

        return out;
    }

    static String unwrapResultUrl(String value) {
        String raw = decodeEntities(value == null ? "" : value.trim());
        if (raw.isEmpty()) return "";
        if (raw.startsWith("//")) raw = "https:" + raw;

        try {
            URI uri = URI.create(raw);
            String scheme = lower(uri.getScheme());
            String host = lower(uri.getHost());

            if (("http".equals(scheme) || "https".equals(scheme))
                    && host != null
                    && !isDuckDuckGoHost(host)) {
                return raw;
            }

            if (!isDuckDuckGoHost(host) || !"/l/".equals(uri.getPath())) {
                return "";
            }

            String query = uri.getRawQuery();
            if (query == null || query.isEmpty()) return "";

            for (String part : query.split("&")) {
                int eq = part.indexOf('=');
                if (eq <= 0) continue;

                String key = part.substring(0, eq);
                if (!"uddg".equals(key)) continue;

                String decoded = URLDecoder.decode(
                        part.substring(eq + 1),
                        StandardCharsets.UTF_8.name()
                );
                URI target = URI.create(decoded);
                String targetScheme = lower(target.getScheme());
                String targetHost = lower(target.getHost());

                if (("http".equals(targetScheme) || "https".equals(targetScheme))
                        && targetHost != null
                        && !isDuckDuckGoHost(targetHost)) {
                    return decoded;
                }
            }
        } catch (Exception ignored) {
            return "";
        }

        return "";
    }

    private static boolean isDuckDuckGoHost(String host) {
        String h = lower(host);
        return "duckduckgo.com".equals(h)
                || h.endsWith(".duckduckgo.com");
    }

    private static String cleanHtml(String value) {
        if (value == null) return "";
        String s = value.replaceAll("(?is)<[^>]+>", " ");
        s = decodeEntities(s);
        return s.replaceAll("\\s+", " ").trim();
    }

    private static String decodeEntities(String value) {
        if (value == null) return "";
        return value
                .replace("&amp;", "&")
                .replace("&quot;", "\"")
                .replace("&#39;", "'")
                .replace("&apos;", "'")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&nbsp;", " ");
    }

    private static String lower(String value) {
        return value == null ? "" : value.toLowerCase(Locale.ROOT);
    }
}
