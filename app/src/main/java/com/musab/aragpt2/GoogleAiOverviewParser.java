package com.musab.aragpt2;

import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public final class GoogleAiOverviewParser {
    private GoogleAiOverviewParser() {}

    public static final class Result {
        public final boolean found;
        public final boolean grounded;
        public final String text;
        public final List<SearchResult> sources;

        Result(boolean found, String text, List<SearchResult> sources) {
            this.text = clean(text);
            this.found = found && !this.text.isEmpty();
            this.grounded = this.found;
            this.sources = Collections.unmodifiableList(
                    new ArrayList<>(sources == null
                            ? Collections.emptyList()
                            : sources));
        }
    }

    public static Result parseEvaluateJavascriptResult(String raw) {
        String value = clean(raw);
        if (value.isEmpty() || "null".equals(value)) {
            return unavailable();
        }

        try {
            Object decoded = new JSONTokener(value).nextValue();
            if (decoded instanceof String) {
                return parseJsonPayload((String) decoded);
            }
            if (decoded instanceof JSONObject) {
                return parseObject((JSONObject) decoded);
            }
            return unavailable();
        } catch (Exception ex) {
            throw new IllegalStateException(
                    "تعذر قراءة نتيجة Google AI Overview", ex);
        }
    }

    public static Result parseJsonPayload(String json) {
        String value = clean(json);
        if (value.isEmpty()) return unavailable();

        try {
            return parseObject(new JSONObject(value));
        } catch (Exception ex) {
            throw new IllegalStateException(
                    "تعذر تحليل Google AI Overview", ex);
        }
    }

    private static Result parseObject(JSONObject root) {
        boolean found = root.optBoolean("found", false);
        String text = clean(root.optString("text", ""));
        if (!found || text.isEmpty()) return unavailable();

        ArrayList<SearchResult> sources = new ArrayList<>();
        Set<String> seenUrls = new HashSet<>();
        JSONArray array = root.optJSONArray("sources");

        if (array != null) {
            for (int i = 0; i < array.length(); i++) {
                JSONObject source = array.optJSONObject(i);
                if (source == null) continue;

                String url = externalSourceUrl(
                        clean(source.optString("url", "")));
                if (!isHttpUrl(url)) continue;

                String key = normalizeUrl(url);
                if (key.isEmpty() || !seenUrls.add(key)) continue;

                String title = clean(source.optString("title", ""));
                if (title.isEmpty()) title = hostOf(url);

                sources.add(new SearchResult(title, url, ""));
            }
        }

        return new Result(true, text, sources);
    }

    private static Result unavailable() {
        return new Result(false, "", Collections.emptyList());
    }

    private static String externalSourceUrl(String value) {
        String raw = clean(value);
        if (!isHttpUrl(raw)) return "";

        try {
            URI uri = URI.create(raw);
            String host = clean(uri.getHost()).toLowerCase(Locale.ROOT);
            if (host.startsWith("www.")) host = host.substring(4);

            boolean googleHost = host.equals("google.com")
                    || host.endsWith(".google.com")
                    || host.equals("google.co.uk")
                    || host.endsWith(".google.co.uk");

            if (!googleHost) return raw;

            String path = clean(uri.getPath());
            if (!"/url".equals(path)) return "";

            String query = uri.getRawQuery();
            if (query == null || query.isEmpty()) return "";

            for (String part : query.split("&")) {
                int eq = part.indexOf('=');
                if (eq <= 0) continue;
                String name = part.substring(0, eq);
                if (!"q".equals(name) && !"url".equals(name)) continue;

                String encoded = part.substring(eq + 1);
                String target;
                try {
                    target = java.net.URLDecoder.decode(
                            encoded, StandardCharsets.UTF_8.name());
                } catch (Exception ignored) {
                    target = encoded;
                }

                if (isHttpUrl(target)) {
                    URI targetUri = URI.create(target);
                    String targetHost = clean(targetUri.getHost())
                            .toLowerCase(Locale.ROOT);
                    if (targetHost.startsWith("www.")) {
                        targetHost = targetHost.substring(4);
                    }
                    if (!targetHost.equals("google.com")
                            && !targetHost.endsWith(".google.com")
                            && !targetHost.equals("google.co.uk")
                            && !targetHost.endsWith(".google.co.uk")) {
                        return target;
                    }
                }
            }
        } catch (Exception ignored) {
            return "";
        }

        return "";
    }

    private static boolean isHttpUrl(String value) {
        try {
            URI uri = URI.create(value);
            String scheme = uri.getScheme();
            return ("http".equalsIgnoreCase(scheme)
                    || "https".equalsIgnoreCase(scheme))
                    && uri.getHost() != null;
        } catch (Exception ignored) {
            return false;
        }
    }

    private static String normalizeUrl(String value) {
        try {
            URI uri = URI.create(value);
            String scheme = clean(uri.getScheme()).toLowerCase(Locale.ROOT);
            String host = clean(uri.getHost()).toLowerCase(Locale.ROOT);
            if (host.startsWith("www.")) host = host.substring(4);
            String path = uri.getRawPath();
            if (path == null || path.isEmpty()) path = "/";
            String query = uri.getRawQuery();
            return scheme + "://" + host + path
                    + (query == null ? "" : "?" + query);
        } catch (Exception ignored) {
            return "";
        }
    }

    private static String hostOf(String value) {
        try {
            String host = URI.create(value).getHost();
            if (host == null) return "";
            host = host.toLowerCase(Locale.ROOT);
            if (host.startsWith("www.")) host = host.substring(4);
            return host;
        } catch (Exception ignored) {
            return "";
        }
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
