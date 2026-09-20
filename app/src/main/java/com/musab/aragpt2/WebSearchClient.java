package com.musab.aragpt2;

import android.util.Xml;

import org.json.JSONArray;
import org.json.JSONObject;
import org.xmlpull.v1.XmlPullParser;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class WebSearchClient {
    private static final int CONNECT_TIMEOUT_MS = 8000;
    private static final int READ_TIMEOUT_MS = 12000;
    private static final int MAX_PAGE_CHARS = 4200;

    private static final Pattern URL_PATTERN =
            Pattern.compile("https?://[^\\s<>\"]+", Pattern.CASE_INSENSITIVE);

    private WebSearchClient() {}

    public static final class WebPayload {
        public final String context;
        public final String sources;
        public final int sourceCount;
        public final boolean directUrl;

        WebPayload(String context, String sources, int sourceCount, boolean directUrl) {
            this.context = context == null ? "" : context;
            this.sources = sources == null ? "" : sources;
            this.sourceCount = sourceCount;
            this.directUrl = directUrl;
        }

        public boolean isUsable() {
            return !context.trim().isEmpty() && sourceCount > 0;
        }
    }

    public static boolean containsUrl(String question) {
        return question != null && URL_PATTERN.matcher(question).find();
    }

    public static boolean shouldAutoSearch(String question) {
        if (question == null) return false;
        if (containsUrl(question)) return true;

        String q = question.toLowerCase(Locale.ROOT).trim();

        String[] explicit = {
                "ابحث", "دورلي", "دور لي", "فتش", "بحث عن", "من النت", "بالنت",
                "على الانترنت", "على الإنترنت", "من الانترنت", "من الإنترنت",
                "search the web", "search online", "look up"
        };
        for (String cue : explicit) if (q.contains(cue)) return true;

        String[] freshness = {
                "اليوم", "الآن", "الان", "حاليا", "حالياً", "أحدث", "احدث",
                "آخر خبر", "آخر الأخبار", "اخر خبر", "اخر الاخبار",
                "سعر اليوم", "السعر الآن", "السعر الان", "ما الجديد",
                "latest", "current", "today", "news", "price now"
        };
        for (String cue : freshness) if (q.contains(cue)) return true;

        // Current office-holder / identity questions are too risky for stale local memory.
        String[] officeCues = {
                "من هو رئيس", "من هي رئيس", "مين رئيس", "من هو الملك", "من هي الملكة",
                "من هو رئيس الوزراء", "من هي رئيسة الوزراء", "من هو الوزير",
                "من هي الوزيرة", "من هو الحاكم", "من يحكم", "الرئيس الحالي",
                "رئيس سوريا", "رئيس أمريكا", "رئيس الولايات المتحدة",
                "who is the president", "prime minister", "current president",
                "current ceo", "who is the ceo"
        };
        for (String cue : officeCues) if (q.contains(cue)) return true;

        return false;
    }

    public static WebPayload resolve(String question, int maxResults) throws Exception {
        String q = question == null ? "" : question.trim();
        if (q.isEmpty()) return new WebPayload("", "", 0, false);

        String url = firstUrl(q);
        if (!url.isEmpty()) {
            return fetchDirectUrl(url);
        }

        return search(q, maxResults);
    }

    public static WebPayload search(String query, int maxResults) throws Exception {
        String q = query == null ? "" : query.trim();
        if (q.isEmpty()) return new WebPayload("", "", 0, false);

        int rawLimit = Math.max(maxResults, Math.min(30, maxResults * 3));
        List<SearchResult> results = new ArrayList<>();

        try {
            results = searchBingRss(q, rawLimit);
        } catch (Exception ignored) {
        }

        SearchQualityGate.Result quality =
                SearchQualityGate.filter(q, results, maxResults);

        if (quality.accepted.isEmpty()) {
            try {
                results = searchDuckDuckGo(q, rawLimit);
            } catch (Exception ignored) {
                results = new ArrayList<>();
            }
            quality = SearchQualityGate.filter(q, results, maxResults);
        }

        if (quality.accepted.isEmpty()) {
            return new WebPayload("", "", 0, false);
        }

        return payloadFromAccepted(quality.accepted, false);
    }

    private static WebPayload fetchDirectUrl(String urlText) throws Exception {
        SearchResult directCandidate = new SearchResult(urlText, urlText, urlText);
        SearchQualityGate.Result directQuality = SearchQualityGate.filter(
                urlText, java.util.Collections.singletonList(directCandidate), 1);
        if (directQuality.accepted.isEmpty()) {
            return new WebPayload("", "", 0, true);
        }

        URL url = new URL(urlText);
        HttpURLConnection conn = open(url, "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5");

        try {
            String contentType = conn.getContentType();
            String html = readLimited(conn.getInputStream(), 350_000);

            if (html.trim().isEmpty()) {
                return new WebPayload("", "", 0, true);
            }

            String title = extractTitle(html);
            String text;

            if (contentType != null && contentType.toLowerCase(Locale.ROOT).contains("text/plain")) {
                text = cleanWhitespace(html);
            } else {
                text = htmlToReadableText(html);
            }

            if (text.length() > MAX_PAGE_CHARS) {
                text = text.substring(0, MAX_PAGE_CHARS) + "…";
            }

            if (title.isEmpty()) title = url.getHost();

            String context = "[1] " + title + "\nURL: " + urlText + "\n" + text;
            String sources = "[1] " + title + "\n" + urlText;

            return new WebPayload(context.trim(), sources.trim(), 1, true);
        } finally {
            conn.disconnect();
        }
    }

    private static List<SearchResult> searchBingRss(String query, int maxResults) throws Exception {
        String encoded = URLEncoder.encode(query, StandardCharsets.UTF_8.name());
        URL url = new URL("https://www.bing.com/search?q=" + encoded + "&format=rss&setlang=ar");
        HttpURLConnection conn = open(url, "application/rss+xml,application/xml,text/xml,*/*;q=0.5");

        try (InputStream in = new BufferedInputStream(conn.getInputStream())) {
            XmlPullParser parser = Xml.newPullParser();
            parser.setInput(in, "UTF-8");

            List<SearchResult> results = new ArrayList<>();
            String currentTag = null;
            String title = "", link = "", description = "";
            boolean inItem = false;

            int event = parser.getEventType();
            while (event != XmlPullParser.END_DOCUMENT && results.size() < maxResults) {
                if (event == XmlPullParser.START_TAG) {
                    currentTag = parser.getName();
                    if ("item".equalsIgnoreCase(currentTag)) {
                        inItem = true;
                        title = link = description = "";
                    }
                } else if (event == XmlPullParser.TEXT && inItem && currentTag != null) {
                    String t = parser.getText();
                    if ("title".equalsIgnoreCase(currentTag)) title += t;
                    else if ("link".equalsIgnoreCase(currentTag)) link += t;
                    else if ("description".equalsIgnoreCase(currentTag)) description += t;
                } else if (event == XmlPullParser.END_TAG) {
                    String name = parser.getName();
                    if ("item".equalsIgnoreCase(name) && inItem) {
                        if (!title.trim().isEmpty() || !description.trim().isEmpty()) {
                            results.add(new SearchResult(
                                    clean(title),
                                    cleanUrl(link),
                                    truncate(clean(description), 420)
                            ));
                        }
                        inItem = false;
                    }
                    currentTag = null;
                }
                event = parser.next();
            }
            return results;
        } finally {
            conn.disconnect();
        }
    }

    private static List<SearchResult> searchDuckDuckGo(String query, int maxResults) throws Exception {
        String encoded = URLEncoder.encode(query, StandardCharsets.UTF_8.name());
        URL url = new URL("https://api.duckduckgo.com/?q=" + encoded +
                "&format=json&no_html=1&no_redirect=1&skip_disambig=1");
        HttpURLConnection conn = open(url, "application/json,*/*;q=0.5");

        try {
            String json = readLimited(conn.getInputStream(), 180_000);
            JSONObject root = new JSONObject(json);
            List<SearchResult> results = new ArrayList<>();

            String abstractText = root.optString("AbstractText", "").trim();
            String abstractUrl = root.optString("AbstractURL", "").trim();
            String heading = root.optString("Heading", "").trim();

            if (!abstractText.isEmpty()) {
                results.add(new SearchResult(
                        heading.isEmpty() ? "نتيجة مباشرة" : heading,
                        abstractUrl,
                        truncate(abstractText, 500)
                ));
            }

            JSONArray topics = root.optJSONArray("RelatedTopics");
            if (topics != null) collectTopics(topics, results, maxResults);
            return results;
        } finally {
            conn.disconnect();
        }
    }

    private static void collectTopics(JSONArray topics, List<SearchResult> out, int max) {
        for (int i = 0; i < topics.length() && out.size() < max; i++) {
            JSONObject item = topics.optJSONObject(i);
            if (item == null) continue;

            JSONArray nested = item.optJSONArray("Topics");
            if (nested != null) {
                collectTopics(nested, out, max);
                continue;
            }

            String text = item.optString("Text", "").trim();
            String url = item.optString("FirstURL", "").trim();
            if (!text.isEmpty()) {
                out.add(new SearchResult("نتيجة ويب", url, truncate(text, 420)));
            }
        }
    }

    static WebPayload payloadFromAccepted(List<SearchResult> results, boolean direct) {
        StringBuilder context = new StringBuilder();
        StringBuilder sources = new StringBuilder();
        int index = 1;

        for (SearchResult r : results) {
            if (r.title.isEmpty() && r.snippet.isEmpty()) continue;

            context.append("[").append(index).append("] ")
                    .append(r.title.isEmpty() ? "نتيجة ويب" : r.title).append("\n");
            if (!r.url.isEmpty()) context.append("URL: ").append(r.url).append("\n");
            if (!r.snippet.isEmpty()) context.append(r.snippet).append("\n");
            context.append("\n");

            sources.append("[").append(index).append("] ")
                    .append(r.title.isEmpty() ? "مصدر" : r.title);
            if (!r.url.isEmpty()) sources.append("\n").append(r.url);
            sources.append("\n");

            index++;
        }

        int count = index - 1;
        return new WebPayload(
                context.toString().trim(),
                sources.toString().trim(),
                count,
                direct
        );
    }

    private static String firstUrl(String text) {
        Matcher m = URL_PATTERN.matcher(text == null ? "" : text);
        if (!m.find()) return "";

        String u = m.group();
        while (u.endsWith(".") || u.endsWith(",") || u.endsWith("،")
                || u.endsWith(")") || u.endsWith("]")) {
            u = u.substring(0, u.length() - 1);
        }
        return u;
    }

    private static HttpURLConnection open(URL url, String accept) throws Exception {
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setConnectTimeout(CONNECT_TIMEOUT_MS);
        conn.setReadTimeout(READ_TIMEOUT_MS);
        conn.setInstanceFollowRedirects(true);
        conn.setRequestProperty(
                "User-Agent",
                "Mozilla/5.0 (Linux; Android 16) AppleWebKit/537.36 " +
                        "(KHTML, like Gecko) Chrome/140.0 Mobile Safari/537.36"
        );
        conn.setRequestProperty("Accept", accept);
        conn.setRequestProperty("Accept-Language", "ar,en;q=0.8");

        int code = conn.getResponseCode();
        if (code < 200 || code >= 400) {
            conn.disconnect();
            throw new IllegalStateException("HTTP " + code);
        }
        return conn;
    }

    private static String readLimited(InputStream input, int maxChars) throws Exception {
        StringBuilder out = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(input, StandardCharsets.UTF_8))) {
            char[] buffer = new char[4096];
            int n;
            while ((n = reader.read(buffer)) != -1) {
                int remaining = maxChars - out.length();
                if (remaining <= 0) break;
                out.append(buffer, 0, Math.min(n, remaining));
                if (out.length() >= maxChars) break;
            }
        }
        return out.toString();
    }

    private static String extractTitle(String html) {
        Matcher m = Pattern.compile(
                "(?is)<title[^>]*>(.*?)</title>"
        ).matcher(html == null ? "" : html);
        if (!m.find()) return "";
        return clean(m.group(1));
    }

    private static String htmlToReadableText(String html) {
        if (html == null) return "";

        String s = html;
        s = s.replaceAll("(?is)<script[^>]*>.*?</script>", " ");
        s = s.replaceAll("(?is)<style[^>]*>.*?</style>", " ");
        s = s.replaceAll("(?is)<noscript[^>]*>.*?</noscript>", " ");
        s = s.replaceAll("(?is)<svg[^>]*>.*?</svg>", " ");
        s = s.replaceAll("(?is)<!--.*?-->", " ");
        s = s.replaceAll("(?i)<br\\s*/?>", "\n");
        s = s.replaceAll("(?i)</p\\s*>", "\n");
        s = s.replaceAll("(?i)</div\\s*>", "\n");
        s = s.replaceAll("(?i)</li\\s*>", "\n");
        s = s.replaceAll("(?s)<[^>]+>", " ");

        s = decodeEntities(s);
        s = s.replaceAll("[ \\t\\x0B\\f\\r]+", " ");
        s = s.replaceAll("\n[ \\t]+", "\n");
        s = s.replaceAll("\n{3,}", "\n\n");
        return s.trim();
    }

    private static String clean(String value) {
        if (value == null) return "";
        return cleanWhitespace(decodeEntities(value.replaceAll("<[^>]+>", " ")));
    }

    private static String cleanWhitespace(String value) {
        if (value == null) return "";
        return value.replaceAll("\\s+", " ").trim();
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

    private static String cleanUrl(String value) {
        return value == null ? "" : value.trim();
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, max) + "…";
    }

}
