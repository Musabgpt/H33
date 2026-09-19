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

public final class WebSearchClient {
    private static final int CONNECT_TIMEOUT_MS = 7000;
    private static final int READ_TIMEOUT_MS = 9000;

    private WebSearchClient() {}

    public static boolean shouldAutoSearch(String question) {
        if (question == null) return false;
        String q = question.toLowerCase(Locale.ROOT);
        String[] cues = {
                "ابحث", "دورلي", "دور لي", "فتش", "بحث عن",
                "اليوم", "الآن", "حاليا", "حالياً", "أحدث", "احدث",
                "آخر خبر", "آخر الأخبار", "اخر خبر", "اخر الاخبار",
                "سعر اليوم", "السعر الآن", "ما الجديد", "مين حاليا", "من هو حاليا",
                "latest", "current", "today", "news", "search the web", "look up"
        };
        for (String cue : cues) if (q.contains(cue)) return true;
        return false;
    }

    public static String search(String query, int maxResults) throws Exception {
        String q = query == null ? "" : query.trim();
        if (q.isEmpty()) return "";

        try {
            String rss = searchBingRss(q, maxResults);
            if (!rss.isEmpty()) return rss;
        } catch (Exception ignored) {
        }

        return searchDuckDuckGo(q, maxResults);
    }

    private static String searchBingRss(String query, int maxResults) throws Exception {
        String encoded = URLEncoder.encode(query, StandardCharsets.UTF_8.name());
        URL url = new URL("https://www.bing.com/search?q=" + encoded + "&format=rss&setlang=ar");
        HttpURLConnection conn = open(url);
        try (InputStream in = new BufferedInputStream(conn.getInputStream())) {
            XmlPullParser parser = Xml.newPullParser();
            parser.setInput(in, "UTF-8");

            List<Result> results = new ArrayList<>();
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
                            results.add(new Result(clean(title), clean(link), clean(description)));
                        }
                        inItem = false;
                    }
                    currentTag = null;
                }
                event = parser.next();
            }
            return format(results);
        } finally {
            conn.disconnect();
        }
    }

    private static String searchDuckDuckGo(String query, int maxResults) throws Exception {
        String encoded = URLEncoder.encode(query, StandardCharsets.UTF_8.name());
        URL url = new URL("https://api.duckduckgo.com/?q=" + encoded +
                "&format=json&no_html=1&no_redirect=1&skip_disambig=1");
        HttpURLConnection conn = open(url);
        try {
            StringBuilder json = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                    conn.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) json.append(line);
            }

            JSONObject root = new JSONObject(json.toString());
            List<Result> results = new ArrayList<>();

            String abstractText = root.optString("AbstractText", "").trim();
            String abstractUrl = root.optString("AbstractURL", "").trim();
            String heading = root.optString("Heading", "").trim();
            if (!abstractText.isEmpty()) {
                results.add(new Result(
                        heading.isEmpty() ? "نتيجة مباشرة" : heading,
                        abstractUrl,
                        abstractText
                ));
            }

            JSONArray topics = root.optJSONArray("RelatedTopics");
            if (topics != null) collectTopics(topics, results, maxResults);
            return format(results);
        } finally {
            conn.disconnect();
        }
    }

    private static void collectTopics(JSONArray topics, List<Result> out, int max) {
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
            if (!text.isEmpty()) out.add(new Result("نتيجة ويب", url, text));
        }
    }

    private static HttpURLConnection open(URL url) throws Exception {
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setConnectTimeout(CONNECT_TIMEOUT_MS);
        conn.setReadTimeout(READ_TIMEOUT_MS);
        conn.setInstanceFollowRedirects(true);
        conn.setRequestProperty("User-Agent", "Mozilla/5.0 (Android) H33/1.0");
        conn.setRequestProperty("Accept-Language", "ar,en;q=0.8");
        int code = conn.getResponseCode();
        if (code < 200 || code >= 400) {
            conn.disconnect();
            throw new IllegalStateException("HTTP " + code);
        }
        return conn;
    }

    private static String format(List<Result> results) {
        if (results.isEmpty()) return "";
        StringBuilder out = new StringBuilder();
        int i = 1;
        for (Result r : results) {
            out.append("[").append(i++).append("] ").append(r.title).append("\n");
            if (!r.url.isEmpty()) out.append("URL: ").append(r.url).append("\n");
            if (!r.snippet.isEmpty()) out.append(truncate(r.snippet, 500)).append("\n");
            out.append("\n");
        }
        return out.toString().trim();
    }

    private static String clean(String value) {
        if (value == null) return "";
        return value.replaceAll("<[^>]+>", " ")
                .replace("&amp;", "&")
                .replace("&quot;", "\"")
                .replace("&#39;", "'")
                .replaceAll("\\s+", " ")
                .trim();
    }

    private static String truncate(String s, int max) {
        if (s == null || s.length() <= max) return s == null ? "" : s;
        return s.substring(0, max) + "…";
    }

    private static final class Result {
        final String title;
        final String url;
        final String snippet;

        Result(String title, String url, String snippet) {
            this.title = title == null ? "" : title;
            this.url = url == null ? "" : url;
            this.snippet = snippet == null ? "" : snippet;
        }
    }
}
