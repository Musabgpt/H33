package com.musab.aragpt2;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class GoogleAiOverviewParserTest {
    @Test
    public void parsesExactOverviewTextAndDeduplicatesSources() {
        String jsPayload =
                "{\"found\":true,"
                + "\"text\":\"رئيس وزراء بريطانيا خلال معظم فترة الحرب العالمية الثانية هو ونستون تشرشل.\","
                + "\"sources\":["
                + "{\"title\":\"Imperial War Museums\",\"url\":\"https://www.iwm.org.uk/a\"},"
                + "{\"title\":\"IWM duplicate\",\"url\":\"https://www.iwm.org.uk/a\"},"
                + "{\"title\":\"Britannica\",\"url\":\"https://www.britannica.com/b\"}"
                + "]}";

        GoogleAiOverviewParser.Result parsed =
                GoogleAiOverviewParser.parseJsonPayload(jsPayload);

        assertTrue(parsed.found);
        assertEquals(
                "رئيس وزراء بريطانيا خلال معظم فترة الحرب العالمية الثانية هو ونستون تشرشل.",
                parsed.text
        );
        assertEquals(2, parsed.sources.size());
        assertEquals("iwm.org.uk", parsed.sources.get(0).host);
        assertEquals("britannica.com", parsed.sources.get(1).host);
    }

    @Test
    public void parsesEvaluateJavascriptQuotedResult() {
        String callback =
                "\"{\\\"found\\\":true,\\\"text\\\":\\\"مرحبا بك!\\\","
                + "\\\"sources\\\":[]}\"";

        GoogleAiOverviewParser.Result parsed =
                GoogleAiOverviewParser.parseEvaluateJavascriptResult(callback);

        assertTrue(parsed.found);
        assertEquals("مرحبا بك!", parsed.text);
    }

    @Test
    public void missingOverviewIsHonestUnavailableState() {
        GoogleAiOverviewParser.Result parsed =
                GoogleAiOverviewParser.parseJsonPayload(
                        "{\"found\":false,\"text\":\"\",\"sources\":[]}"
                );

        assertFalse(parsed.found);
        assertEquals("", parsed.text);
        assertTrue(parsed.sources.isEmpty());
    }

    @Test
    public void domScriptSearchesArabicAndEnglishOverviewMarkers() {
        String script = GoogleAiOverviewDomScript.buildExtractionScript();

        assertTrue(script.contains("نبذة باستخدام الذكاء الاصطناعي"));
        assertTrue(script.contains("AI Overview"));
        assertTrue(script.contains("JSON.stringify"));
    }
    @Test
    public void unwrapsGoogleRedirectAndSkipsGoogleInternalLinks() {
        String json = "{"
                + "\"found\":true,"
                + "\"text\":\"جواب\","
                + "\"sources\":["
                + "{\"title\":\"BBC\",\"url\":\"https://www.google.com/url?q=https%3A%2F%2Fwww.bbc.com%2Fnews%2Farticle&sa=U\"},"
                + "{\"title\":\"Google internal\",\"url\":\"https://www.google.com/search?q=more\"}"
                + "]}";

        GoogleAiOverviewParser.Result parsed =
                GoogleAiOverviewParser.parseJsonPayload(json);

        assertEquals(1, parsed.sources.size());
        assertEquals("bbc.com", parsed.sources.get(0).host);
        assertEquals(
                "https://www.bbc.com/news/article",
                parsed.sources.get(0).url
        );
    }

}
