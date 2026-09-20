package com.musab.aragpt2;

import org.json.JSONObject;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class GeminiInteractionsClientTest {
    @Test
    public void requestUsesInteractionsApiGoogleSearchTool() throws Exception {
        HostedProviderConfig config =
                new HostedProviderConfig("gemini", "gemini-3.8-flash", "secret");

        JSONObject request =
                GeminiInteractionsClient.buildRequest(config, "من هو رئيس فرنسا؟");

        assertEquals("gemini-3.8-flash", request.getString("model"));
        assertEquals("من هو رئيس فرنسا؟", request.getString("input"));
        assertEquals(
                "google_search",
                request.getJSONArray("tools").getJSONObject(0).getString("type")
        );
    }

    @Test
    public void parserExtractsAnswerAndUniqueUrlCitations() throws Exception {
        String json = "{"
                + "\"status\":\"completed\","
                + "\"steps\":[{"
                + "\"type\":\"model_output\","
                + "\"content\":[{"
                + "\"type\":\"text\","
                + "\"text\":\"الجواب grounded\","
                + "\"annotations\":["
                + "{\"type\":\"url_citation\",\"url\":\"https://example.com/a\",\"title\":\"Example\"},"
                + "{\"type\":\"url_citation\",\"url\":\"https://example.com/a\",\"title\":\"Example\"},"
                + "{\"type\":\"url_citation\",\"url\":\"https://other.org/b\",\"title\":\"Other\"}"
                + "]"
                + "}]}]}";

        GeminiInteractionsClient.ParsedResponse parsed =
                GeminiInteractionsClient.parseResponse(json);

        assertEquals("الجواب grounded", parsed.answer);
        assertEquals(2, parsed.sources.size());
        assertEquals("example.com", parsed.sources.get(0).host);
        assertEquals("other.org", parsed.sources.get(1).host);
        assertTrue(parsed.grounded);
    }

    @Test(expected = IllegalStateException.class)
    public void parserRejectsCompletedResponseWithoutText() {
        GeminiInteractionsClient.parseResponse(
                "{\"status\":\"completed\",\"steps\":[]}"
        );
    }
}
