package com.musab.aragpt2;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class DuckDuckGoHtmlParserTest {
    @Test
    public void parsesTitlesSnippetsAndOriginalUrls() {
        String html =
                "<html><body>"
                + "<a rel=\"nofollow\" class=\"result__a\" "
                + "href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Fsana.sy%2Fpresidency%2F&amp;rut=x\">"
                + "أخبار رئاسة الجمهورية العربية السورية</a>"
                + "<a class=\"result__snippet\">الرئيس السوري الحالي وأخبار الرئاسة السورية</a>"
                + "<a rel=\"nofollow\" class=\"result__a\" "
                + "href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Far.wikipedia.org%2Fwiki%2F%25D8%25B1%25D8%25A6%25D9%258A%25D8%25B3_%25D8%25B3%25D9%2588%25D8%25B1%25D9%258A%25D8%25A7&amp;rut=y\">"
                + "رئيس سوريا - ويكيبيديا</a>"
                + "<div class=\"result__snippet\">معلومات عن منصب رئيس سوريا الحالي.</div>"
                + "</body></html>";

        List<SearchResult> results = DuckDuckGoHtmlParser.parse(html, 5);

        assertEquals(2, results.size());
        assertEquals("sana.sy", results.get(0).host);
        assertEquals("https://sana.sy/presidency/", results.get(0).url);
        assertTrue(results.get(0).snippet.contains("الرئيس السوري"));
        assertEquals("ar.wikipedia.org", results.get(1).host);
        assertTrue(results.get(1).title.contains("رئيس سوريا"));
    }

    @Test
    public void ignoresDuckDuckGoInternalOrMalformedLinks() {
        String html =
                "<a class=\"result__a\" href=\"/settings\">Settings</a>"
                + "<a class=\"result__a\" href=\"javascript:alert(1)\">Bad</a>";

        assertTrue(DuckDuckGoHtmlParser.parse(html, 5).isEmpty());
    }

    @Test
    public void decodesHtmlEntitiesAndStripsTags() {
        String html =
                "<a class=\"result__a\" "
                + "href=\"https://example.org/page\"><b>Syria</b> &amp; President</a>"
                + "<div class=\"result__snippet\">Current &quot;president&quot; info</div>";

        List<SearchResult> results = DuckDuckGoHtmlParser.parse(html, 5);

        assertEquals(1, results.size());
        assertEquals("Syria & President", results.get(0).title);
        assertEquals("Current \"president\" info", results.get(0).snippet);
    }
}
