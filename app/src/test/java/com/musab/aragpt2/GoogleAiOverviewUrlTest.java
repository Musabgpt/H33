package com.musab.aragpt2;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class GoogleAiOverviewUrlTest {
    @Test
    public void encodesArabicQueryAndLocale() {
        String url = GoogleAiOverviewProvider.buildSearchUrl(
                "من هو رئيس وزراء بريطانيا في الحرب العالمية الثانية",
                "ar",
                "GB"
        );

        assertTrue(url.startsWith("https://www.google.com/search?"));
        assertTrue(url.contains("hl=ar"));
        assertTrue(url.contains("gl=GB"));
        assertTrue(url.contains("q="));
        assertTrue(!url.contains(" "));
    }

    @Test
    public void defaultsLocaleWhenBlank() {
        String url = GoogleAiOverviewProvider.buildSearchUrl("hello", "", "");
        assertTrue(url.contains("hl=ar"));
        assertTrue(url.contains("gl=GB"));
    }
    @Test
    public void googleHostCheckRejectsLookalikeDomains() {
        assertTrue(GoogleAiOverviewProvider.isAllowedGoogleHost("google.com"));
        assertTrue(GoogleAiOverviewProvider.isAllowedGoogleHost("www.google.com"));
        assertTrue(GoogleAiOverviewProvider.isAllowedGoogleHost("google.co.uk"));
        assertTrue(!GoogleAiOverviewProvider.isAllowedGoogleHost("evilgoogle.com"));
        assertTrue(!GoogleAiOverviewProvider.isAllowedGoogleHost("google.com.evil.example"));
    }

}
