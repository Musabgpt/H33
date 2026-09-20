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
        assertTrue(GoogleAiOverviewProvider.isAllowedGoogleHost("consent.google.com"));
        assertTrue(!GoogleAiOverviewProvider.isAllowedGoogleHost("accounts.google.com"));
        assertTrue(!GoogleAiOverviewProvider.isAllowedGoogleHost("support.google.com"));
        assertTrue(!GoogleAiOverviewProvider.isAllowedGoogleHost("evilgoogle.com"));
        assertTrue(!GoogleAiOverviewProvider.isAllowedGoogleHost("google.com.evil.example"));
    }

    @Test
    public void chromeLikeUserAgentRemovesWebViewMarkers() {
        String input =
                "Mozilla/5.0 (Linux; Android 16; Pixel Build/ABC; wv) "
                + "AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 "
                + "Chrome/140.0.0.0 Mobile Safari/537.36";

        String ua = GoogleAiOverviewProvider.chromeLikeUserAgent(input);

        assertTrue(!ua.contains("; wv"));
        assertTrue(!ua.contains("Version/4.0"));
        assertTrue(ua.contains("Chrome/140.0.0.0"));
        assertTrue(ua.contains("Mobile Safari/537.36"));
    }

    @Test
    public void distinguishesGoogleConsentFromChallengePages() {
        assertTrue(GoogleAiOverviewProvider.isConsentPage(
                "https://consent.google.com/m?continue=https://www.google.com/search"));
        assertTrue(!GoogleAiOverviewProvider.isChallengePage(
                "https://consent.google.com/m"));

        assertTrue(GoogleAiOverviewProvider.isChallengePage(
                "https://www.google.com/sorry/index?continue=x"));
        assertTrue(GoogleAiOverviewProvider.isChallengePage(
                "https://www.google.com/recaptcha/api2/anchor"));
        assertTrue(!GoogleAiOverviewProvider.isConsentPage(
                "https://www.google.com/sorry/index"));
    }

}
