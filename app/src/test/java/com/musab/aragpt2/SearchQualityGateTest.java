package com.musab.aragpt2;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class SearchQualityGateTest {
    @Test
    public void rejectsAdultDomainsEvenWhenSearchReturnedThem() {
        List<SearchResult> raw = Arrays.asList(
                new SearchResult("Today's selection - XNXX.COM",
                        "https://www.xnxx.com/todays-selection/", "adult content"),
                new SearchResult("Free Porn Videos",
                        "https://xnxx.dev/", "xxx sex videos")
        );

        SearchQualityGate.Result result =
                SearchQualityGate.filter("من هو رئيس سوريا الحالي؟", raw, 5);

        assertTrue(result.accepted.isEmpty());
        assertEquals(2, result.rejected.size());
    }

    @Test
    public void keepsRelevantAndDropsIrrelevantResults() {
        List<SearchResult> raw = Arrays.asList(
                new SearchResult("رئاسة الجمهورية العربية السورية",
                        "https://example.gov.sy/president", "رئيس الجمهورية السورية"),
                new SearchResult("Football scores",
                        "https://sports.example/score", "latest match results")
        );

        SearchQualityGate.Result result =
                SearchQualityGate.filter("من هو رئيس سوريا الحالي؟", raw, 5);

        assertEquals(1, result.accepted.size());
        assertEquals("example.gov.sy", result.accepted.get(0).host);
    }

    @Test
    public void removesDuplicateHostsAfterBestResult() {
        List<SearchResult> raw = Arrays.asList(
                new SearchResult("Syria president official",
                        "https://example.org/a", "Syria president"),
                new SearchResult("Another result",
                        "https://example.org/b", "Syria president current")
        );

        SearchQualityGate.Result result =
                SearchQualityGate.filter("Syria current president", raw, 5);

        assertEquals(1, result.accepted.size());
    }

    @Test
    public void acceptsBilingualEvidenceWithArabicEntityOverlap() {
        List<SearchResult> raw = Arrays.asList(
                new SearchResult("Syria president official",
                        "https://official.example/president",
                        "معلومات عن رئيس سوريا الحالي / current Syria president"),
                new SearchResult("Syria president official",
                        "https://english-only.example/president",
                        "current Syria president information")
        );

        SearchQualityGate.Result result =
                SearchQualityGate.filter("من هو رئيس سوريا الحالي؟", raw, 5);

        assertEquals(1, result.accepted.size());
        assertEquals("official.example", result.accepted.get(0).host);
    }
    @Test
    public void rejectsSingleTokenCoincidenceForMultiTermQuestion() {
        List<SearchResult> raw = Arrays.asList(
                new SearchResult(
                        "دليل السياحة في سوريا",
                        "https://travel.example/syria",
                        "أفضل الأماكن السياحية والتاريخية في سوريا"
                )
        );

        SearchQualityGate.Result result =
                SearchQualityGate.filter("من هو رئيس سوريا الحالي؟", raw, 5);

        assertTrue(result.accepted.isEmpty());
        assertEquals(1, result.rejected.size());
    }

    @Test
    public void rejectsEnglishInterrogativeNoiseWithoutTopicMatch() {
        List<SearchResult> raw = Arrays.asList(
                new SearchResult(
                        "What does many mean?",
                        "https://dictionary.example/many",
                        "How many are there? Definition, examples, and usage of many."
                )
        );

        SearchQualityGate.Result result =
                SearchQualityGate.filter(
                        "How many continents are there in the world and what are their names?",
                        raw,
                        5
                );

        assertTrue(result.accepted.isEmpty());
        assertEquals(1, result.rejected.size());
    }

    @Test
    public void rejectsArabicInterrogativeNoiseWithoutContinentMatch() {
        List<SearchResult> raw = Arrays.asList(
                new SearchResult(
                        "كم عدد الكلمات في العالم؟",
                        "https://language.example/count",
                        "شرح عن عدد الكلمات المستخدمة في العالم"
                )
        );

        SearchQualityGate.Result result =
                SearchQualityGate.filter("كم عدد قارات العالم وماهي أسماؤها؟", raw, 5);

        assertTrue(result.accepted.isEmpty());
        assertEquals(1, result.rejected.size());
    }

}
