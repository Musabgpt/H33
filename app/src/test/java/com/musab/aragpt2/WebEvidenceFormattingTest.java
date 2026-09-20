package com.musab.aragpt2;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class WebEvidenceFormattingTest {
    @Test
    public void payloadUsesOnlyAcceptedResults() {
        List<SearchResult> accepted = Arrays.asList(
                new SearchResult("Official result",
                        "https://official.example/a", "relevant text")
        );

        WebSearchClient.WebPayload p =
                WebSearchClient.payloadFromAccepted(accepted, false);

        assertEquals(1, p.sourceCount);
        assertTrue(p.context.contains("relevant text"));
        assertTrue(p.sources.contains("official.example"));
        assertFalse(p.sources.contains("xnxx"));
    }
}
