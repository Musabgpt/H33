package com.musab.aragpt2;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class FreshnessPolicyTest {
    @Test
    public void detectsCurrentArabicAndEnglishQuestions() {
        assertTrue(FreshnessPolicy.requiresFreshEvidence(
                "من هو الرئيس الحالي؟"));
        assertTrue(FreshnessPolicy.requiresFreshEvidence(
                "What is the latest price today?"));
    }

    @Test
    public void leavesStableKnowledgeAsGeneral() {
        assertFalse(FreshnessPolicy.requiresFreshEvidence(
                "ما هي عاصمة فرنسا؟"));
        assertFalse(FreshnessPolicy.requiresFreshEvidence(
                "Explain photosynthesis."));
    }
}
