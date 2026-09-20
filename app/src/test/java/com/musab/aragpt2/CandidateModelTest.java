package com.musab.aragpt2;

import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class CandidateModelTest {
    @Test
    public void unavailableCandidateCannotPretendToHaveAnswer() {
        AnswerCandidate c = AnswerCandidate.unavailable(
                "hosted",
                AnswerCandidate.Kind.HOSTED,
                "gemini",
                "غير متاح"
        );

        assertFalse(c.available);
        assertEquals("", c.answer);
        assertTrue(c.sources.isEmpty());
    }

    @Test
    public void candidateSetHasStableTurnIdAndThreeSlots() {
        AnswerCandidate local = AnswerCandidate.available(
                "local", AnswerCandidate.Kind.LOCAL, "qwen", "جواب محلي",
                Collections.emptyList(), "جاهز");
        AnswerCandidate web = AnswerCandidate.unavailable(
                "web", AnswerCandidate.Kind.WEB, "web-evidence", "لا توجد أدلة");
        AnswerCandidate hosted = AnswerCandidate.unavailable(
                "hosted", AnswerCandidate.Kind.HOSTED, "none", "غير مهيأ");

        CandidateSet set = new CandidateSet(
                "turn-123", "سؤال", local, web, hosted);

        assertEquals("turn-123", set.turnId);
        assertEquals("local", set.local.id);
        assertEquals("web", set.web.id);
        assertEquals("hosted", set.hosted.id);
        assertEquals(3, set.all().size());
    }
}
