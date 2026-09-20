package com.musab.aragpt2;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class ConversationHistoryTest {
    @Test
    public void replacementTargetsTurnIdNotQuestionText() {
        ConversationHistory h = new ConversationHistory(16);
        h.appendTurn("t1", "نفس السؤال", "جواب 1");
        h.appendTurn("t2", "نفس السؤال", "جواب 2");

        assertTrue(h.replaceAnswer("t1", "جواب مختار"));

        List<ConversationHistory.Turn> turns = h.snapshot();
        assertEquals("جواب مختار", turns.get(1).content);
        assertEquals("جواب 2", turns.get(3).content);
    }

    @Test
    public void trimmingPreservesCompleteUserAssistantPairs() {
        ConversationHistory h = new ConversationHistory(4);
        h.appendTurn("t1", "س1", "ج1");
        h.appendTurn("t2", "س2", "ج2");
        h.appendTurn("t3", "س3", "ج3");

        List<ConversationHistory.Turn> turns = h.snapshot();
        assertEquals(4, turns.size());
        assertEquals("t2", turns.get(0).turnId);
        assertEquals("user", turns.get(0).role);
        assertEquals("t3", turns.get(2).turnId);
        assertEquals("assistant", turns.get(3).role);
    }

    @Test
    public void latestMatchingReplacementSupportsLegacyTurnsWithoutIds() {
        ConversationHistory h = new ConversationHistory(8);
        h.appendTurn("", "سؤال قديم", "جواب قديم");
        assertTrue(h.replaceLatestAnswer("سؤال قديم", "جواب قديم", "جواب مصحح"));
        assertEquals("جواب مصحح", h.snapshot().get(1).content);
    }
}
