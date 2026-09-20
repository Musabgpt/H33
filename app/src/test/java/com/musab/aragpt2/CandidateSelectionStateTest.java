package com.musab.aragpt2;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class CandidateSelectionStateTest {
    @Test
    public void duplicateSelectionIsRejected() {
        CandidateSelectionState state = new CandidateSelectionState("t1");

        assertTrue(state.select("web"));
        assertFalse(state.select("local"));
        assertEquals("web", state.selectedCandidateId());
    }

    @Test
    public void correctionSupersedesSelectionInState() {
        CandidateSelectionState state = new CandidateSelectionState("t1");

        assertTrue(state.select("web"));
        assertTrue(state.correct("الجواب الصحيح"));

        assertEquals("", state.selectedCandidateId());
        assertEquals("الجواب الصحيح", state.correction());
        assertTrue(state.isFinalized());
    }

    @Test
    public void blankSelectionOrCorrectionIsRejected() {
        CandidateSelectionState state = new CandidateSelectionState("t1");

        assertFalse(state.select("  "));
        assertFalse(state.correct(""));
        assertFalse(state.isFinalized());
    }
}
