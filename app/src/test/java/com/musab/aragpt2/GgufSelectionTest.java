package com.musab.aragpt2;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Regression tests for the external GGUF picker contract. */
public class GgufSelectionTest {
    @Test
    public void acceptsGgufFileNames() {
        assertTrue(ModelImportActivity.isGgufName("DeepSeek-Coder-1.3B.Q4_K_M.gguf"));
        assertTrue(ModelImportActivity.isGgufName("model.GGUF"));
    }

    @Test
    public void rejectsNonGgufFiles() {
        assertFalse(ModelImportActivity.isGgufName("model.bin"));
        assertFalse(ModelImportActivity.isGgufName("model.ggml"));
    }
}
