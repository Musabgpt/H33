package com.musab.aragpt2;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TextFileToolTest {
    @Test
    public void preservesJsonlExtensionForTrainingExports() {
        assertEquals(
                "H33_DPO_123.jsonl",
                TextFileTool.normalizeFileName("H33_DPO_123.jsonl")
        );
    }

    @Test
    public void extensionlessNamesStillBecomeTxt() {
        assertEquals(
                "answer.txt",
                TextFileTool.normalizeFileName("answer")
        );
    }
}
