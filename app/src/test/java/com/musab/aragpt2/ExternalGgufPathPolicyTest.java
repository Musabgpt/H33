package com.musab.aragpt2;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/** Regression tests for resolving local shared-storage GGUF documents to native paths. */
public class ExternalGgufPathPolicyTest {
    @Test
    public void resolvesPrimaryStorageDocumentId() {
        assertEquals(
                "/storage/emulated/0/Download/DeepSeek-Coder-1.3B.Q4_K_M.gguf",
                ExternalGgufPathPolicy.primaryStoragePath(
                        "primary:Download/DeepSeek-Coder-1.3B.Q4_K_M.gguf",
                        "/storage/emulated/0"));
    }

    @Test
    public void rejectsNonPrimaryDocumentIds() {
        assertNull(ExternalGgufPathPolicy.primaryStoragePath(
                "home:DeepSeek-Coder-1.3B.Q4_K_M.gguf",
                "/storage/emulated/0"));
    }

    @Test
    public void rejectsTraversalOutsideStorageRoot() {
        assertNull(ExternalGgufPathPolicy.primaryStoragePath(
                "primary:../Android/data/com.example/file.gguf",
                "/storage/emulated/0"));
    }
}
