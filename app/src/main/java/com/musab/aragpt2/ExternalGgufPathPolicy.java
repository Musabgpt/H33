package com.musab.aragpt2;

import java.io.File;
import java.io.IOException;

/** Pure path policy for local files exposed through Android's primary external-storage provider. */
final class ExternalGgufPathPolicy {
    private ExternalGgufPathPolicy() {}

    static String primaryStoragePath(String documentId, String storageRoot) {
        if (documentId == null || storageRoot == null || !documentId.startsWith("primary:")) {
            return null;
        }

        String relative = documentId.substring("primary:".length());
        if (relative.isEmpty()) return null;

        try {
            File root = new File(storageRoot).getCanonicalFile();
            File candidate = new File(root, relative).getCanonicalFile();
            String rootPath = root.getPath();
            String candidatePath = candidate.getPath();
            if (!candidatePath.equals(rootPath)
                    && !candidatePath.startsWith(rootPath + File.separator)) {
                return null;
            }
            return candidatePath;
        } catch (IOException ignored) {
            return null;
        }
    }
}
