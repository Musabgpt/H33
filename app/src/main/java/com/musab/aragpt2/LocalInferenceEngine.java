package com.musab.aragpt2;

public interface LocalInferenceEngine {
    @FunctionalInterface
    interface StreamListener {
        void onUpdate(String fullText);
    }

    String generate(
            String question,
            int maxNewTokens,
            String evidenceContext,
            boolean evidenceOnly,
            StreamListener listener
    ) throws Exception;

    void cancel();
}
