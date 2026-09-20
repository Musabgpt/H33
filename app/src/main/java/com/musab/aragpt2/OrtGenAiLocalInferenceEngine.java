package com.musab.aragpt2;

public final class OrtGenAiLocalInferenceEngine implements LocalInferenceEngine {
    private final QwenEngine engine;

    public OrtGenAiLocalInferenceEngine(QwenEngine engine) {
        if (engine == null) throw new IllegalArgumentException("engine required");
        this.engine = engine;
    }

    @Override
    public String generate(
            String question,
            int maxNewTokens,
            String evidenceContext,
            boolean evidenceOnly,
            StreamListener listener) throws Exception {
        QwenEngine.StreamListener adapter =
                listener == null ? null : listener::onUpdate;
        return engine.generateCandidate(
                question,
                maxNewTokens,
                evidenceContext,
                evidenceOnly,
                adapter
        );
    }

    @Override
    public void cancel() {
        engine.cancelGeneration();
    }
}
