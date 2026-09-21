package com.musab.aragpt2;

/**
 * Compatibility name kept for the existing candidate pipeline.
 * It no longer contains or references ONNX Runtime; it delegates to the native GGUF engine.
 */
public final class OrtGenAiLocalInferenceEngine implements LocalInferenceEngine {
    private final CodeModelEngine engine;

    public OrtGenAiLocalInferenceEngine(CodeModelEngine engine) {
        if (engine == null) throw new IllegalArgumentException("engine required");
        this.engine = engine;
    }

    @Override
    public String generate(String question, int maxNewTokens, String evidenceContext,
                           boolean evidenceOnly, StreamListener listener) throws Exception {
        return engine.generate(question, maxNewTokens, evidenceContext, evidenceOnly, listener);
    }

    @Override
    public void cancel() {
        engine.cancelGeneration();
    }
}
