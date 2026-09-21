package com.musab.aragpt2;

public final class NativeLocalInferenceEngine implements LocalInferenceEngine {
    private final CodeModelEngine engine;
    public NativeLocalInferenceEngine(CodeModelEngine engine) {
        if (engine == null) throw new IllegalArgumentException("engine required");
        this.engine = engine;
    }
    @Override public String generate(String question, int maxNewTokens, String evidenceContext,
                                     boolean evidenceOnly, StreamListener listener) throws Exception {
        return engine.generateCandidate(question, maxNewTokens, listener);
    }
    @Override public void cancel() { engine.cancelGeneration(); }
}
