package com.musab.aragpt2;

public interface LocalAnswerProvider {
    AnswerCandidate answer(
            String question,
            LocalInferenceEngine.StreamListener listener
    ) throws Exception;
}
