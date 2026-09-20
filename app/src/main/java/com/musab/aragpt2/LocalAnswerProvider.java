package com.musab.aragpt2;

public interface LocalAnswerProvider {
    AnswerCandidate answer(
            String question,
            QwenEngine.StreamListener listener
    ) throws Exception;
}
