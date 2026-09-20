package com.musab.aragpt2;

public interface WebEvidenceAnswerProvider {
    AnswerCandidate answer(String question) throws Exception;
}
