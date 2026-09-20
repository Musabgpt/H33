package com.musab.aragpt2;

public interface HostedAnswerProvider {
    AnswerCandidate answer(String question) throws Exception;
}
