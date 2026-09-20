package com.musab.aragpt2;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public final class CandidateSet {
    public final String turnId;
    public final String question;
    public final AnswerCandidate local;
    public final AnswerCandidate web;
    public final AnswerCandidate hosted;

    public CandidateSet(String turnId, String question,
                        AnswerCandidate local,
                        AnswerCandidate web,
                        AnswerCandidate hosted) {
        this.turnId = clean(turnId);
        this.question = clean(question);
        this.local = requireKind(local, AnswerCandidate.Kind.LOCAL, "local");
        this.web = requireKind(web, AnswerCandidate.Kind.WEB, "web");
        this.hosted = requireKind(hosted, AnswerCandidate.Kind.HOSTED, "hosted");
    }

    public List<AnswerCandidate> all() {
        return Collections.unmodifiableList(Arrays.asList(local, web, hosted));
    }

    public AnswerCandidate byId(String candidateId) {
        String id = clean(candidateId);
        for (AnswerCandidate c : all()) {
            if (c.id.equals(id)) return c;
        }
        return null;
    }

    private static AnswerCandidate requireKind(
            AnswerCandidate candidate, AnswerCandidate.Kind kind, String id) {
        if (candidate == null || candidate.kind != kind) {
            return AnswerCandidate.unavailable(id, kind, "none", "غير متاح");
        }
        return candidate;
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
