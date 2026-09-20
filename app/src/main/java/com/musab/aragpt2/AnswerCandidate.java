package com.musab.aragpt2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class AnswerCandidate {
    public enum Kind { LOCAL, WEB, HOSTED }

    public final String id;
    public final Kind kind;
    public final String provider;
    public final String answer;
    public final boolean available;
    public final List<SearchResult> sources;
    public final String status;

    private AnswerCandidate(String id, Kind kind, String provider, String answer,
                            boolean available, List<SearchResult> sources, String status) {
        this.id = clean(id);
        this.kind = kind;
        this.provider = clean(provider);
        this.answer = available ? clean(answer) : "";
        this.available = available && !this.answer.isEmpty();
        this.sources = Collections.unmodifiableList(
                new ArrayList<>(sources == null ? Collections.emptyList() : sources)
        );
        this.status = clean(status);
    }

    public static AnswerCandidate available(
            String id, Kind kind, String provider, String answer,
            List<SearchResult> sources, String status) {
        return new AnswerCandidate(id, kind, provider, answer, true, sources, status);
    }

    public static AnswerCandidate unavailable(
            String id, Kind kind, String provider, String status) {
        return new AnswerCandidate(
                id, kind, provider, "", false, Collections.emptyList(), status);
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
