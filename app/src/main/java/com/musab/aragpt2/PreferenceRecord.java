package com.musab.aragpt2;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

public final class PreferenceRecord {
    public static final int SCHEMA_VERSION = 1;

    public static final class CandidateSnapshot {
        public final String id;
        public final String kind;
        public final String provider;
        public final String answer;
        public final boolean available;
        public final List<SearchResult> sources;
        public final String status;

        CandidateSnapshot(String id, String kind, String provider, String answer,
                          boolean available, List<SearchResult> sources, String status) {
            this.id = clean(id);
            this.kind = clean(kind);
            this.provider = clean(provider);
            this.answer = clean(answer);
            this.available = available && !this.answer.isEmpty();
            this.sources = Collections.unmodifiableList(
                    new ArrayList<>(sources == null ? Collections.emptyList() : sources));
            this.status = clean(status);
        }

        static CandidateSnapshot fromCandidate(AnswerCandidate candidate) {
            return new CandidateSnapshot(
                    candidate.id,
                    candidate.kind.name(),
                    candidate.provider,
                    candidate.answer,
                    candidate.available,
                    candidate.sources,
                    candidate.status
            );
        }

        JSONObject toJsonObject() throws Exception {
            JSONObject obj = new JSONObject();
            obj.put("id", id);
            obj.put("kind", kind);
            obj.put("provider", provider);
            obj.put("answer", answer);
            obj.put("available", available);
            obj.put("status", status);

            JSONArray sourceArray = new JSONArray();
            for (SearchResult source : sources) {
                JSONObject s = new JSONObject();
                s.put("title", source.title);
                s.put("url", source.url);
                s.put("host", source.host);
                s.put("snippet", source.snippet);
                s.put("relevance", source.relevance);
                sourceArray.put(s);
            }
            obj.put("sources", sourceArray);
            return obj;
        }

        static CandidateSnapshot fromJsonObject(JSONObject obj) {
            ArrayList<SearchResult> sources = new ArrayList<>();
            JSONArray sourceArray = obj.optJSONArray("sources");
            if (sourceArray != null) {
                for (int i = 0; i < sourceArray.length(); i++) {
                    JSONObject s = sourceArray.optJSONObject(i);
                    if (s == null) continue;
                    sources.add(new SearchResult(
                            s.optString("title", ""),
                            s.optString("url", ""),
                            s.optString("snippet", ""),
                            s.optString("host", ""),
                            s.optDouble("relevance", 0.0)
                    ));
                }
            }

            return new CandidateSnapshot(
                    obj.optString("id", ""),
                    obj.optString("kind", ""),
                    obj.optString("provider", ""),
                    obj.optString("answer", ""),
                    obj.optBoolean("available", false),
                    sources,
                    obj.optString("status", "")
            );
        }
    }

    public final int schemaVersion;
    public final String id;
    public final long timestampMs;
    public final String turnId;
    public final String question;
    public final List<CandidateSnapshot> candidates;
    public final String selectionType;
    public final String candidateId;
    public final String correction;

    private PreferenceRecord(
            int schemaVersion,
            String id,
            long timestampMs,
            String turnId,
            String question,
            List<CandidateSnapshot> candidates,
            String selectionType,
            String candidateId,
            String correction) {
        this.schemaVersion = schemaVersion;
        this.id = clean(id);
        this.timestampMs = timestampMs;
        this.turnId = clean(turnId);
        this.question = clean(question);
        this.candidates = Collections.unmodifiableList(new ArrayList<>(candidates));
        this.selectionType = clean(selectionType);
        this.candidateId = clean(candidateId);
        this.correction = clean(correction);
    }

    public static PreferenceRecord selection(
            CandidateSet set, String candidateId, long timestampMs) {
        if (set == null) throw new IllegalArgumentException("candidate set required");
        AnswerCandidate selected = set.byId(candidateId);
        if (selected == null || !selected.available) {
            throw new IllegalArgumentException("selected candidate unavailable");
        }
        return create(set, "candidate", selected.id, "", timestampMs);
    }

    public static PreferenceRecord correction(
            CandidateSet set, String correction, long timestampMs) {
        if (set == null) throw new IllegalArgumentException("candidate set required");
        String answer = clean(correction);
        if (answer.isEmpty()) throw new IllegalArgumentException("correction required");
        return create(set, "user_correction", "", answer, timestampMs);
    }

    private static PreferenceRecord create(
            CandidateSet set, String selectionType, String candidateId,
            String correction, long timestampMs) {
        ArrayList<CandidateSnapshot> snapshots = new ArrayList<>();
        for (AnswerCandidate candidate : set.all()) {
            snapshots.add(CandidateSnapshot.fromCandidate(candidate));
        }
        return new PreferenceRecord(
                SCHEMA_VERSION,
                UUID.randomUUID().toString(),
                timestampMs,
                set.turnId,
                set.question,
                snapshots,
                selectionType,
                candidateId,
                correction
        );
    }

    public String effectiveAnswer() {
        if ("user_correction".equals(selectionType)) return correction;
        if (!"candidate".equals(selectionType)) return "";

        for (CandidateSnapshot candidate : candidates) {
            if (candidate.id.equals(candidateId) && candidate.available) {
                return candidate.answer;
            }
        }
        return "";
    }

    public CandidateSnapshot selectedCandidate() {
        if (!"candidate".equals(selectionType)) return null;
        for (CandidateSnapshot candidate : candidates) {
            if (candidate.id.equals(candidateId)) return candidate;
        }
        return null;
    }

    public String toJson() throws Exception {
        JSONObject obj = new JSONObject();
        obj.put("schema_version", schemaVersion);
        obj.put("id", id);
        obj.put("timestamp_ms", timestampMs);
        obj.put("turn_id", turnId);
        obj.put("question", question);

        JSONArray candidateArray = new JSONArray();
        for (CandidateSnapshot candidate : candidates) {
            candidateArray.put(candidate.toJsonObject());
        }
        obj.put("candidates", candidateArray);

        JSONObject selection = new JSONObject();
        selection.put("type", selectionType);
        if (candidateId.isEmpty()) {
            selection.put("candidate_id", JSONObject.NULL);
        } else {
            selection.put("candidate_id", candidateId);
        }
        obj.put("selection", selection);

        if (correction.isEmpty()) {
            obj.put("correction", JSONObject.NULL);
        } else {
            obj.put("correction", correction);
        }
        return obj.toString();
    }

    public static PreferenceRecord fromJson(String json) throws Exception {
        JSONObject obj = new JSONObject(json);
        int schema = obj.optInt("schema_version", 0);
        if (schema != SCHEMA_VERSION) {
            throw new IllegalArgumentException("unsupported schema version");
        }

        JSONArray candidateArray = obj.optJSONArray("candidates");
        ArrayList<CandidateSnapshot> candidates = new ArrayList<>();
        if (candidateArray != null) {
            for (int i = 0; i < candidateArray.length(); i++) {
                JSONObject c = candidateArray.optJSONObject(i);
                if (c != null) candidates.add(CandidateSnapshot.fromJsonObject(c));
            }
        }

        JSONObject selection = obj.optJSONObject("selection");
        String selectionType = selection == null
                ? "" : selection.optString("type", "");
        String candidateId = "";
        if (selection != null && !selection.isNull("candidate_id")) {
            candidateId = selection.optString("candidate_id", "");
        }

        String correction = obj.isNull("correction")
                ? "" : obj.optString("correction", "");

        PreferenceRecord record = new PreferenceRecord(
                schema,
                obj.optString("id", ""),
                obj.optLong("timestamp_ms", 0L),
                obj.optString("turn_id", ""),
                obj.optString("question", ""),
                candidates,
                selectionType,
                candidateId,
                correction
        );

        if (record.turnId.isEmpty() || record.question.isEmpty() ||
                record.effectiveAnswer().isEmpty()) {
            throw new IllegalArgumentException("invalid preference record");
        }
        return record;
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
