package com.musab.aragpt2;

public final class CandidateSelectionState {
    private final String turnId;
    private String selectedCandidateId = "";
    private String correction = "";

    public CandidateSelectionState(String turnId) {
        this.turnId = clean(turnId);
        if (this.turnId.isEmpty()) {
            throw new IllegalArgumentException("turnId required");
        }
    }

    public synchronized boolean select(String candidateId) {
        String id = clean(candidateId);
        if (id.isEmpty() || isFinalized()) return false;
        selectedCandidateId = id;
        correction = "";
        return true;
    }

    public synchronized boolean correct(String answer) {
        String value = clean(answer);
        if (value.isEmpty()) return false;
        selectedCandidateId = "";
        correction = value;
        return true;
    }

    public synchronized boolean isFinalized() {
        return !selectedCandidateId.isEmpty() || !correction.isEmpty();
    }

    public synchronized String selectedCandidateId() {
        return selectedCandidateId;
    }

    public synchronized String correction() {
        return correction;
    }

    public String turnId() {
        return turnId;
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
