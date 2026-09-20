package com.musab.aragpt2;

import org.json.JSONObject;

import java.util.Collections;
import java.util.List;

public final class PreferenceDatasetExporter {
    public String toSftJsonl(List<PreferenceRecord> records) {
        StringBuilder out = new StringBuilder();
        for (PreferenceRecord record : safe(records)) {
            if (record == null) continue;
            String answer = clean(record.effectiveAnswer());
            String prompt = clean(record.question);
            if (prompt.isEmpty() || answer.isEmpty()) continue;

            JSONObject row = new JSONObject();
            try {
                row.put("id", record.turnId);
                row.put("prompt", prompt);
                row.put("response", answer);
                row.put("memory_semantics", record.memorySemantics);
                row.put("conflict_status", record.conflictStatus);
                row.put("freshness_class", record.freshnessClass);
                row.put("selected_provenance", record.selectedProvenance);
                row.put("training_eligible", record.trainingEligible);
            } catch (Exception ex) {
                throw new IllegalStateException("تعذر تكوين سجل SFT", ex);
            }
            appendLine(out, row.toString());
        }
        return out.toString();
    }

    public String toPreferenceJsonl(List<PreferenceRecord> records) {
        StringBuilder out = new StringBuilder();
        for (PreferenceRecord record : safe(records)) {
            if (record == null) continue;

            String prompt = clean(record.question);
            String chosen = clean(record.effectiveAnswer());
            if (prompt.isEmpty() || chosen.isEmpty()) continue;

            for (PreferenceRecord.CandidateSnapshot candidate : record.candidates) {
                if (candidate == null || !candidate.available) continue;
                if ("candidate".equals(record.selectionType)
                        && candidate.id.equals(record.candidateId)) {
                    continue;
                }

                String rejected = clean(candidate.answer);
                if (rejected.isEmpty() || rejected.equals(chosen)) continue;

                JSONObject row = new JSONObject();
                try {
                    row.put("id", record.turnId);
                    row.put("prompt", prompt);
                    row.put("chosen", chosen);
                    row.put("rejected", rejected);
                    row.put("chosen_source",
                            "user_correction".equals(record.selectionType)
                                    ? "user_correction"
                                    : record.candidateId);
                    row.put("rejected_source", candidate.id);
                    row.put("memory_semantics", record.memorySemantics);
                    row.put("conflict_status", record.conflictStatus);
                    row.put("freshness_class", record.freshnessClass);
                    row.put("training_eligible", record.trainingEligible);
                } catch (Exception ex) {
                    throw new IllegalStateException(
                            "تعذر تكوين سجل preference", ex);
                }
                appendLine(out, row.toString());
            }
        }
        return out.toString();
    }

    private static List<PreferenceRecord> safe(List<PreferenceRecord> records) {
        return records == null ? Collections.emptyList() : records;
    }

    private static void appendLine(StringBuilder out, String line) {
        if (out.length() > 0) out.append('\n');
        out.append(line);
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
