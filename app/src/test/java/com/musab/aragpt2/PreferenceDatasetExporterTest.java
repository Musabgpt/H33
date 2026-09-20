package com.musab.aragpt2;

import org.json.JSONObject;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class PreferenceDatasetExporterTest {
    @Test
    public void selectedWebAnswerBecomesSftAndRejectsOtherAvailableCandidates() throws Exception {
        CandidateSet set = new CandidateSet(
                "turn-web",
                "ما الجواب؟",
                available("local", AnswerCandidate.Kind.LOCAL, "qwen", "جواب محلي"),
                available("web", AnswerCandidate.Kind.WEB, "web-evidence", "جواب البحث"),
                AnswerCandidate.unavailable(
                        "hosted", AnswerCandidate.Kind.HOSTED, "none", "غير متاح")
        );
        PreferenceRecord record =
                PreferenceRecord.selection(set, "web", 100L);

        PreferenceDatasetExporter exporter = new PreferenceDatasetExporter();

        JSONObject sft = new JSONObject(exporter.toSftJsonl(
                Collections.singletonList(record)).trim());
        assertEquals("ما الجواب؟", sft.getString("prompt"));
        assertEquals("جواب البحث", sft.getString("response"));
        assertEquals("turn-web", sft.getString("id"));

        List<String> dpoLines = nonEmptyLines(exporter.toPreferenceJsonl(
                Collections.singletonList(record)));
        assertEquals(1, dpoLines.size());

        JSONObject dpo = new JSONObject(dpoLines.get(0));
        assertEquals("جواب البحث", dpo.getString("chosen"));
        assertEquals("جواب محلي", dpo.getString("rejected"));
    }

    @Test
    public void manualCorrectionBecomesChosenAgainstEveryMachineCandidate() throws Exception {
        CandidateSet set = new CandidateSet(
                "turn-correction",
                "صحح هذا",
                available("local", AnswerCandidate.Kind.LOCAL, "qwen", "أ"),
                available("web", AnswerCandidate.Kind.WEB, "web-evidence", "ب"),
                available("hosted", AnswerCandidate.Kind.HOSTED, "gemini", "ج")
        );
        PreferenceRecord record =
                PreferenceRecord.correction(set, "هذا هو الجواب الصحيح", 200L);

        PreferenceDatasetExporter exporter = new PreferenceDatasetExporter();

        JSONObject sft = new JSONObject(exporter.toSftJsonl(
                Collections.singletonList(record)).trim());
        assertEquals("هذا هو الجواب الصحيح", sft.getString("response"));

        List<String> dpoLines = nonEmptyLines(exporter.toPreferenceJsonl(
                Collections.singletonList(record)));
        assertEquals(3, dpoLines.size());

        for (String line : dpoLines) {
            JSONObject dpo = new JSONObject(line);
            assertEquals("هذا هو الجواب الصحيح", dpo.getString("chosen"));
        }
    }

    @Test
    public void recordsWithoutEffectiveAnswerAreSkipped() {
        PreferenceDatasetExporter exporter = new PreferenceDatasetExporter();
        assertEquals("", exporter.toSftJsonl(Collections.emptyList()));
        assertEquals("", exporter.toPreferenceJsonl(Collections.emptyList()));
    }

    private static AnswerCandidate available(
            String id, AnswerCandidate.Kind kind, String provider, String answer) {
        return AnswerCandidate.available(
                id, kind, provider, answer,
                Collections.emptyList(), "جاهز");
    }

    private static List<String> nonEmptyLines(String text) {
        if (text == null || text.trim().isEmpty()) return Collections.emptyList();
        return Arrays.asList(text.trim().split("\\R+"));
    }
}
