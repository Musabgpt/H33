package com.musab.aragpt2;

import org.json.JSONObject;
import org.junit.Test;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class PreferenceRecordTest {
    private static CandidateSet sampleSet(String turnId) {
        SearchResult source = new SearchResult(
                "Official source", "https://official.example/a", "evidence");
        AnswerCandidate local = AnswerCandidate.available(
                "local", AnswerCandidate.Kind.LOCAL, "qwen2.5-0.5b-int4",
                "local answer", Collections.emptyList(), "محلي");
        AnswerCandidate web = AnswerCandidate.available(
                "web", AnswerCandidate.Kind.WEB, "web-evidence",
                "web answer", Arrays.asList(source), "ويب");
        AnswerCandidate hosted = AnswerCandidate.unavailable(
                "hosted", AnswerCandidate.Kind.HOSTED, "none", "غير متاح");
        return new CandidateSet(turnId, "question", local, web, hosted);
    }

    @Test
    public void selectionSerializesAllCandidatesAndSources() throws Exception {
        PreferenceRecord record =
                PreferenceRecord.selection(sampleSet("t1"), "web", 1234L);

        JSONObject json = new JSONObject(record.toJson());

        assertEquals(2, json.getInt("schema_version"));\n        assertEquals("user_approved", json.getString("memory_semantics"));
        assertEquals("t1", json.getString("turn_id"));
        assertEquals("question", json.getString("question"));
        assertEquals(3, json.getJSONArray("candidates").length());
        assertEquals("candidate",
                json.getJSONObject("selection").getString("type"));
        assertEquals("web",
                json.getJSONObject("selection").getString("candidate_id"));
        assertTrue(record.effectiveAnswer().contains("web answer"));
        assertTrue(record.toJson().contains("https://official.example/a"));
    }

    @Test
    public void correctionSupersedesEarlierSelection() throws Exception {
        File dir = Files.createTempDirectory("h33-pref").toFile();
        File file = new File(dir, "prefs.jsonl");
        PreferenceStore store = new PreferenceStore(file);

        store.append(PreferenceRecord.selection(sampleSet("t1"), "web", 1L));
        store.append(PreferenceRecord.correction(
                sampleSet("t1"), "الجواب الصحيح", 2L));

        PreferenceRecord latest = store.latestDecision("t1");
        assertNotNull(latest);
        assertEquals("user_correction", latest.selectionType);
        assertEquals("الجواب الصحيح", latest.effectiveAnswer());
    }

    @Test
    public void truncatedFinalLineDoesNotDestroyEarlierRecords() throws Exception {
        File dir = Files.createTempDirectory("h33-pref-crash").toFile();
        File file = new File(dir, "prefs.jsonl");
        PreferenceStore store = new PreferenceStore(file);
        store.append(PreferenceRecord.selection(sampleSet("t1"), "local", 1L));

        try (FileOutputStream out = new FileOutputStream(file, true)) {
            out.write("{\"schema_version\":1".getBytes(StandardCharsets.UTF_8));
            out.getFD().sync();
        }

        assertEquals(1, store.loadAll().size());
        assertEquals("t1", store.loadAll().get(0).turnId);
    }
}
