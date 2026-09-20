package com.musab.aragpt2;

import org.junit.Test;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

public class ConversationFileStoreTest {
    @Test
    public void loadSkipsMalformedLineAndKeepsValidTurns() throws Exception {
        File dir = Files.createTempDirectory("h33-chat-load").toFile();
        File file = new File(dir, "current_chat.jsonl");
        String data =
                "{\"turn_id\":\"t1\",\"role\":\"user\",\"content\":\"سؤال\"}\n"
                + "{broken-json\n"
                + "{\"turn_id\":\"t1\",\"role\":\"assistant\",\"content\":\"جواب\"}\n";
        Files.write(file.toPath(), data.getBytes(StandardCharsets.UTF_8));

        ConversationFileStore store = new ConversationFileStore(file);
        List<ConversationHistory.Turn> turns = store.load();

        assertEquals(2, turns.size());
        assertEquals("t1", turns.get(0).turnId);
        assertEquals("user", turns.get(0).role);
        assertEquals("assistant", turns.get(1).role);
        assertEquals("جواب", turns.get(1).content);
    }

    @Test
    public void saveReplacesOldSnapshotAndLeavesNoTempFile() throws Exception {
        File dir = Files.createTempDirectory("h33-chat-save").toFile();
        File file = new File(dir, "current_chat.jsonl");
        Files.write(
                file.toPath(),
                "{\"turn_id\":\"old\",\"role\":\"user\",\"content\":\"قديم\"}\n"
                        .getBytes(StandardCharsets.UTF_8)
        );

        ConversationFileStore store = new ConversationFileStore(file);
        store.save(Arrays.asList(
                new ConversationHistory.Turn("t2", "user", "سؤال جديد"),
                new ConversationHistory.Turn("t2", "assistant", "جواب جديد")
        ));

        List<ConversationHistory.Turn> turns = store.load();
        assertEquals(2, turns.size());
        assertEquals("t2", turns.get(0).turnId);
        assertEquals("سؤال جديد", turns.get(0).content);
        assertEquals("جواب جديد", turns.get(1).content);
        assertFalse(new File(dir, "current_chat.jsonl.tmp").exists());
    }
}
