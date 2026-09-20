package com.musab.aragpt2;

import org.junit.Test;

import java.util.Collections;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class CandidateCoordinatorTest {
    private static AnswerCandidate ok(String id, AnswerCandidate.Kind kind, String answer) {
        return AnswerCandidate.available(
                id, kind, id + "-provider", answer,
                Collections.emptyList(), "جاهز");
    }

    @Test
    public void localFailureStillAllowsHostedCandidate() throws Exception {
        LocalAnswerProvider local = (q, listener) -> {
            throw new IllegalStateException("local failed");
        };
        WebEvidenceAnswerProvider web = q ->
                AnswerCandidate.unavailable("web", AnswerCandidate.Kind.WEB,
                        "web-evidence", "web unavailable");
        HostedAnswerProvider hosted = q ->
                ok("hosted", AnswerCandidate.Kind.HOSTED, "جواب مستضاف");

        AtomicInteger commits = new AtomicInteger();
        CandidateCoordinator coordinator = new CandidateCoordinator(
                local, web, hosted,
                (turnId, question, answer) -> commits.incrementAndGet()
        );

        CandidateSet set = coordinator.create("سؤال", null);

        assertFalse(set.local.available);
        assertTrue(set.hosted.available);
        assertTrue(set.hosted.answer.contains("مستضاف"));
        assertTrue(commits.get() == 0);
    }

    @Test
    public void webFailureDoesNotCommitLocalCandidateBeforeSelection() throws Exception {
        LocalAnswerProvider local = (q, listener) ->
                ok("local", AnswerCandidate.Kind.LOCAL, "جواب محلي");
        WebEvidenceAnswerProvider web = q -> {
            throw new IllegalStateException("web failed");
        };
        HostedAnswerProvider hosted = q ->
                AnswerCandidate.unavailable("hosted", AnswerCandidate.Kind.HOSTED,
                        "none", "hosted unavailable");

        AtomicInteger commits = new AtomicInteger();
        CandidateCoordinator coordinator = new CandidateCoordinator(
                local, web, hosted,
                (turnId, question, answer) -> commits.incrementAndGet()
        );

        CandidateSet set = coordinator.create("سؤال", null);

        assertTrue(set.local.available);
        assertFalse(set.web.available);
        assertTrue(commits.get() == 0);
    }

    @Test
    public void hostedFailureDoesNotDiscardLocalOrWebCandidates() throws Exception {
        LocalAnswerProvider local = (q, listener) ->
                ok("local", AnswerCandidate.Kind.LOCAL, "جواب محلي");
        WebEvidenceAnswerProvider web = q ->
                ok("web", AnswerCandidate.Kind.WEB, "جواب بحث");
        HostedAnswerProvider hosted = q -> {
            throw new IllegalStateException("hosted failed");
        };

        CandidateCoordinator coordinator = new CandidateCoordinator(
                local, web, hosted, (turnId, question, answer) -> {}
        );

        CandidateSet set = coordinator.create("سؤال", null);

        assertTrue(set.local.available);
        assertTrue(set.web.available);
        assertFalse(set.hosted.available);
    }
    @Test
    public void webAndHostedCandidatesOverlapAfterLocalCompletes() throws Exception {
        LocalAnswerProvider local = (q, listener) ->
                ok("local", AnswerCandidate.Kind.LOCAL, "جواب محلي");

        CountDownLatch webStarted = new CountDownLatch(1);
        CountDownLatch hostedStarted = new CountDownLatch(1);

        WebEvidenceAnswerProvider web = q -> {
            webStarted.countDown();
            assertTrue(hostedStarted.await(1, TimeUnit.SECONDS));
            return ok("web", AnswerCandidate.Kind.WEB, "جواب بحث");
        };

        HostedAnswerProvider hosted = q -> {
            hostedStarted.countDown();
            assertTrue(webStarted.await(1, TimeUnit.SECONDS));
            return ok("hosted", AnswerCandidate.Kind.HOSTED, "Google AI");
        };

        CandidateCoordinator coordinator = new CandidateCoordinator(
                local, web, hosted, (turnId, question, answer) -> {}
        );

        CandidateSet set = coordinator.create("سؤال", null);

        assertTrue(set.web.available);
        assertTrue(set.hosted.available);
    }

}
