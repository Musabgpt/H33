# H33 Evidence-First Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert H33 to the approved evidence-first architecture while preserving the three-candidate UI and current Qwen/hosted behavior.

**Architecture:** Build on `h33-evidence-first`. Extract local inference behind an app-owned interface, add ML Kit pivot/fallback, add deterministic freshness/query planning/evidence filtering, move canonical decision persistence out of MainActivity, and upgrade preference records to user-approved-memory semantics with conflict metadata.

**Tech Stack:** Java, Android minSdk 28 / targetSdk 36, Qwen2.5-0.5B INT4, ONNX Runtime GenAI, Google ML Kit Translate 17.0.3, JUnit 4.

**Spec:** `docs/superpowers/specs/2026-09-20-h33-evidence-first-architecture-design.md`

## Global Constraints

- No learned judge or LitJev path.
- Candidate generation never commits canonical memory.
- User choice/correction is user-approved memory, not automatic objective truth.
- Current/fresh questions require fresh evidence or abstention.
- Web Evidence fails closed.
- Translation is optional with direct-language fallback.
- Keep three candidate slots and Google AI Overview provider.
- GitHub is repository/build CI, not benchmark environment.

## Review Focus

1. Arabic/English detection must not corrupt URLs/numbers or English-only input.
2. ML Kit unavailability/download failure must fall back to direct local inference.
3. Multi-query retrieval must dedupe without treating question words as topical evidence.
4. v1 preference records must remain readable after schema v2 migration.
5. A failed preference/canonical write must not produce a false saved-success state.

---

### Task 1: Core policy and inference boundaries

**Files:**
- Create: `LocalInferenceEngine.java`
- Create: `OrtGenAiLocalInferenceEngine.java`
- Create: `FreshnessPolicy.java`
- Create: `SearchQueryPlan.java`
- Create: `SearchQueryPlanner.java`
- Modify: `SearchQualityGate.java`
- Test: existing `SearchQueryPlannerTest.java`, `SearchQualityGateTest.java`; add `FreshnessPolicyTest.java`

**Produces:** deterministic query planning/freshness and app-owned inference interface.

- [ ] Add RED tests for freshness and focused query behavior.
- [ ] Implement pure-Java policies.
- [ ] Harden stopword/noise handling in SearchQualityGate.
- [ ] Add ONNX adapter without changing QwenEngine internals yet.
- [ ] Run pure-Java local checks where possible.
- [ ] Commit.

### Task 2: Translation bridge and pivoted local candidate

**Files:**
- Create: `TranslationBridge.java`
- Create: `MlKitTranslationBridge.java`
- Create: `PivotingLocalAnswerProvider.java`
- Modify: `app/build.gradle.kts`
- Modify: `MainActivity.java` wiring only.

**Produces:** Arabic->English->Qwen->Arabic helper path with direct fallback.

- [ ] Define structured translation result/fallback contract.
- [ ] Add ML Kit dependency 17.0.3.
- [ ] Implement model download/use without blocking app startup.
- [ ] Use original question on any translation failure.
- [ ] Preserve original user input in conversation/preference data.
- [ ] Commit.

### Task 3: Evidence retrieval and fail-closed synthesis

**Files:**
- Create: `WebEvidenceRetriever.java`
- Modify: `WebSearchClient.java`
- Modify: `QwenWebEvidenceAnswerProvider.java`

**Produces:** full/focused bilingual retrieval, merge/dedupe/filter, accepted-evidence-only synthesis.

- [ ] Split retrieval from answer synthesis.
- [ ] Execute unique planned queries across DDG HTML.
- [ ] Merge and quality-filter results; use existing fallbacks only when accepted set is insufficient.
- [ ] Return unavailable if accepted evidence is empty/insufficient.
- [ ] Keep sources mapped to accepted evidence only.
- [ ] Commit.

### Task 4: User-approved memory semantics and decision service

**Files:**
- Create: `MemoryConflictStatus.java`
- Create: `UserDecisionService.java`
- Modify: `PreferenceRecord.java`
- Modify: `PreferenceStore.java`
- Modify: `PreferenceDatasetExporter.java`
- Modify: `MainActivity.java`

**Produces:** schema v2, backward-compatible v1 read, decision transaction outside UI, conflict metadata.

- [ ] Add v2 fields: memory_semantics, conflict_status, freshness_class, selected_provenance, training_eligible.
- [ ] Keep v1 parser compatibility.
- [ ] Move preference+canonical transaction from MainActivity to UserDecisionService.
- [ ] Add deterministic current-evidence conflict marker.
- [ ] Keep user choice even on conflict and surface warning.
- [ ] Remove immediate correction-as-global-truth behavior.
- [ ] Commit.

### Task 5: Coordinator/UI integration cleanup

**Files:**
- Modify: `CandidateCoordinator.java`
- Modify: `MainActivity.java`
- Modify: provider wiring.

**Produces:** orchestration-only coordinator and thinner MainActivity.

- [ ] Remove obsolete canonical store dependency from CandidateCoordinator.
- [ ] Wire LocalInferenceEngine + TranslationBridge + retriever + UserDecisionService.
- [ ] Preserve streaming/local/web/hosted cards, copy/TXT, correction/new-chat and Google consent/challenge UI.
- [ ] Label user-approved memory and factual conflict clearly.
- [ ] Commit.

### Task 6: Build and review

**Files:**
- Modify only what build/review findings require.

- [ ] Perform local pure-Java compile/checks available in this environment.
- [ ] Run CodeRabbit/available review tooling on branch diff.
- [ ] Fix Important/Critical findings.
- [ ] Trigger existing APK build workflow on `h33-evidence-first` only after implementation.
- [ ] Read actual build logs.
- [ ] If green, fetch APK artifact and report exact status; if red, debug the concrete build error only.
