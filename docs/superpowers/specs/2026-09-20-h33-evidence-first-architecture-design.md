# H33 — Evidence-First Architecture Design

Date: 2026-09-20  
Target implementation branch: `h33-evidence-first`  
Base branch: `h33-bilingual-bridge` @ `71c2dd1b113a014d6758cfd7ae1b4c5123736be6`

## 1. Decision

H33 will use an evidence-first, three-candidate architecture.

The application will NOT add a learned judge, LitJev scorer, or Qwen-0.5B truth judge in this phase.

The three candidates remain:

1. H33 Local — local Qwen candidate.
2. Web Evidence — answer synthesized only from accepted web evidence.
3. Hosted — Google AI Overview / hosted result when available.

The user is the final authority for preference. The user is not automatically the final authority for objective factual truth.

A user selection or correction becomes:
- user-approved conversational memory;
- a preference event;
- a future training example.

It does not become an unconditional global fact.

When a selected/corrected answer conflicts with strong current web evidence, H33 preserves the user's choice as personal memory/preference and records a factual-conflict flag.

## 2. Implementation strategy

### Recommended: incremental architecture extraction

Build on `h33-bilingual-bridge` and preserve working components:
- CandidateSet / AnswerCandidate model;
- CandidateCoordinator no-auto-commit behavior;
- PreferenceStore and dataset export;
- Google AI Overview provider;
- current Qwen INT4 runtime;
- existing chat/session persistence.

Add clear interfaces around inference, translation, retrieval planning, freshness, evidence policy, and memory policy.

This is preferred over:

### Minimal patch in MainActivity

Faster initially, but rejected because MainActivity is already large and would become the owner of translation, retrieval, evidence, memory, and UI policy.

### Full rewrite

Cleaner in theory, but rejected because it would discard validated candidate, persistence, UI, and Qwen work and increase regression risk.

## 3. Core invariants

1. Candidate generation never commits canonical memory.
2. Only user selection or manual correction commits user-approved conversational memory.
3. A user-approved answer is not silently promoted to objective truth.
4. Current/fresh questions require fresh evidence or explicit abstention.
5. Web Evidence fails closed.
6. Local Qwen is a candidate, not a factual authority.
7. Translation is a reasoning bridge, not a factual authority.
8. Hosted output is an independent candidate, not a factual authority.
9. Preference collection is not weight training.
10. Weight updates happen only in later periodic training runs with replay and evaluation.
11. ONNX Runtime GenAI is hidden behind an application-owned inference interface.

## 4. New component boundaries

### 4.1 LocalInferenceEngine

Create an application-owned interface:

`LocalInferenceEngine`

Responsibilities:
- local text generation;
- optional evidence-constrained generation;
- streaming callback;
- cancellation-compatible generation semantics.

Implementation:

`OrtGenAiLocalInferenceEngine`

wraps the existing `QwenEngine`.

Qwen-specific and ONNX Runtime GenAI-specific calls must not leak into orchestration, translation, retrieval, or preference code.

`QwenEngine` may remain internally during migration, but callers outside the adapter should move toward `LocalInferenceEngine`.

### 4.2 TranslationBridge

Create:

`TranslationBridge`

Operations:
- Arabic -> English;
- English -> Arabic;
- availability/status;
- failure as structured fallback, not exception-driven app collapse.

Implementation:

`MlKitTranslationBridge`

using Google ML Kit Translate.

Rules:
- preserve original user input;
- never overwrite original text;
- if translation model is unavailable/downloading/fails, return fallback state;
- caller uses original-language path;
- numbers, URLs, and named entities are not deliberately transformed by application preprocessing;
- translated text is never stored as the user's original message.

### 4.3 LocalReasoningProvider

Replace direct Qwen local calling with a provider that applies the pivot policy.

Flow for Arabic/Syrian input:

```
original question
  -> TranslationBridge Arabic->English
       -> success: LocalInferenceEngine on English
       -> failure: LocalInferenceEngine on original
  -> optional TranslationBridge English->Arabic for display
       -> failure: show valid generated text in original internal language with status
```

English input goes directly to local inference.

The provider must expose provenance/status such as:
- `local-pivot-en`
- `local-direct-ar`
- `local-direct-en`
- `translation-fallback`

The pivot is a helper path with fallback, never a hard dependency.

## 5. Freshness and intent routing

Create:

`FreshnessPolicy`

This is deterministic and intentionally small.

It detects current-information intent from cues such as:
- current / today / now / latest / news / price;
- Arabic equivalents such as الحالي / اليوم / الآن / آخر / سعر;
- years and explicit temporal wording where relevant.

It must not hardcode political office-holder names.

Output:

`STATIC_OR_GENERAL`
`CURRENT_OR_FRESH`

For CURRENT_OR_FRESH:
- local candidate may still be shown;
- UI/provenance must not present local memory as current truth;
- Web Evidence receives priority labeling;
- if Web Evidence is unavailable, the system abstains from claiming a current fact.

This is routing/policy, not a judge.

## 6. Search query planning

Implement the existing contract:

`SearchQueryPlanner`

Input:
- original query;
- optional English translation.

Produces a plan containing up to four queries:

1. original full;
2. original focused;
3. English full, when translation exists and differs meaningfully;
4. English focused.

Focused query rules:
- remove interrogative/function noise;
- preserve entities;
- preserve numbers;
- preserve years;
- preserve URLs;
- preserve topic-bearing nouns/adjectives.

Examples:

`How many continents are there in the world and what are their names?`
-> `continents world names`

`كم عدد قارات العالم وماهي أسماؤها؟`
-> concepts containing `قارات`, `العالم`, `أسماء` while removing interrogative noise.

## 7. Retrieval flow

Web retrieval becomes multi-query retrieval:

```
SearchQueryPlan
  -> DDG HTML for each unique query
  -> per-result normalization
  -> SearchQualityGate
  -> merge
  -> cross-query dedupe
  -> rank
  -> if insufficient:
       Bing RSS fallback
  -> if still insufficient:
       DDG Instant fallback
```

Network providers remain retrieval mechanisms only.

No provider's mere presence means evidence is usable.

## 8. SearchQualityGate

Strengthen deterministic relevance.

Current known problem:
English interrogative tokens such as `how`, `many`, `what`, `are`, `there` and Arabic terms such as `كم`, `ماهي` can create false relevance.

Update stopword/noise handling.

The gate should use:
- significant token overlap;
- title weighting;
- snippet weighting;
- topic/entity preservation;
- host dedupe;
- deny-list rejection;
- minimum topical-match rule.

Do not add a learned judge.

The gate outputs accepted/rejected evidence and reasons sufficient for diagnostics.

## 9. Web Evidence synthesis

`QwenWebEvidenceAnswerProvider` must be refactored so retrieval and synthesis are separate.

Proposed components:

`WebEvidenceRetriever`
`EvidenceAnswerProvider`

The answer generation contract is strict:

- generate only from accepted evidence;
- do not fill missing facts from local model memory;
- if accepted evidence is insufficient, return unavailable/abstain;
- preserve source mapping;
- current claims require current evidence;
- source count includes accepted sources only.

The local inference engine may synthesize the answer from evidence, but the result remains a Web Evidence candidate with explicit provenance.

## 10. Factual conflict semantics

Create a small record-level policy:

`MemoryConflictStatus`

Values:
- `NONE`
- `POTENTIAL_FACTUAL_CONFLICT`
- `CURRENT_EVIDENCE_CONFLICT`

This does not block the user's preference choice.

When the user chooses/corrects:
- store the selected/corrected answer;
- store the fact that it was user-approved;
- if fresh accepted evidence clearly conflicts, attach a conflict status;
- UI may show a warning such as "اختيارك محفوظ، لكنه يختلف عن أدلة حديثة".

The first implementation should use deterministic comparison signals only:
- exact normalized contradiction when the candidates provide clearly different short facts/numbers/names;
- current-evidence conflict only when accepted fresh evidence exists.

Do not invent semantic truth classification using Qwen-0.5B.

If deterministic conflict cannot be established, use `NONE` rather than guessing.

## 11. Preference and memory schema

Upgrade preference schema from v1 to v2 with backward-compatible reading.

Add fields conceptually:

```json
{
  "memory_semantics": "user_approved",
  "conflict_status": "NONE",
  "freshness_class": "STATIC_OR_GENERAL",
  "selected_provenance": "...",
  "training_eligible": true
}
```

Rules:
- v1 records remain readable;
- v1 maps to user-approved semantics with no conflict metadata;
- raw candidate snapshots remain preserved;
- manual correction remains highest-priority feedback;
- no record claims weight modification.

Conversation history contains only the selected/corrected canonical assistant answer.

Unselected candidates never enter future conversational context.

## 12. Training buffer

Selections and corrections append to the local preference/training buffer.

Do not train on-device after a tap.

Dataset export should continue supporting:
- SFT form;
- preference/DPO-like form.

Add metadata so later training can:
- prioritize manual corrections;
- identify selected candidate origin;
- exclude conflict-marked factual examples if desired;
- replay older validated data.

Actual training, replay ratios, optimizer behavior, and quantized-model replacement remain a separate future training implementation.

## 13. Candidate orchestration

Refactor `CandidateCoordinator` so it owns orchestration only.

Dependencies should become:
- `LocalAnswerProvider`;
- `WebEvidenceAnswerProvider`;
- `HostedAnswerProvider`.

It must not own canonical commit logic.

The existing no-auto-commit invariant remains.

Canonical decision handling should move out of `MainActivity` into:

`UserDecisionService`

Responsibilities:
- validate selection/correction;
- append preference record;
- commit/replace canonical conversation answer atomically where possible;
- calculate/store conflict/freshness metadata;
- return a structured decision result.

MainActivity becomes primarily UI rendering and event forwarding.

## 14. MainActivity refactor

MainActivity currently mixes:
- UI;
- candidate orchestration;
- persistence transaction logic;
- Google interaction;
- preference writing;
- canonical memory replacement.

For this implementation, move only architecture-critical logic out:
- user decision transaction -> `UserDecisionService`;
- local path policy -> provider/bridge;
- search planning/retrieval -> dedicated classes.

Do not perform a broad UI rewrite.

Keep:
- current comparison cards;
- streaming local card;
- copy/TXT;
- correction dialog;
- new chat;
- Google consent/challenge UI;
- existing visual behavior.

Update labels to distinguish:
- user-approved answer;
- web-grounded/current evidence;
- factual-conflict warning when applicable.

## 15. Hosted candidate

Keep `GoogleAiOverviewProvider` behind `HostedAnswerProvider`.

Do not make it part of canonical truth logic.

Rules remain:
- exact hosted result when available;
- unavailable on consent/challenge/no-overview;
- no bypassing access controls;
- failure does not block Local or Web candidates.

## 16. Error handling

### Translation
Failure -> local direct fallback.

### Local inference
Failure -> local candidate unavailable; external candidates continue.

### Retrieval
Failure -> web candidate unavailable unless another fallback provider returns usable evidence.

### Evidence
Insufficient relevance -> fail closed.

### Hosted
Failure/unavailable -> hosted card unavailable.

### Preference persistence
If preference write fails, do not claim selection saved.

### Canonical memory write
If canonical write fails, decision transaction reports failure and must not claim durable success.

### Conflict detection
Uncertain -> no conflict flag rather than guessed conflict.

## 17. Dependencies

Add Google ML Kit Translate to the Android app dependency set.

Keep:
- minSdk 28;
- arm64-v8a;
- Qwen/Qwen2.5-0.5B-Instruct INT4;
- ONNX Runtime GenAI adapter.

No LitJev/JNI logits dependency is added.

## 18. Files expected to be created

Production:
- `LocalInferenceEngine.java`
- `OrtGenAiLocalInferenceEngine.java`
- `TranslationBridge.java`
- `MlKitTranslationBridge.java`
- `PivotingLocalAnswerProvider.java`
- `FreshnessPolicy.java`
- `SearchQueryPlan.java`
- `SearchQueryPlanner.java`
- `WebEvidenceRetriever.java`
- `UserDecisionService.java`
- `MemoryConflictStatus.java`

Likely modified:
- `QwenEngine.java`
- `QwenLocalAnswerProvider.java` or replaced by pivoting provider
- `QwenWebEvidenceAnswerProvider.java`
- `WebSearchClient.java`
- `SearchQualityGate.java`
- `CandidateCoordinator.java`
- `PreferenceRecord.java`
- `PreferenceStore.java`
- `PreferenceDatasetExporter.java`
- `MainActivity.java`
- `app/build.gradle.kts`

## 19. Validation policy

Implementation uses TDD for deterministic Java behavior.

GitHub Actions is not used as the benchmark environment.

Local/deterministic checks cover:
- SearchQueryPlanner;
- SearchQualityGate;
- FreshnessPolicy;
- conflict metadata;
- preference v1 -> v2 migration;
- no candidate auto-commit;
- UserDecisionService transaction behavior.

Android-specific integration validation is limited to actual Android-capable environments.

GitHub is used for repository storage and final APK build/CI, not for architecture benchmark decisions.

Before claiming completion:
- JVM suite must be green in an available execution environment;
- Android lint/build must be green when an Android toolchain is available;
- APK must build successfully on the designated build workflow;
- code review must be completed;
- no unverified claim that ML Kit runtime behavior was tested if no Android runtime was available.

## 20. Acceptance criteria

H33 implementation is accepted when:

- local candidate uses the pivot when translation is available and falls back cleanly;
- original user input is preserved;
- ONNX Runtime GenAI is behind `LocalInferenceEngine`;
- current/fresh questions are identified deterministically;
- SearchQueryPlanner produces full/focused bilingual retrieval queries;
- irrelevant interrogative-noise results no longer qualify as topical evidence;
- Web Evidence fails closed;
- no Qwen/LitJev learned judge exists in the decision path;
- user selection/correction is the only canonical memory commit trigger;
- user-approved memory is distinct from objective factual truth;
- factual conflict metadata can be stored without overriding user preference;
- feedback is buffered/exported, not trained immediately;
- existing three-candidate UI and Google hosted candidate remain functional;
- MainActivity no longer owns architecture-critical decision logic;
- final APK build succeeds.

## 21. Explicitly deferred

Not part of this implementation:
- learned judge;
- LitJev/logit scoring;
- calibrated truth probabilities;
- automatic online weight updates;
- periodic training execution itself;
- semantic conflict classifier;
- remote preference upload;
- redesign of the whole UI.

These may be reconsidered only after real H33 usage yields sufficient evaluation data.
