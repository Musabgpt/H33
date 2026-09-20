# H33 — Three-Candidate Answer Selection & Preference Learning Design

Date: 2026-09-20  
Target branch: `qwen2.5-mobile`  
Future training branch: `qwen-partial-train`

## 1. Purpose

H33 should stop treating one local answer plus raw search results as a single answer path.

For each question, H33 will produce up to three clearly separated candidate answers:

1. **H33 Local** — Qwen2.5-0.5B-Instruct running locally without web evidence.
2. **Web Evidence Answer** — a separate answer produced only from filtered, relevant web evidence.
3. **Browser/Hosted Model Answer** — an answer from an external model/search provider when an official supported provider is configured.

The user is the final judge. They may choose any candidate as best, or reject all candidates and provide a correction.

The selected answer becomes the canonical answer for the conversation and is stored as training evidence for later SFT/DPO-style fine-tuning.

## 2. Success criteria

A successful implementation must:

- Show the three answer paths separately and label their provenance.
- Never label a response as web-grounded unless usable relevant evidence exists.
- Reject clearly irrelevant, adult, spam, duplicate, or low-relevance search results before they reach the answer generator.
- Never fabricate a browser-model answer when the configured provider is unavailable.
- Preserve current session memory and make the selected/corrected answer the canonical assistant turn.
- Persist user preference records locally in a structured, exportable format.
- Preserve the current TXT export, copy, correction, new-chat, streaming, and Fable behavior.
- Produce data that can later be transformed into SFT and preference/DPO datasets.
- Keep current Qwen INT4 inference stable while training work remains isolated in `qwen-partial-train`.

## 3. Non-goals for the first implementation phase

The first phase will not:

- Update Qwen base weights on every tap.
- Claim that storing a correction is equivalent to weight training.
- Scrape private/authenticated browser UI pages to impersonate Google or Microsoft model output.
- Hardcode API keys or credentials in the APK or repository.
- Replace Qwen INT4 inference with a training runtime.
- Automatically publish user feedback to a remote service.

## 4. High-level architecture

Introduce four explicit layers:

### 4.1 Candidate orchestration

A new coordinator owns one question lifecycle and produces a `CandidateSet`.

Conceptually:

```
Question
  ├── LocalCandidateProvider
  ├── WebEvidenceProvider
  └── HostedModelProvider (optional)
          ↓
      CandidateSet
          ↓
   Selection / correction
          ↓
   PreferenceStore
          ↓
 Canonical conversation answer
```

The three candidate paths must not silently overwrite each other.

### 4.2 Local candidate

The current `QwenEngine` remains the source of the local H33 answer.

Rules:

- No web context.
- Existing Fable prompt remains active.
- Existing explicit recent-session context remains active.
- The result may be stored as the provisional assistant answer for continuity.
- If the user later chooses another candidate or supplies a correction, the provisional answer is replaced by the canonical one in conversation history.

### 4.3 Web evidence candidate

`WebSearchClient` becomes a retrieval component, not the authority that declares search success.

A new web-quality stage must run before answer synthesis.

Each result should contain at least:

- title
- URL
- hostname
- snippet
- relevance score
- rejection reason when filtered

The quality gate must include:

1. URL/host validation.
2. Adult/spam/malware-domain rejection.
3. Duplicate-domain and near-duplicate-result suppression.
4. Query relevance scoring using normalized terms/entities.
5. Minimum evidence threshold before declaring a web answer available.
6. Preferential ranking for authoritative or first-party sources when applicable.
7. No source count inflation from rejected results.

The Web Evidence Answer is generated from the accepted evidence only.

Its prompt must explicitly instruct the generator:

- use only supplied evidence;
- do not answer from model memory when evidence does not support the answer;
- state that evidence is insufficient when necessary;
- attach source references that map to the accepted evidence set.

This path may reuse the local Qwen runtime for synthesis, but it is a separate evidence-constrained generation path and must not be presented as the same answer as H33 Local.

### 4.4 Hosted/browser-model candidate

Add a provider interface such as:

```java
interface HostedAnswerProvider {
    Availability availability();
    HostedAnswer answer(String question) throws Exception;
}
```

Initial implementation should support an official provider only when configured.

Provider rules:

- Never scrape or parse changing consumer UI pages as the primary integration.
- Never claim "Google", "Microsoft", "Copilot", "Gemini", or another provider unless the returned answer actually came from that configured provider.
- If no provider is configured, return a structured `UNAVAILABLE` state.
- Provider errors must not break H33 Local or Web Evidence candidates.
- Credentials must be supplied through a safe configuration path, not committed to Git.

Google-search-grounded Gemini is the preferred first hosted provider if a supported API configuration is available. Microsoft can be added behind the same interface later.

## 5. Search quality gate

The current failure mode is that Bing RSS results are accepted solely because they exist.

That must change.

### 5.1 Result filtering

A result is rejected before use if any of the following apply:

- domain is on a deny list for adult/spam/malicious content;
- title/snippet contains strong adult/spam indicators unrelated to the query;
- URL is malformed or unsupported;
- result is a duplicate of an already accepted result;
- relevance score is below the configured threshold.

### 5.2 Relevance

Relevance should be deterministic and testable.

Initial scoring can combine:

- normalized token overlap between query and title/snippet;
- entity/phrase overlap;
- title match weighting;
- recency cue weighting for explicitly current queries;
- domain-quality weighting.

The implementation may later be improved, but phase 1 must not rely on "search engine returned it, therefore it is relevant."

### 5.3 Regression case

The exact class of failure shown by the user must become a regression test:

- Query: `من هو رئيس سوريا الحالي؟`
- Irrelevant adult-domain results must be rejected.
- A web answer must not be marked usable from those rejected results.
- The test must not hardcode a political office-holder name; it tests filtering/provenance, not a static political answer.

## 6. UI/UX

Each user question produces one comparison group.

Recommended card order:

1. **H33 المحلي**
2. **جواب البحث**
3. **جواب نموذج المتصفح**

Each available card contains:

- provider label;
- answer text;
- provenance/status badge;
- sources when applicable;
- `اختر هذا الجواب`;
- existing copy/TXT actions where sensible.

If a candidate is unavailable, show a compact unavailable state instead of a fabricated answer.

Below the candidate group:

- `كلهم خطأ — سأكتب التصحيح`

When the user chooses an answer:

- visually mark it as selected;
- persist the preference;
- make it the canonical assistant answer in conversation memory;
- disable accidental duplicate submission for that comparison group;
- allow correction later, which supersedes the previous selection.

## 7. Preference data model

Create a dedicated `PreferenceStore`; do not overload `CorrectionMemory`.

Suggested JSONL record:

```json
{
  "schema_version": 1,
  "id": "uuid",
  "timestamp_ms": 0,
  "question": "...",
  "session_context": "...",
  "candidates": [
    {
      "id": "local",
      "provider": "qwen2.5-0.5b-int4",
      "answer": "...",
      "available": true,
      "sources": []
    },
    {
      "id": "web",
      "provider": "web-evidence",
      "answer": "...",
      "available": true,
      "sources": [
        {"title": "...", "url": "...", "host": "..."}
      ]
    },
    {
      "id": "hosted",
      "provider": "configured-provider-name",
      "answer": "...",
      "available": false,
      "sources": []
    }
  ],
  "selection": {
    "type": "candidate",
    "candidate_id": "web"
  },
  "correction": null
}
```

If the user writes a correction:

```json
{
  "selection": {"type": "user_correction", "candidate_id": null},
  "correction": "..."
}
```

A later correction should supersede the earlier choice for canonical conversation behavior while preserving the historical record.

## 8. Conversation-memory behavior

The app currently preserves recent turns and injects explicit `SESSION_CONTEXT`.

Keep that behavior.

For a comparison turn:

1. User question is added normally.
2. H33 Local answer may be added provisionally so the next turn has continuity even before selection.
3. When the user chooses Web, Hosted, or a manual correction, replace the provisional assistant answer with the chosen canonical answer.
4. Save the resulting conversation atomically.

This prevents all three candidates from polluting future context.

## 9. Training data pipeline

Preference collection and weight training are separate stages.

### 9.1 Immediate behavior

Immediately after a choice/correction:

- save the preference locally;
- update canonical conversation memory;
- optionally use exact-answer memory for repeated identical questions.

No claim of base-weight modification is made.

### 9.2 Dataset export

Add an export path that can transform stored records into:

**SFT examples**

```
prompt -> selected/corrected answer
```

**Preference/DPO examples**

```
prompt
chosen  -> selected answer
rejected -> non-selected candidate
```

If the user supplied a manual correction, that correction has the highest training priority.

### 9.3 Partial Qwen training

Training work belongs in `qwen-partial-train`.

Before training work starts:

- sync it with the latest validated `qwen2.5-mobile`;
- keep the stable mobile branch untouched by experimental training code;
- define a reproducible baseline evaluation set;
- train only a controlled subset/adapter strategy first;
- compare before/after quality and regression behavior;
- only export a new INT4 mobile model after evaluation passes.

The exact trainable parameter set/runtime is an implementation decision for the later training spec. This design does not pretend the current ORT GenAI INT4 package is directly trainable in place.

## 10. Testing strategy

### 10.1 Unit/static tests

Add deterministic tests for:

- relevance scoring;
- adult/spam-domain filtering;
- duplicate suppression;
- empty/failed search behavior;
- provenance labeling;
- preference JSON serialization;
- correction superseding a previous selection;
- canonical conversation replacement.

### 10.2 CI integration tests

GitHub Actions must verify more than "Bing returned an item."

Add tests that assert:

- irrelevant seeded results are rejected;
- usable search requires accepted relevant results;
- session-context smoke test still passes;
- app builds successfully;
- APK architecture audit still passes.

Network-dependent search smoke tests should be isolated from deterministic filtering tests so an external search outage is distinguishable from an H33 regression.

### 10.3 Android emulator QA

Validate on an Android emulator:

- ask a normal question;
- receive local/web/hosted cards;
- hosted-unavailable behavior;
- select each candidate type;
- provide manual correction;
- start a follow-up question and verify canonical context;
- keyboard behavior;
- copy/TXT actions;
- new-chat behavior;
- app restart persistence;
- logcat for crashes/errors.

### 10.4 Performance

Measure before/after:

- time to first local token;
- total latency for three-candidate turn;
- peak PSS/RAM;
- UI jank during streaming;
- impact of web/provider calls on responsiveness.

The three providers should be parallelized only where thread-safety and device memory permit. Qwen generation must not be invoked concurrently against a runtime that is not safe for it.

## 11. Review and quality gates

Before calling the implementation complete:

1. Local tests pass.
2. GitHub Actions pass.
3. Android emulator QA passes.
4. Performance regression is checked.
5. CodeRabbit review is completed.
6. SonarQube analysis/quality checks are run when integration is available.
7. Changes are submitted through a PR.
8. GH Review Loop processes actionable reviewer feedback until clean, capped, or a human decision is required.
9. Final verification is run before merge.

## 12. Migration plan

Phase A — stabilize and refactor search:
- introduce structured search results;
- add deterministic filtering/relevance tests;
- fix provenance.

Phase B — candidate comparison UI:
- local candidate;
- web-evidence candidate;
- optional hosted candidate;
- selection/correction UX.

Phase C — preference persistence:
- `PreferenceStore`;
- canonical conversation replacement;
- dataset export.

Phase D — hosted provider:
- add one official provider implementation;
- secure configuration;
- graceful unavailable state.

Phase E — training:
- sync `qwen-partial-train`;
- build SFT/DPO dataset;
- run controlled partial fine-tuning;
- evaluate;
- quantize and integrate only after passing regression checks.

## 13. Failure handling

- Local model failure: show local candidate as unavailable; do not block independent external candidates.
- Search failure: show web candidate unavailable; do not present local memory as web evidence.
- Hosted provider failure: show hosted candidate unavailable; do not block other candidates.
- Persistence failure: do not claim a preference/correction was saved.
- Training export failure: preserve original preference store and fail without mutating it.

## 14. Security and privacy

- User questions, choices, and corrections remain local by default.
- External providers receive only the data required for the specific external request.
- Do not send local correction history to a hosted provider unless explicitly designed and disclosed later.
- Never commit credentials.
- Validate URLs before fetching and restrict unsafe schemes.
- Keep model/provider provenance visible to the user.

## 15. Acceptance checklist

The feature is ready only when all of these are true:

- [ ] Three candidate slots exist with honest availability/provenance.
- [ ] Web results pass relevance and domain filtering before use.
- [ ] The reported adult-domain failure class is covered by a regression test.
- [ ] User can select local/web/hosted or write a correction.
- [ ] Selection is persisted in structured preference JSONL.
- [ ] Selected/corrected answer becomes canonical conversation memory.
- [ ] Existing explicit session-context test continues to pass.
- [ ] No UI regression in keyboard/chat behavior.
- [ ] Dataset export produces SFT and preference-compatible records.
- [ ] GitHub Actions pass.
- [ ] Android emulator QA passes.
- [ ] CodeRabbit review completed.
- [ ] SonarQube checks completed when available.
- [ ] GH Review Loop reaches a clean or explicitly bounded terminal state.

