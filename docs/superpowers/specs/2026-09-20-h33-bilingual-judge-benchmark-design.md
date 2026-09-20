# H33 Bilingual Bridge + Judge Benchmark Design

## Goal

Measure, before production changes, whether H33 benefits from an English reasoning pivot and whether a LitJev-style logit judge is reliable enough to participate in answer selection. Then implement only the components whose predeclared gates pass.

## Current baseline

H33 runs Qwen/Qwen2.5-0.5B-Instruct as an INT4 ONNX Runtime GenAI model on Android. The UI exposes three independent candidates: local Qwen, Web Evidence, and Google AI Overview. Current code also commits the local candidate to conversation state before the user selects an answer, and SearchQualityGate uses exact lexical overlap with incomplete stopword sets.

## Benchmark questions

### Language path

Use parallel English, MSA, and Syrian rows from MBZUAI/Dialectal-Arabic-MMLU. For the same (domain, qid) evaluate:

- English question + English instruction.
- MSA question + Arabic instruction.
- MSA question + English instruction.
- Syrian question + Arabic instruction.
- Syrian question + English instruction.

The English row is a human-parallel upper bound for a perfect Arabic-to-English pivot; it is not treated as proof that ML Kit preserves every Syrian input.

### Judge path

Build deterministic evidence-grounded A/B/C/N cases from the same benchmark. In supported cases exactly one candidate matches explicit evidence. In abstention cases evidence is intentionally insufficient and all candidates are wrong. Compare direct next-token logit scoring with greedy free-text judging.

Record accuracy, abstention accuracy, latency, and raw-confidence ECE. Raw confidence is diagnostic only and must not be displayed as probability of truth.

## Predeclared gates

### English pivot gate

A pivot trial is justified only when EN+EN instruction beats Syrian direct (SYR+AR instruction) by at least 3 percentage points and the paired bootstrap 95% CI lower bound is above zero. MSA is reported separately.

### Logit judge gate

The judge may be integrated only as a secondary decision signal when:
- overall accuracy is at least 80%;
- abstention accuracy is at least 70%;
- logit scoring is no worse than generated judging by more than 2 percentage points; and
- logit scoring is at least 25% faster than generated judging.

Evidence/freshness hard rules remain higher priority and the user remains the only source of canonical preference truth.

## Production invariants independent of benchmark outcome

- Unselected local/web/hosted candidates never enter canonical conversation memory.
- User selection or manual correction is the only canonical commit event.
- Current/freshness questions require fresh evidence and may abstain.
- Search query planning must remove interrogative/function-word retrieval noise while preserving entities, numbers, and topical concepts.
- Web candidates remain provenance-preserving and fail closed when no relevant evidence is available.

## Conditional implementation

If the pivot gate passes, add an on-device bilingual bridge using Google ML Kit Translation with graceful fallback to the original language when the translation model is unavailable. Preserve the original Arabic text and names/numbers. Use English as the internal local-generation language and generate both Arabic and English search queries.

If the pivot gate fails, do not add translation to the local model path. Still fix query planning, evidence relevance, freshness, and memory semantics.

If the logit judge gate passes, prototype a small isolated scorer around ORT GenAI native logits. Do not ship a probability-to-user UI. If the gate fails, use deterministic evidence/freshness/agreement rules only.

## Validation

Production work uses TDD. Required regression tests cover memory no-auto-commit, English many/how/what retrieval noise, Arabic كم/ماهي retrieval noise, entity preservation, fallback when translation is unavailable, and abstention when fresh evidence conflicts or is missing. Final validation requires Android JVM tests, lint, Qwen smoke, network smoke, APK build/audit, and an independent code review.
