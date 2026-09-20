# H33 Bilingual Bridge + Judge Implementation Plan

> REQUIRED SUB-SKILL: use superpowers:executing-plans task-by-task with TDD.

**Goal:** Benchmark H33's language and judge hypotheses, then implement only the paths that meet the design gates while fixing canonical-memory and retrieval regressions.

**Architecture:** The benchmark runs outside Android on the same Qwen2.5-0.5B checkpoint with paired EN/MSA/SYR questions. Production changes are conditional: a passing pivot gate enables an ML Kit bilingual bridge; a passing judge gate enables only a secondary logit signal. Memory and retrieval correctness fixes are unconditional.

**Tech Stack:** Python 3.11, PyTorch/Transformers, Hugging Face Datasets, Java/JUnit, Android/Gradle, Google ML Kit when gated, ONNX Runtime GenAI.

**Spec:** docs/superpowers/specs/2026-09-20-h33-bilingual-judge-benchmark-design.md

## Global constraints

- Base production branch is h33-three-candidate-native at 26839bc7218c81e16398277c45bd38b8a50e2dd7.
- Qwen remains Qwen/Qwen2.5-0.5B-Instruct; benchmark must not substitute a larger judge.
- No candidate becomes canonical until user selection/correction.
- Raw judge confidence is diagnostic only.
- No JNI scorer unless the judge gate passes.
- No ML Kit pivot unless the pivot gate passes.

## Task 1 — Real benchmark
Create benchmarks/h33_bilingual_benchmark.py and .github/workflows/h33-bilingual-benchmark.yml. Run it, verify dataset/model/sample/token metadata, and freeze gate booleans from the predeclared formulas without changing thresholds after results.

## Task 2 — Canonical memory TDD
Add a failing CandidateCoordinatorTest proving a successful local candidate causes zero canonical commits before user selection. Run RED, remove eager local commit, then run focused and full JVM tests GREEN.

## Task 3 — Query planning and relevance TDD
Add failing tests for English How many continents... rejecting dictionary pages about many and Arabic كم عدد قارات العالم؟ rejecting interrogative-only matches. Add SearchQueryPlanner tests that preserve topical nouns, entities, and numbers while removing interrogatives. Implement the minimum planner, integrate it into WebSearchClient/SearchQualityGate, then run focused and full JVM tests.

## Task 4 — Conditional bilingual bridge
If pivot_gate_passed=false, record a ruling and skip. If true, add failing tests for translation success, model-unavailable fallback, and entity/number preservation; add ML Kit behind an interface; route local generation through canonical English only when translation succeeds; produce Arabic+English retrieval queries; run full JVM tests and lint.

## Task 5 — Conditional logit judge
If judge_gate_passed=false, record a ruling and skip JNI/logit production code. If true, write a failing deterministic scorer contract test, implement the smallest ORT GenAI native/JNI scorer, compare it with fixed Python fixtures, keep it secondary to evidence/freshness rules, and never expose raw confidence as truth probability.

## Task 6 — End-to-end verification
Run Android JVM tests, lint, Qwen/session smoke, web connectivity smoke, debug APK build and architecture audit. Run the strongest available independent review and fix Critical/Important findings with RED→GREEN tests. Publish an artifact only after applicable gates are green.
