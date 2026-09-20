# H33 Candidate Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace H33's single mixed local/web answer path with a three-slot comparison flow whose local and web candidates are independent, whose web evidence is filtered before use, and whose user selection/correction is persisted as preference-training data.

**Architecture:** Keep Qwen2.5 INT4 as the local inference engine, but separate candidate generation from conversation persistence. Search becomes structured retrieval followed by a deterministic quality gate. Each question receives a stable turn ID; one provisional local answer is written to conversation history, and any later user choice replaces that exact turn by ID. Preference data is append-only JSONL and exportable to SFT/DPO-compatible JSONL.

**Tech Stack:** Android Java, ONNX Runtime GenAI 0.15.2, Qwen2.5-0.5B-Instruct INT4, org.json, JUnit 4, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-20-h33-three-candidate-preference-learning-design.md`

## Global Constraints

- Target branch for stable app work: `qwen2.5-mobile`.
- Keep Qwen INT4 inference stable; training work remains isolated in `qwen-partial-train`.
- Android remains ARM64-only with `minSdk=28`, `targetSdk=36`, and `compileSdk=36`.
- Preserve Fable prompt behavior, explicit recent-session context, streaming, copy, TXT export, new-chat, and keyboard/insets behavior.
- Never label a response as web-grounded unless accepted relevant evidence exists.
- Reject clearly irrelevant, adult, spam, duplicate, or low-relevance search results before generation.
- Never commit credentials.
- User selections/corrections remain local by default.
- Do not claim local preference storage modifies Qwen base weights.

## Review Focus

- Search returns only denied/adult domains: expected web candidate = unavailable; pinned in Task 1.
- Search returns mixed relevant + irrelevant results in Arabic/English: expected accepted set contains only relevant unique results; pinned in Task 1.
- User chooses an older comparison card after a newer turn exists: expected only the matching `turnId` is replaced; pinned in Task 3.
- App crashes/restarts after a preference append: expected prior complete JSONL records remain readable and the newest partial line is ignored; pinned in Task 5.
- User corrects a previously selected candidate: expected correction supersedes canonical conversation answer while preserving prior preference history; pinned in Task 5 and Task 6.

---

### Task 1: Deterministic web-result quality gate

**Files:**
- Modify: `app/build.gradle.kts`
- Create: `app/src/main/java/com/musab/aragpt2/SearchResult.java`
- Create: `app/src/main/java/com/musab/aragpt2/SearchQualityGate.java`
- Create: `app/src/test/java/com/musab/aragpt2/SearchQualityGateTest.java`

**Interfaces:**
- Produces: `SearchResult`, `SearchQualityGate.filter(String, List<SearchResult>, int)`
- Consumes: no Android framework APIs; this layer must stay plain Java and unit-testable.

- [ ] **Step 1: Add the JUnit dependency**

In `app/build.gradle.kts`, add:

```kotlin
dependencies {
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("com.google.android.material:material:1.13.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.0")
    implementation(files("libs/onnxruntime-genai.aar"))
    testImplementation("junit:junit:4.13.2")
}
```

- [ ] **Step 2: Write failing quality-gate tests**

Create `SearchQualityGateTest.java` with tests equivalent to:

```java
@Test
public void rejectsAdultDomainsEvenWhenSearchReturnedThem() {
    List<SearchResult> raw = Arrays.asList(
        new SearchResult("Today's selection - XNXX.COM",
            "https://www.xnxx.com/todays-selection/", "adult content"),
        new SearchResult("Free Porn Videos", "https://xnxx.dev/", "xxx sex videos")
    );

    SearchQualityGate.Result result =
        SearchQualityGate.filter("من هو رئيس سوريا الحالي؟", raw, 5);

    assertTrue(result.accepted.isEmpty());
    assertEquals(2, result.rejected.size());
}

@Test
public void keepsRelevantAndDropsIrrelevantResults() {
    List<SearchResult> raw = Arrays.asList(
        new SearchResult("رئاسة الجمهورية العربية السورية",
            "https://example.gov.sy/president", "رئيس الجمهورية السورية"),
        new SearchResult("Football scores",
            "https://sports.example/score", "latest match results")
    );

    SearchQualityGate.Result result =
        SearchQualityGate.filter("من هو رئيس سوريا الحالي؟", raw, 5);

    assertEquals(1, result.accepted.size());
    assertEquals("example.gov.sy", result.accepted.get(0).host);
}

@Test
public void removesDuplicateHostsAfterBestResult() {
    List<SearchResult> raw = Arrays.asList(
        new SearchResult("Syria president official",
            "https://example.org/a", "Syria president"),
        new SearchResult("Another result",
            "https://example.org/b", "Syria president current")
    );

    SearchQualityGate.Result result =
        SearchQualityGate.filter("Syria current president", raw, 5);

    assertEquals(1, result.accepted.size());
}
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
gradle :app:testDebugUnitTest --tests 'com.musab.aragpt2.SearchQualityGateTest' --stacktrace
```

Expected: FAIL because `SearchResult` and `SearchQualityGate` do not exist.

- [ ] **Step 4: Implement `SearchResult`**

Use this shape:

```java
public final class SearchResult {
    public final String title;
    public final String url;
    public final String snippet;
    public final String host;
    public final double relevance;

    public SearchResult(String title, String url, String snippet) {
        this(title, url, snippet, safeHost(url), 0.0);
    }

    public SearchResult(String title, String url, String snippet,
                        String host, double relevance) {
        this.title = clean(title);
        this.url = clean(url);
        this.snippet = clean(snippet);
        this.host = clean(host).toLowerCase(Locale.ROOT);
        this.relevance = relevance;
    }

    public SearchResult withRelevance(double score) {
        return new SearchResult(title, url, snippet, host, score);
    }
}
```

`safeHost` must parse `java.net.URI`, strip a leading `www.`, and return an empty string on failure.

- [ ] **Step 5: Implement `SearchQualityGate`**

Use deterministic filtering with these rules:

```java
private static final Set<String> DENIED_HOST_FRAGMENTS = new HashSet<>(Arrays.asList(
    "xnxx", "xvideos", "pornhub", "redtube", "youporn", "xhamster"
));

private static final Set<String> AR_STOP = new HashSet<>(Arrays.asList(
    "من", "هو", "هي", "ما", "ماذا", "هل", "في", "على", "الى", "إلى",
    "الحالي", "حاليا", "حالياً", "اليوم", "الآن", "الان"
));

private static final Set<String> EN_STOP = new HashSet<>(Arrays.asList(
    "who", "is", "the", "a", "an", "current", "today", "now", "of", "in"
));
```

Compute relevance from normalized significant query tokens:

```java
double titleCoverage = coverage(queryTokens, tokens(r.title));
double bodyCoverage = coverage(queryTokens, tokens(r.snippet));
double score = Math.min(1.0, titleCoverage * 0.65 + bodyCoverage * 0.35);
boolean relevant = matchedCount(queryTokens, tokens(r.title + " " + r.snippet)) > 0
        && score >= 0.18;
```

Reject malformed/empty hosts, denied hosts, irrelevant results, then keep only the highest-scored result per host and cap to `maxResults`.

Return:

```java
public static final class Result {
    public final List<SearchResult> accepted;
    public final List<SearchResult> rejected;
}
```

- [ ] **Step 6: Re-run tests**

Run the same Gradle command.

Expected: PASS.

- [ ] **Step 7: Add the review-focus mixed-language test**

Add this deterministic bilingual test: query `من هو رئيس سوريا الحالي؟`; result title `Syria president official`; snippet `معلومات عن رئيس سوريا الحالي / current Syria president`. Assert the result is accepted because the Arabic entity token `سوريا` is present in the evidence. Add a second English-only result with no Arabic overlap and assert it is rejected in phase 1; cross-language transliteration is explicitly outside this first quality-gate implementation.

- [ ] **Step 8: Commit**

```bash
git add app/build.gradle.kts app/src/main/java/com/musab/aragpt2/SearchResult.java app/src/main/java/com/musab/aragpt2/SearchQualityGate.java app/src/test/java/com/musab/aragpt2/SearchQualityGateTest.java
git commit -m "test: add deterministic web search quality gate"
```

### Task 2: Refactor web retrieval to expose accepted evidence only

**Files:**
- Modify: `app/src/main/java/com/musab/aragpt2/WebSearchClient.java`
- Create: `app/src/test/java/com/musab/aragpt2/WebEvidenceFormattingTest.java`

**Interfaces:**
- Consumes: `SearchQualityGate.filter(...)`
- Produces: `WebSearchClient.WebPayload` whose `sourceCount`, `context`, and `sources` include accepted results only.

- [ ] **Step 1: Write failing formatting tests**

Test a package-visible static helper:

```java
@Test
public void payloadUsesOnlyAcceptedResults() {
    List<SearchResult> accepted = Arrays.asList(
        new SearchResult("Official result", "https://official.example/a", "relevant text")
    );
    WebSearchClient.WebPayload p = WebSearchClient.payloadFromAccepted(accepted, false);

    assertEquals(1, p.sourceCount);
    assertTrue(p.sources.contains("official.example"));
}
```

- [ ] **Step 2: Run and confirm failure**

```bash
gradle :app:testDebugUnitTest --tests 'com.musab.aragpt2.WebEvidenceFormattingTest' --stacktrace
```

Expected: FAIL because `payloadFromAccepted` does not exist.

- [ ] **Step 3: Replace private `Result` with `SearchResult`**

Update Bing RSS and DuckDuckGo parsers so they return `List<SearchResult>`.

Request up to `Math.max(maxResults * 3, maxResults)` raw results before filtering so low-quality results do not consume all slots.

- [ ] **Step 4: Apply the quality gate before building payload**

Implement:

```java
SearchQualityGate.Result quality = SearchQualityGate.filter(q, results, maxResults);
if (quality.accepted.isEmpty()) {
    return new WebPayload("", "", 0, false);
}
return payloadFromAccepted(quality.accepted, false);
```

For direct URLs, reject denied/malformed hosts before fetch. Direct URL mode does not require query-token relevance because the user explicitly supplied the URL.

- [ ] **Step 5: Make payload formatting testable**

Add package-visible:

```java
static WebPayload payloadFromAccepted(List<SearchResult> results, boolean direct)
```

Include title, URL, snippet, and accepted-result index. Count only accepted entries.

- [ ] **Step 6: Run unit tests**

```bash
gradle :app:testDebugUnitTest --stacktrace
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add app/src/main/java/com/musab/aragpt2/WebSearchClient.java app/src/test/java/com/musab/aragpt2/WebEvidenceFormattingTest.java
git commit -m "fix: filter web evidence before H33 uses it"
```

### Task 3: Give every conversation turn a stable ID and separate generation from persistence

**Files:**
- Create: `app/src/main/java/com/musab/aragpt2/ConversationHistory.java`
- Create: `app/src/test/java/com/musab/aragpt2/ConversationHistoryTest.java`
- Modify: `app/src/main/java/com/musab/aragpt2/QwenEngine.java`

**Interfaces:**
- Produces: `ConversationHistory.Turn`, `appendTurn(String turnId, String question, String answer)`, `replaceAnswer(String turnId, String answer)`.
- Produces from Qwen: `generateCandidate(...)`, `commitCanonicalTurn(...)`, `replaceCanonicalAnswer(...)`.

- [ ] **Step 1: Write failing history tests**

Include:

```java
@Test
public void replacementTargetsTurnIdNotQuestionText() {
    ConversationHistory h = new ConversationHistory(16);
    h.appendTurn("t1", "نفس السؤال", "جواب 1");
    h.appendTurn("t2", "نفس السؤال", "جواب 2");

    assertTrue(h.replaceAnswer("t1", "جواب مختار"));
    assertEquals("جواب مختار", h.snapshot().get(1).content);
    assertEquals("جواب 2", h.snapshot().get(3).content);
}
```

Also test trimming preserves user/assistant pairs.

- [ ] **Step 2: Run and confirm failure**

```bash
gradle :app:testDebugUnitTest --tests 'com.musab.aragpt2.ConversationHistoryTest' --stacktrace
```

- [ ] **Step 3: Implement plain-Java `ConversationHistory`**

Each stored message has:

```java
public final String turnId;
public final String role;
public final String content;
```

`appendTurn` appends exactly one user and one assistant record with the same `turnId`. `replaceAnswer` searches assistant records by `turnId` from newest to oldest.

- [ ] **Step 4: Refactor Qwen generation**

Refactor current `generateStream` internals into:

```java
public String generateCandidate(
        String question,
        int maxNewTokens,
        String evidenceContext,
        boolean evidenceOnly,
        StreamListener listener
) throws Exception
```

This method must not mutate conversation history.

For `evidenceOnly=true`, prepend this instruction before the accepted evidence:

```
EVIDENCE_ONLY: أجب فقط مما تدعمه النتائج التالية. إذا لم تكفِ النتائج، قل بوضوح إن الأدلة غير كافية. لا تستخدم معرفتك الداخلية لتعويض نقص الأدلة.
```

Use deterministic generation when evidence exists: `do_sample=false`.

- [ ] **Step 5: Add explicit persistence methods**

Add:

```java
public void commitCanonicalTurn(String turnId, String question, String answer) throws Exception
public boolean replaceCanonicalAnswer(String turnId, String answer) throws Exception
```

Persist `turn_id`, `role`, and `content` in `current_chat.jsonl`. When loading old lines without `turn_id`, accept them with an empty ID so existing installs still open.

- [ ] **Step 6: Update exact-memory shortcut**

The exact-answer shortcut may return a candidate immediately, but it must no longer append to conversation inside generation. Persistence happens only through `commitCanonicalTurn`.

- [ ] **Step 7: Run unit tests and Qwen smoke test**

```bash
gradle :app:testDebugUnitTest --stacktrace
python scripts/smoke_qwen_genai.py
```

Expected: unit tests PASS and both `QWEN_SMOKE_OK` and `SESSION_MEMORY_SMOKE_OK` print.

- [ ] **Step 8: Commit**

```bash
git add app/src/main/java/com/musab/aragpt2/ConversationHistory.java app/src/test/java/com/musab/aragpt2/ConversationHistoryTest.java app/src/main/java/com/musab/aragpt2/QwenEngine.java
git commit -m "refactor: separate Qwen candidates from chat persistence"
```

### Task 4: Add candidate models, orchestration, and hosted-provider placeholder

**Files:**
- Create: `app/src/main/java/com/musab/aragpt2/AnswerCandidate.java`
- Create: `app/src/main/java/com/musab/aragpt2/CandidateSet.java`
- Create: `app/src/main/java/com/musab/aragpt2/HostedAnswerProvider.java`
- Create: `app/src/main/java/com/musab/aragpt2/UnavailableHostedAnswerProvider.java`
- Create: `app/src/main/java/com/musab/aragpt2/CandidateCoordinator.java`
- Create: `app/src/test/java/com/musab/aragpt2/CandidateModelTest.java`

**Interfaces:**
- Produces: immutable candidate types consumed by UI and preference storage.
- Consumes: `QwenEngine`, `WebSearchClient`, `HostedAnswerProvider`.

- [ ] **Step 1: Write failing model tests**

Pin these invariants:

```java
@Test
public void unavailableCandidateCannotPretendToHaveAnswer() {
    AnswerCandidate c = AnswerCandidate.unavailable(
        AnswerCandidate.Kind.HOSTED, "غير متاح");
    assertFalse(c.available);
    assertEquals("", c.answer);
}

@Test
public void candidateSetHasStableTurnId() {
    CandidateSet set = new CandidateSet("turn-123", "سؤال", ...);
    assertEquals("turn-123", set.turnId);
}
```

- [ ] **Step 2: Implement candidate value objects**

`AnswerCandidate` fields:

```java
enum Kind { LOCAL, WEB, HOSTED }
String id;
Kind kind;
String provider;
String answer;
boolean available;
List<SearchResult> sources;
String status;
```

`CandidateSet` contains `turnId`, `question`, and exactly three candidate slots.

- [ ] **Step 3: Define hosted provider abstraction**

```java
public interface HostedAnswerProvider {
    AnswerCandidate answer(String question) throws Exception;
}
```

`UnavailableHostedAnswerProvider.answer` returns an unavailable HOSTED candidate; it never throws merely because no provider is configured.

- [ ] **Step 4: Implement synchronous coordinator**

`CandidateCoordinator.create(String question, Listener listener)`:

1. Create `turnId = UUID.randomUUID().toString()`.
2. Generate local candidate with no web context and no persistence.
3. Resolve/filter web evidence.
4. If evidence usable, generate evidence-only web candidate; otherwise mark WEB unavailable.
5. Ask hosted provider; catch provider-specific failure and mark HOSTED unavailable.
6. Commit LOCAL as the provisional canonical turn only after candidate generation is finished.
7. Return the complete `CandidateSet`.

Do not run two Qwen generations concurrently against the same `QwenEngine`.

- [ ] **Step 5: Run tests**

```bash
gradle :app:testDebugUnitTest --stacktrace
```

- [ ] **Step 6: Commit**

```bash
git add app/src/main/java/com/musab/aragpt2/AnswerCandidate.java app/src/main/java/com/musab/aragpt2/CandidateSet.java app/src/main/java/com/musab/aragpt2/HostedAnswerProvider.java app/src/main/java/com/musab/aragpt2/UnavailableHostedAnswerProvider.java app/src/main/java/com/musab/aragpt2/CandidateCoordinator.java app/src/test/java/com/musab/aragpt2/CandidateModelTest.java
git commit -m "feat: add three-candidate answer orchestration"
```

### Task 5: Persist append-only preference decisions safely

**Files:**
- Create: `app/src/main/java/com/musab/aragpt2/PreferenceRecord.java`
- Create: `app/src/main/java/com/musab/aragpt2/PreferenceStore.java`
- Create: `app/src/test/java/com/musab/aragpt2/PreferenceRecordTest.java`
- Modify: `app/build.gradle.kts` to add `testImplementation("org.json:json:20250517")` so JSON serialization tests run on the local JVM.

**Interfaces:**
- Produces: `recordSelection(CandidateSet, String candidateId)`, `recordCorrection(CandidateSet, String correction)`, `latestDecision(String turnId)`.
- Consumes: `CandidateSet`.

- [ ] **Step 1: Write failing serialization tests**

Test that a record contains `schema_version=1`, `turn_id`, question, all three candidates, provider names, source URLs, decision type, and correction.

Also test latest-event selection:

```java
store.append(selectionFor("t1", "web"));
store.append(correctionFor("t1", "الجواب الصحيح"));
assertEquals("user_correction", store.latestDecision("t1").selectionType);
```

- [ ] **Step 2: Test crash-tolerant reading**

Write a valid JSONL line followed by a truncated line such as `{"schema_version":1`. Assert loading returns the complete earlier event and ignores the damaged final line.

- [ ] **Step 3: Implement `PreferenceRecord`**

Use explicit fields from the spec, plus:

```java
String turnId;
long timestampMs;
String selectionType; // candidate | user_correction
String candidateId;   // nullable
String correction;    // empty when not correction
```

Serialize one event per line.

- [ ] **Step 4: Implement `PreferenceStore`**

Production constructor: `PreferenceStore(Context context)` stores `h33_preferences_v1.jsonl` under `context.getFilesDir()`. Add package-private `PreferenceStore(File file)` for deterministic JVM tests using a temporary directory.

Each append must write one complete line, flush, and `getFD().sync()`. Reads parse line-by-line and skip only malformed lines instead of discarding the whole file.

- [ ] **Step 5: Run tests**

```bash
gradle :app:testDebugUnitTest --stacktrace
```

- [ ] **Step 6: Commit**

```bash
git add app/src/main/java/com/musab/aragpt2/PreferenceRecord.java app/src/main/java/com/musab/aragpt2/PreferenceStore.java app/src/test/java/com/musab/aragpt2/PreferenceRecordTest.java app/build.gradle.kts
git commit -m "feat: persist H33 preference decisions"
```

### Task 6: Render three answer cards and make selection canonical

**Files:**
- Modify: `app/src/main/java/com/musab/aragpt2/MainActivity.java`
- Modify: `app/src/main/res/layout/activity_main.xml` only if a reusable container ID is needed.
- Create: `app/src/main/java/com/musab/aragpt2/CandidateSelectionState.java`
- Create: `app/src/test/java/com/musab/aragpt2/CandidateSelectionStateTest.java`

**Interfaces:**
- Consumes: `CandidateCoordinator`, `PreferenceStore`, `QwenEngine.replaceCanonicalAnswer`.
- Produces: visible three-card comparison UX.

- [ ] **Step 1: Add failing selection-state tests**

```java
@Test
public void duplicateSelectionIsRejected() {
    CandidateSelectionState s = new CandidateSelectionState("t1");
    assertTrue(s.select("web"));
    assertFalse(s.select("local"));
}

@Test
public void correctionSupersedesSelectionInState() {
    CandidateSelectionState s = new CandidateSelectionState("t1");
    assertTrue(s.select("web"));
    s.correct("الجواب الصحيح");
    assertEquals("الجواب الصحيح", s.correction());
}
```

- [ ] **Step 2: Replace single assistant bubble flow**

In `sendMessage()`, call `CandidateCoordinator.create(question, ...)` on the existing background executor.

Render one comparison block containing labeled cards in this order:

```
H33 المحلي
جواب البحث
جواب نموذج المتصفح
```

Unavailable candidates show their status only; never show a fabricated answer.

- [ ] **Step 3: Preserve streaming for the local candidate**

Use coordinator callback events so `H33 المحلي` can stream into its own TextView while later candidate work continues sequentially.

- [ ] **Step 4: Implement choose buttons**

Each available candidate gets `اختر هذا الجواب`.

On click:

```java
preferenceStore.recordSelection(candidateSet, candidate.id);
engine.replaceCanonicalAnswer(candidateSet.turnId, candidate.answer);
```

Then visually mark the selected card and disable all choose buttons in that comparison block.

If persistence or replacement fails, do not mark the choice successful.

- [ ] **Step 5: Implement manual correction**

Add one block-level button: `كلهم خطأ — سأكتب التصحيح`.

On confirm:

```java
preferenceStore.recordCorrection(candidateSet, better);
engine.replaceCanonicalAnswer(candidateSet.turnId, better);
engine.rememberCorrect(candidateSet.question, better);
```

Update the visible canonical marker to `تصحيحك`.

- [ ] **Step 6: Keep existing per-answer utilities**

Keep `نسخ` and `TXT` on available cards. Remove the old ambiguous `✓ صحيح` action from individual assistant bubbles because the comparison selection buttons now represent the training preference explicitly.

- [ ] **Step 7: Run unit tests and build**

```bash
gradle :app:testDebugUnitTest :app:assembleDebug --stacktrace
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add app/src/main/java/com/musab/aragpt2/MainActivity.java app/src/main/res/layout/activity_main.xml app/src/main/java/com/musab/aragpt2/CandidateSelectionState.java app/src/test/java/com/musab/aragpt2/CandidateSelectionStateTest.java
git commit -m "feat: add H33 answer comparison and user selection"
```

### Task 7: Export SFT and preference datasets

**Files:**
- Create: `app/src/main/java/com/musab/aragpt2/PreferenceDatasetExporter.java`
- Create: `app/src/test/java/com/musab/aragpt2/PreferenceDatasetExporterTest.java`
- Modify: `app/src/main/java/com/musab/aragpt2/MainActivity.java`

**Interfaces:**
- Consumes: latest effective preference decision per `turnId`.
- Produces: SFT JSONL and DPO/preference JSONL text.

- [ ] **Step 1: Write failing export tests**

For a selected WEB answer, assert SFT emits:

```json
{"prompt":"...","response":"selected web answer"}
```

and DPO emits one record per non-selected available candidate:

```json
{"prompt":"...","chosen":"selected web answer","rejected":"local answer"}
```

For a manual correction, assert the correction is `chosen` and all available machine answers are rejected candidates.

- [ ] **Step 2: Implement exporter**

Expose:

```java
public String toSftJsonl(List<PreferenceRecord> latest)
public String toPreferenceJsonl(List<PreferenceRecord> latest)
```

Skip records with no effective selected/corrected answer.

- [ ] **Step 3: Add export action to composer menu**

Add `تصدير بيانات التعلم`. Save:

```
H33_SFT_<timestamp>.jsonl
H33_DPO_<timestamp>.jsonl
```

through `TextFileTool.save`.

- [ ] **Step 4: Run tests and build**

```bash
gradle :app:testDebugUnitTest :app:assembleDebug --stacktrace
```

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/musab/aragpt2/PreferenceDatasetExporter.java app/src/test/java/com/musab/aragpt2/PreferenceDatasetExporterTest.java app/src/main/java/com/musab/aragpt2/MainActivity.java
git commit -m "feat: export H33 SFT and preference datasets"
```

### Task 8: Strengthen CI and run device-level verification

**Files:**
- Modify: `.github/workflows/build-apk.yml`
- No new script is required: keep the connectivity smoke inline in `.github/workflows/build-apk.yml`; deterministic quality belongs to Java unit tests.

**Interfaces:**
- Consumes: all tests from Tasks 1–7.
- Produces: CI evidence that filtering, session memory, build, and APK audit pass.

- [ ] **Step 1: Run JVM tests before model conversion**

Add a CI step after Java/Gradle setup and before APK build:

```yaml
- name: Run deterministic Android JVM tests
  run: gradle :app:testDebugUnitTest --stacktrace
```

If test compilation requires the generated ORT GenAI AAR, keep this step after the verified AAR download.

- [ ] **Step 2: Replace the weak web smoke assertion**

Keep the network smoke test, but make its purpose explicit: connectivity only. It should assert Bing RSS is reachable and separately print `WEB_CONNECTIVITY_SMOKE_OK`. It must not be treated as relevance validation; relevance is owned by deterministic Java tests.

- [ ] **Step 3: Run the full CI-equivalent commands locally where possible**

```bash
gradle :app:testDebugUnitTest :app:assembleDebug --stacktrace
python scripts/smoke_qwen_genai.py
```

Expected: PASS and memory smoke success.

- [ ] **Step 4: Push and wait for GitHub Actions**

Verify the workflow has green conclusions for unit tests, Qwen smoke, connectivity smoke, assembleDebug, and APK audit.

- [ ] **Step 5: Run Android emulator QA using `Test Android Apps`**

Exercise this exact flow:

```
1. Launch H33.
2. Ask a normal question.
3. Verify three labeled candidate slots.
4. Ask "من هو رئيس سوريا الحالي؟" and verify no XNXX-like source appears.
5. Select WEB; ask a follow-up referring to the selected answer; verify session context uses WEB answer.
6. Start another turn; then attempt an old-card selection and verify the correct turn ID is changed or the stale card is disabled.
7. Use "كلهم خطأ" and enter a correction.
8. Restart app and verify conversation + preference count persist.
9. Test copy, TXT, export datasets, new chat, and keyboard resize.
10. Capture screenshot and logcat.
```

- [ ] **Step 6: Run Android performance evidence**

Capture at minimum `dumpsys meminfo` after one three-candidate turn and `gfxinfo`/frame evidence while cards render. Compare against the pre-change build if available. Do not claim no regression from a single subjective run.

- [ ] **Step 7: Commit CI changes**

```bash
git add .github/workflows/build-apk.yml
git commit -m "ci: verify H33 candidate and web quality behavior"
```

- [ ] **Step 8: Open PR and enter review gates**

Run CodeRabbit review, SonarQube when integration is available, then open/refresh the PR and run GH Review Loop until clean, capped, or a human decision is required.

