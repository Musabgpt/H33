# H33 Hosted Browser-Model Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real third candidate backed by an official hosted model with web grounding, beginning with Gemini Grounding with Google Search, without embedding credentials in source control or pretending an unavailable provider answered.

**Architecture:** Keep the hosted path behind `HostedAnswerProvider`. Use an injectable HTTP transport so response parsing is unit-testable without secrets or network. Store the user's API key encrypted with Android Keystore, expose provider setup from the existing composer menu, and return a structured unavailable state whenever credentials or network access are missing.

**Tech Stack:** Android Java, Android Keystore AES/GCM, `HttpURLConnection`, Gemini Generate Content REST API with `google_search`, org.json, JUnit 4.

**Spec:** `docs/superpowers/specs/2026-09-20-h33-three-candidate-preference-learning-design.md`

## Global Constraints

- Hosted answer must come from the configured official provider; never scrape consumer browser UI as the primary integration.
- Never commit API keys or place them in Gradle source.
- Provider failure must not block LOCAL or WEB candidates.
- Provider provenance must be visible.
- External provider receives only the current question needed for that request; local correction history is not uploaded.
- Default Gemini model for the first implementation: `gemini-3.8-flash`; keep the model name configurable so it can be changed without code edits.

## Review Focus

- Missing key: hosted candidate must be unavailable, not an exception; pinned in Task 1.
- Malformed Gemini JSON or missing candidates: hosted candidate must be unavailable with a clear status; pinned in Task 3.
- Grounded response has sources but repeated URIs: UI/persistence should receive unique sources; pinned in Task 3.
- HTTP 401/403: key must remain stored, but answer call returns auth failure status; pinned in Task 3.
- App process restart: encrypted key can be decrypted and used again; verified in Task 4 emulator QA.

---

### Task 1: Provider configuration contract and unavailable behavior

**Files:**
- Modify: `app/src/main/java/com/musab/aragpt2/HostedAnswerProvider.java`
- Create: `app/src/main/java/com/musab/aragpt2/HostedProviderConfig.java`
- Create: `app/src/test/java/com/musab/aragpt2/HostedProviderConfigTest.java`

**Interfaces:**
- Produces: provider name, model name, configured/unconfigured state.

- [ ] **Step 1: Write failing tests**

```java
@Test
public void blankKeyMeansUnconfigured() {
    HostedProviderConfig c = new HostedProviderConfig("gemini", "gemini-3.8-flash", "");
    assertFalse(c.isConfigured());
}
```

- [ ] **Step 2: Implement immutable config**

Normalize provider/model/key with trim. `isConfigured()` requires provider `gemini`, non-empty model, and non-empty key.

- [ ] **Step 3: Run tests and commit**

```bash
gradle :app:testDebugUnitTest --tests 'com.musab.aragpt2.HostedProviderConfigTest' --stacktrace
git add app/src/main/java/com/musab/aragpt2/HostedAnswerProvider.java app/src/main/java/com/musab/aragpt2/HostedProviderConfig.java app/src/test/java/com/musab/aragpt2/HostedProviderConfigTest.java
git commit -m "feat: define hosted provider configuration"
```

### Task 2: Encrypt provider credentials with Android Keystore

**Files:**
- Create: `app/src/main/java/com/musab/aragpt2/ProviderSecretStore.java`
- Modify: `app/src/main/java/com/musab/aragpt2/MainActivity.java`

**Interfaces:**
- Produces: `saveGeminiKey(String)`, `loadGeminiKey()`, `clearGeminiKey()`.

- [ ] **Step 1: Implement Android Keystore key creation**

Use alias `h33_hosted_provider_key_v1`, AES/GCM/NoPadding, 256-bit key when supported, with encrypt/decrypt purpose.

- [ ] **Step 2: Store only ciphertext + IV in SharedPreferences**

Preference file: `h33_hosted_provider_secrets`.

Keys:

```
gemini_key_ciphertext
gemini_key_iv
gemini_model
```

Do not log plaintext key.

- [ ] **Step 3: Add provider setup dialog**

Composer menu action: `إعداد نموذج المتصفح`.

Dialog fields:
- API key password field
- model field defaulting to `gemini-3.8-flash`
- save
- clear credentials

Status after save: `تم حفظ إعداد نموذج المتصفح على الجهاز`.

- [ ] **Step 4: Build**

```bash
gradle :app:assembleDebug --stacktrace
```

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/musab/aragpt2/ProviderSecretStore.java app/src/main/java/com/musab/aragpt2/MainActivity.java
git commit -m "feat: store hosted provider credentials securely"
```

### Task 3: Implement Gemini Google-Search-grounded provider with injectable transport

**Files:**
- Create: `app/src/main/java/com/musab/aragpt2/HttpJsonTransport.java`
- Create: `app/src/main/java/com/musab/aragpt2/UrlConnectionJsonTransport.java`
- Create: `app/src/main/java/com/musab/aragpt2/GeminiGroundedProvider.java`
- Create: `app/src/test/java/com/musab/aragpt2/GeminiGroundedProviderTest.java`

**Interfaces:**
- Consumes: `HostedProviderConfig`, `HttpJsonTransport`.
- Produces: HOSTED `AnswerCandidate` with provider `gemini-google-search` and unique sources.

- [ ] **Step 1: Write fake-transport tests**

Seed a response with:

```json
{
  "candidates": [{
    "content": {"parts": [{"text": "جواب Gemini"}]},
    "groundingMetadata": {
      "groundingChunks": [
        {"web": {"uri": "https://example.com/a", "title": "A"}},
        {"web": {"uri": "https://example.com/a", "title": "A duplicate"}},
        {"web": {"uri": "https://example.org/b", "title": "B"}}
      ]
    }
  }]
}
```

Assert answer = `جواب Gemini` and sources size = 2.

Add tests for HTTP 401 and empty candidates.

- [ ] **Step 2: Implement request body**

POST to:

```
https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent
```

Headers:

```
x-goog-api-key: <key>
Content-Type: application/json
```

Body:

```json
{
  "contents": [{"parts": [{"text": "<question>"}]}],
  "tools": [{"google_search": {}}]
}
```

- [ ] **Step 3: Parse text and grounding metadata**

Concatenate text parts from the first candidate. Parse `groundingMetadata.groundingChunks[].web.uri/title`. Deduplicate by normalized URI.

Do not return an available candidate if answer text is empty.

- [ ] **Step 4: Run tests**

```bash
gradle :app:testDebugUnitTest --tests 'com.musab.aragpt2.GeminiGroundedProviderTest' --stacktrace
```

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/musab/aragpt2/HttpJsonTransport.java app/src/main/java/com/musab/aragpt2/UrlConnectionJsonTransport.java app/src/main/java/com/musab/aragpt2/GeminiGroundedProvider.java app/src/test/java/com/musab/aragpt2/GeminiGroundedProviderTest.java
git commit -m "feat: add Gemini grounded browser-model candidate"
```

### Task 4: Wire provider into H33 and verify restart/error behavior

**Files:**
- Modify: `app/src/main/java/com/musab/aragpt2/MainActivity.java`
- Modify: `app/src/main/java/com/musab/aragpt2/CandidateCoordinator.java`
- Modify: `.github/workflows/build-apk.yml` only for deterministic unit tests; no CI secret is required.

**Interfaces:**
- Consumes: encrypted settings.
- Produces: actual HOSTED card when configured.

- [ ] **Step 1: Construct provider at startup**

If key exists:

```java
HostedAnswerProvider hosted = new GeminiGroundedProvider(
    config,
    new UrlConnectionJsonTransport()
);
```

Otherwise use `UnavailableHostedAnswerProvider`.

- [ ] **Step 2: Keep failures isolated**

Coordinator catches hosted exceptions and converts them to unavailable HOSTED state without discarding LOCAL or WEB candidates.

- [ ] **Step 3: Run deterministic tests and build**

```bash
gradle :app:testDebugUnitTest :app:assembleDebug --stacktrace
```

- [ ] **Step 4: Emulator QA**

On Android emulator:
1. With no key, verify hosted card says unavailable.
2. Save a valid key and model.
3. Ask a current-information question.
4. Verify hosted card provider label is Gemini/Google Search and includes returned sources.
5. Force airplane/offline mode and verify LOCAL still works while hosted becomes unavailable.
6. Restart app and verify provider remains configured.
7. Clear credentials and verify hosted returns to unavailable.

- [ ] **Step 5: Commit and review**

```bash
git add app/src/main/java/com/musab/aragpt2/MainActivity.java app/src/main/java/com/musab/aragpt2/CandidateCoordinator.java .github/workflows/build-apk.yml
git commit -m "feat: wire grounded hosted candidate into H33"
```

Then run CodeRabbit, SonarQube when available, PR review, and GH Review Loop.

