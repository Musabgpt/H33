# H33 Google AI Overview Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the third H33 candidate the exact AI Overview rendered by the ordinary Google Search results page, including its source links, with no Gemini API key and no H33 rewriting.

**Architecture:** Keep the existing `HostedAnswerProvider` interface only as a neutral third-candidate boundary. Implement `GoogleAiOverviewProvider` with an Android `WebView` created on the main thread. Load a normal Google Search URL, poll the rendered DOM using `evaluateJavascript()`, extract the smallest plausible container containing the AI Overview label and answer text, and return its exact text plus unique HTTP(S) links. The provider runs from H33's worker thread and waits on a bounded result; it destroys the WebView after success/failure.

**Tech Stack:** Android Java, Android WebView, `evaluateJavascript()`, org.json, JUnit 4.

**Spec:** `docs/superpowers/specs/2026-09-20-h33-three-candidate-preference-learning-design.md`

## Global Constraints

- No Gemini/Copilot API key is required or stored.
- Do not use `addJavascriptInterface` on Google content.
- Do not bypass consent, CAPTCHA, authentication, or anti-bot/access-control pages.
- Do not import/share Chrome account cookies or passwords.
- Show Google AI Overview text exactly as extracted; do not run it through Qwen.
- If no overview appears within the timeout, the third candidate is honestly unavailable.
- LOCAL and WEB candidates must still work if Google extraction fails.
- DOM extraction must be bounded and memory-clean; destroy each transient WebView.

## Review Focus

- Google page has no AI Overview: HOSTED/third candidate unavailable, not fabricated.
- AI Overview source URLs repeat: deduplicate by normalized URL.
- evaluateJavascript returns a quoted JSON string: decode exactly once.
- Google changes CSS classes: extractor relies on visible labels/structure, not one generated class.
- WebView timeout or renderer error: clean up WebView and return unavailable.
- Query contains Arabic/Unicode: URL encoding preserves it.
- External links inside the overview are collected but never automatically navigated to.

---

### Task 1: Parser and DOM extraction script

**Files:**
- Create: `app/src/main/java/com/musab/aragpt2/GoogleAiOverviewParser.java`
- Create: `app/src/main/java/com/musab/aragpt2/GoogleAiOverviewDomScript.java`
- Test: `app/src/test/java/com/musab/aragpt2/GoogleAiOverviewParserTest.java`

- [ ] RED tests cover exact text, quoted evaluateJavascript return, missing overview, duplicate links, Arabic/English markers.
- [ ] Implement parser using `org.json`; never rewrite answer text beyond trimming outer whitespace.
- [ ] Implement a self-contained JavaScript IIFE that returns `JSON.stringify({found,text,sources})`.
- [ ] Search for the smallest `div/section/article` whose visible text contains an AI Overview marker and enough answer text.
- [ ] Clone the candidate container, remove script/style/button/svg controls, remove only the overview label and obvious UI-only trailing labels such as `عرض المزيد` / `Show more`.
- [ ] Collect unique HTTP(S) anchors from the original candidate container.

### Task 2: WebView provider

**Files:**
- Create: `app/src/main/java/com/musab/aragpt2/GoogleAiOverviewProvider.java`
- Create: `app/src/test/java/com/musab/aragpt2/GoogleAiOverviewUrlTest.java`

- [ ] Build `https://www.google.com/search?q=<encoded>&hl=<lang>&gl=<country>` deterministically.
- [ ] Require `answer()` to be called off the main thread.
- [ ] On the main thread create a transient WebView with JavaScript + DOM storage enabled.
- [ ] Do not install a JavaScript bridge.
- [ ] After page finish, poll `evaluateJavascript` at bounded intervals for at most ~8 seconds.
- [ ] On found overview, map it to `AnswerCandidate.available("hosted", HOSTED, "google-ai-overview", exactText, sources, "Google AI Overview")`.
- [ ] On timeout/blocked/renderer error, return `AnswerCandidate.unavailable(..., "Google AI Overview غير متاح لهذا البحث")`.
- [ ] Stop loading, load blank page, remove callbacks, and destroy the WebView in every terminal path.

### Task 3: Wire the third card

**Files:**
- Modify: `app/src/main/java/com/musab/aragpt2/MainActivity.java`

- [ ] Replace `UnavailableHostedAnswerProvider` with `GoogleAiOverviewProvider(this)`.
- [ ] Rename third card label to `Google AI Overview`.
- [ ] Preserve exact extracted formatting/newlines.
- [ ] Keep copy/TXT/select actions and preference persistence unchanged.
- [ ] A Google failure must leave LOCAL and WEB cards usable.

### Task 4: Regression and device QA

- [ ] Run all deterministic JVM tests before model conversion.
- [ ] Build Qwen INT4, smoke-test Arabic/session memory, and build APK.
- [ ] On a real Android device, test queries known to show AI Overview in the user's Google Search.
- [ ] Compare H33's third-card text against the visible Google AI Overview for the same query.
- [ ] Test a query where Google shows no overview and confirm honest unavailable state.
- [ ] Test Arabic and English queries.
- [ ] Capture logcat and memory impact; verify transient WebView is destroyed.

### Task 5: Review gates

- [ ] CodeRabbit review.
- [ ] SonarQube where integration is reachable.
- [ ] PR against `qwen2.5-mobile`.
- [ ] GH Review Loop until clean/capped/human-decision state.
