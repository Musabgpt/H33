# H33 M1 Current State — 2026-09-20

This file is the continuity checkpoint for the H33 workstream. Read it before making changes.

## Branches

- Stable mobile base: `qwen2.5-mobile`
- Validated three-candidate app implementation: `h33-three-candidate-native`
  - validated head: `e96d1a298dede0342388ce5828ead79c0523910b`
  - PR #1 -> `qwen2.5-mobile`
  - GitHub Actions: JVM tests, Android Lint, Qwen Arabic/session smoke, web connectivity, APK build/audit all green
  - do not merge into stable until physical Android validation is complete
- Long-lived training base: `qwen-partial-train`
  - fast-forwarded to validated app head before training review
- Active partial-training implementation: `h33-partial-train-native`
  - PR #3 -> `qwen-partial-train`
  - keep PR #3 draft until real Kaggle GPU evidence exists

## App behavior implemented

Each question produces three independent slots:

1. H33 local Qwen answer.
2. Web-evidence answer grounded only in accepted relevant search results.
3. Exact rendered Google AI Overview when Google provides it.

The Google candidate uses a transient Android WebView and `evaluateJavascript()`; it does not use Gemini API and does not pass the overview through Qwen.

Search quality includes:
- adult/spam domain regression rejection,
- weak-match rejection,
- duplicate-host suppression,
- provenance/source handling.

Google WebView hardening includes:
- no `addJavascriptInterface`,
- file/content/file-URL universal access disabled,
- mixed content disabled,
- Safe Browsing enabled,
- third-party cookies disabled,
- allowlist restricted to Google Search + consent hosts,
- Google Accounts/support navigation blocked,
- consent/challenge must be completed manually if Google requires it,
- transient WebView destroyed after terminal state.

## Preference / learning semantics

- `CorrectionMemory` and `PreferenceStore` are persistence, not weight training.
- User selection/correction becomes the canonical answer.
- Preference decision persistence is transactional:
  1. snapshot previous canonical answer,
  2. apply selected/corrected answer,
  3. append preference record,
  4. rollback canonical answer if preference append fails.
- Exports:
  - SFT JSONL: canonical user-selected/corrected answer.
  - preference JSONL: chosen answer against available rejected machine answers.
- latest decision per turn supersedes earlier revisions.

## Partial original-Qwen training implemented

The trainer starts from `Qwen/Qwen2.5-0.5B-Instruct` in floating point and modifies selected original model parameters.

First experiment:
- final 2 decoder blocks trainable,
- final RMSNorm trainable,
- tied embedding / LM head frozen,
- hard maximum trainable-ratio guard.

Implemented and unit-tested:
- assistant-only SFT masking,
- deterministic epoch order,
- correct gradient-accumulation groups,
- optimizer-boundary resume cursor,
- atomic checkpoint publication,
- optimizer + scheduler restoration,
- Python/Torch/CUDA RNG restoration,
- dataset/training fingerprint validation,
- Transformers 4.x/5.x dtype API compatibility,
- proof that selected original Qwen weights changed,
- failure if optimizer steps occur without a trainable-weight fingerprint change,
- held-out general + preference evaluation gate,
- INT4 export blocked unless evaluation passes,
- INT4 export blocked unless original-weight-change proof exists.

## Kaggle compatibility

Latest public Kaggle/docker-python release checked during implementation:
- v170 GPU, published 2026-06-29
- Kaggle requirements specify `transformers>=5.0.0`

Runtime truth still comes from the actual notebook environment. Do not replace Torch/Transformers before recording:

```bash
python training/check_environment.py
pip freeze
```

Notebook:
- `training/H33_PARTIAL_QWEN_KAGGLE.ipynb`
- ~9 KB, well below 1 MB
- does not install/replace Torch or Transformers
- preflight -> prepare data -> 2 optimizer-step real-weight smoke -> exact resume to step 3 -> evaluation gate
- INT4 export remains opt-in

## CI evidence

Training workflow:
- `.github/workflows/training-smoke.yml`
- `python -m compileall -q training`
- complete deterministic unit suite
- latest training head before this checkpoint was green

App workflow:
- JVM tests green
- Android Lint green
- Qwen Arabic/session smoke green
- web connectivity smoke green
- debug APK build + static audit green

## External validation still required

These cannot be truthfully claimed complete from GitHub CI alone:

1. **Physical Android validation**
   - install latest APK on a real Android device,
   - verify three-card UI,
   - compare H33 third card to visible Google AI Overview for the same query,
   - test Google consent/challenge flow,
   - verify no-overview state,
   - test Arabic/English queries,
   - inspect memory/crash behavior.

2. **Real Kaggle GPU smoke**
   - run `training/H33_PARTIAL_QWEN_KAGGLE.ipynb`,
   - retain `h33-environment.json` and `h33-kaggle-freeze.txt`,
   - verify step 2 checkpoint,
   - verify resume starts at optimizer step 2 and reaches step 3,
   - verify `original_weights_changed=true`,
   - inspect evaluation report,
   - run INT4 export only if the gate passes.

3. **External review tools**
   - CodeRabbit was requested on PRs but no bot review was returned at last check.
   - SonarQube is not configured in the repository; Android Lint is active. Do not install Sonar tooling without explicit approval if its integration asks for installation.

## Exact next action

Do not add more training features before real execution evidence.

Next:
1. run the Kaggle notebook with exported H33 SFT/DPO data;
2. bring back the environment JSON, freeze list, training summary, and evaluation report;
3. diagnose from those artifacts before increasing training duration or opening more layers.

If Android testing is available first, install the latest successful APK from PR #1's branch and validate Google AI Overview extraction before stable merge.
