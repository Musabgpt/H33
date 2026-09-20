# H33 Partial Original-Qwen Weight Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a controlled subset of the original Qwen2.5-0.5B-Instruct weights from H33's selected/corrected preference data, evaluate regression risk, and only then export a new INT4 mobile model.

**Architecture:** Work only on `qwen-partial-train`, first syncing it to the validated mobile branch. Training starts from the original Hugging Face FP16 model, freezes the entire network, then unfreezes only the final two transformer blocks plus final norm and LM head; these are genuine original Qwen parameters, not an adapter. Use SFT first from the user's canonical answers. Add DPO only after there is enough preference data and baseline SFT is stable. Quantization happens after training/evaluation, never in-place on the INT4 runtime model.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, Qwen/Qwen2.5-0.5B-Instruct, Kaggle GPU, ONNX Runtime GenAI builder for final INT4 export.

**Spec:** `docs/superpowers/specs/2026-09-20-h33-three-candidate-preference-learning-design.md`

## Global Constraints

- Training branch: `qwen-partial-train`; stable mobile work stays on `qwen2.5-mobile`.
- The first training implementation must modify original Qwen weights in selected final layers.
- LoRA/adapter experiments, if added later, must be labeled separately and never presented as base-weight modification.
- Never train directly against the INT4 mobile artifact.
- Use deterministic seeds.
- Save checkpoints only after optimizer steps.
- Do not publish trained weights automatically.
- Verify Kaggle environment compatibility before installing/changing package versions.
- Final Android integration happens only after regression evaluation passes.

## Review Focus

- Export contains duplicate turns/revisions: trainer must use only latest effective decision per turn; pinned in Task 2.
- Manual correction vs selected machine answer: correction must win; pinned in Task 2.
- Trainable-parameter selection accidentally unfreezes the whole model: hard assertion must fail; pinned in Task 3.
- Resume checkpoint after interruption: optimizer step, scheduler, RNG, and dataset offset must restore together; pinned in Task 4.
- Targeted gain accompanied by broad regression: export to INT4 must be blocked by evaluation gate; pinned in Task 5.

---

### Task 1: Synchronize the training branch and snapshot Kaggle compatibility

**Files:**
- Create on `qwen-partial-train`: `training/check_environment.py`
- Create: `training/README.md`

**Interfaces:**
- Produces: machine-readable environment report used before training.

- [ ] **Step 1: Fast-forward/merge validated mobile branch**

Before any training code, update `qwen-partial-train` from the validated `qwen2.5-mobile` head. Resolve conflicts in favor of the validated mobile behavior unless a training-only file is involved.

- [ ] **Step 2: Add environment inspection script**

```python
import json, platform
import torch, transformers

report = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(),
    "gpus": [
        torch.cuda.get_device_name(i)
        for i in range(torch.cuda.device_count())
    ],
}
print(json.dumps(report, indent=2, ensure_ascii=False))
```

If `datasets` is imported by the final trainer, add its version to the report.

- [ ] **Step 3: Run on Kaggle before dependency changes**

Run:

```bash
python training/check_environment.py
pip freeze > /kaggle/working/h33-kaggle-freeze.txt
```

Save the freeze list as a run artifact. Do not install a replacement Torch/Transformers stack until the current environment is checked against the trainer.

- [ ] **Step 4: Commit**

```bash
git add training/check_environment.py training/README.md
git commit -m "chore: capture H33 Kaggle training environment"
```

### Task 2: Validate exported H33 SFT/DPO data and collapse revisions

**Files:**
- Create: `training/prepare_preferences.py`
- Create: `training/tests/test_prepare_preferences.py`

**Interfaces:**
- Consumes: H33 exported JSONL.
- Produces: normalized `train_sft.jsonl`, `train_preferences.jsonl`, `eval.jsonl`.

- [ ] **Step 1: Write failing tests**

Seed records with the same `turn_id`: first a WEB selection, then a user correction. Assert the correction is the only effective target.

Also assert empty answers and malformed JSON lines are rejected with counts, not silently treated as valid examples.

- [ ] **Step 2: Implement normalization**

For each `turn_id`, keep the latest valid event by timestamp/order.

SFT output:

```json
{"id":"turn-id","prompt":"question","response":"effective chosen/corrected answer"}
```

Preference output for each available rejected answer:

```json
{"id":"turn-id","prompt":"question","chosen":"effective answer","rejected":"other candidate answer"}
```

- [ ] **Step 3: Deterministic split**

Use seed `3407`. Sort by `id`, shuffle with `random.Random(3407)`. If there are 20 or more effective turns, set `eval_count = max(2, round(n * 0.10))`; if there are 2–19 turns, set `eval_count = 1`; if there is only 1 turn, stop with `not enough data for train/eval split` instead of training.

- [ ] **Step 4: Run tests**

```bash
python -m unittest training.tests.test_prepare_preferences -v
```

- [ ] **Step 5: Commit**

```bash
git add training/prepare_preferences.py training/tests/test_prepare_preferences.py
git commit -m "feat: prepare H33 preference training data"
```

### Task 3: Implement partial original-weight SFT with a hard trainable-parameter guard

**Files:**
- Create: `training/train_partial_sft.py`
- Create: `training/tests/test_trainable_selection.py`

**Interfaces:**
- Consumes: normalized SFT JSONL.
- Produces: FP16 trained checkpoint under `/kaggle/working/h33-qwen-partial`.

- [ ] **Step 1: Write trainable-selection test**

Extract parameter-selection logic into:

```python
def select_trainable_parameters(model, last_n_blocks=2):
    ...
```

Test with a tiny mock module hierarchy that only:
- final 2 blocks,
- final norm,
- lm_head

are trainable.

- [ ] **Step 2: Load original Qwen in FP16**

```python
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model.config.use_cache = False
model.gradient_checkpointing_enable()
```

- [ ] **Step 3: Freeze all parameters and open only original final layers**

```python
for p in model.parameters():
    p.requires_grad = False

for block in model.model.layers[-2:]:
    for p in block.parameters():
        p.requires_grad = True

for p in model.model.norm.parameters():
    p.requires_grad = True

for p in model.lm_head.parameters():
    p.requires_grad = True
```

Add a startup assertion:

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
ratio = trainable / total
assert 0.0 < ratio < 0.35, f"unsafe trainable ratio: {ratio:.4f}"
```

Print exact trainable parameter names and counts.

- [ ] **Step 4: Format chat SFT examples**

Use the model's chat template:

```python
messages = [
    {"role": "user", "content": row["prompt"]},
    {"role": "assistant", "content": row["response"]},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
```

Labels must mask prompt tokens so loss is computed on assistant response tokens only.

- [ ] **Step 5: Train with conservative defaults**

Initial configuration:

```python
seed = 3407
learning_rate = 2e-5
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
max_length = 512
weight_decay = 0.01
max_grad_norm = 1.0
```

Use AdamW only over `requires_grad=True` parameters.

- [ ] **Step 6: Save after optimizer steps**

Checkpoint directory includes:
- model weights/config/tokenizer
- optimizer
- scheduler
- global optimizer step
- epoch/data position
- Python/Torch/CUDA RNG states

Write checkpoint metadata atomically to `trainer_state.json.tmp`, fsync, then rename.

- [ ] **Step 7: Run smoke training**

On a tiny 8-example dataset, run 2 optimizer steps. Assert:
- loss is finite,
- at least one selected Qwen base tensor changes,
- a frozen early-layer tensor remains byte-identical.

- [ ] **Step 8: Commit**

```bash
git add training/train_partial_sft.py training/tests/test_trainable_selection.py
git commit -m "feat: train selected original Qwen layers"
```

### Task 4: Add exact resume semantics

**Files:**
- Modify: `training/train_partial_sft.py`
- Create: `training/tests/test_resume_state.py`

**Interfaces:**
- Produces: restart from last completed optimizer step, not last prefetched batch.

- [ ] **Step 1: Write failing deterministic resume test**

Run a tiny model for 4 optimizer steps continuously and compare against:
- run 2 steps,
- save,
- reload,
- run 2 more steps.

With the same seed and data order, final selected weights must match within a tight floating-point tolerance.

- [ ] **Step 2: Persist RNG and data cursor**

Save and restore:

```python
random.getstate()
torch.get_rng_state()
torch.cuda.get_rng_state_all()
global_step
epoch
example_index
```

Advance the cursor only after `optimizer.step()` completes.

- [ ] **Step 3: Run test**

```bash
python -m unittest training.tests.test_resume_state -v
```

- [ ] **Step 4: Commit**

```bash
git add training/train_partial_sft.py training/tests/test_resume_state.py
git commit -m "feat: resume partial Qwen training at optimizer boundaries"
```

### Task 5: Add regression evaluation gate

**Files:**
- Create: `training/evaluate_partial.py`
- Create: `training/eval_cases.jsonl`

**Interfaces:**
- Consumes: baseline model + trained model.
- Produces: `evaluation_report.json` and pass/fail exit code.

- [ ] **Step 1: Build fixed evaluation groups**

Include:
- user-selected/corrected held-out prompts,
- Arabic factual/general questions,
- arithmetic/basic reasoning,
- multi-turn/session-reference prompts,
- refusal to invent when evidence is absent.

Do not freeze time-sensitive political answers into the static suite; test instruction-following/provenance behavior instead.

- [ ] **Step 2: Measure targeted preference gain**

For held-out preference cases, compare sequence log-probability of `chosen` vs `rejected`. Report fraction where chosen scores higher.

- [ ] **Step 3: Measure broad regression**

For static deterministic cases, compare exact/normalized expected answer checks where appropriate and record baseline vs trained pass rates.

Block export if broad pass rate drops by more than 5 percentage points from baseline.

- [ ] **Step 4: Emit machine-readable report**

```json
{
  "baseline_general_pass_rate": 0.0,
  "trained_general_pass_rate": 0.0,
  "preference_win_rate": 0.0,
  "passes_gate": false
}
```

Exit non-zero when `passes_gate=false`.

- [ ] **Step 5: Commit**

```bash
git add training/evaluate_partial.py training/eval_cases.jsonl
git commit -m "test: gate partial Qwen training on regressions"
```

### Task 6: Quantize only a passing checkpoint and integrate it into the mobile build

**Files:**
- Create: `training/export_mobile_int4.py`
- Modify: `.github/workflows/build-apk.yml` on the training branch or create a training-only workflow with explicit model-source input.
- Modify: `scripts/smoke_qwen_genai.py` only if model path selection needs to be parameterized.

**Interfaces:**
- Consumes: passing FP16 trained checkpoint.
- Produces: ORT GenAI INT4 model package and APK smoke evidence.

- [ ] **Step 1: Refuse export without a passing evaluation report**

```python
report = json.load(open(args.evaluation_report))
if not report.get("passes_gate"):
    raise SystemExit("evaluation gate failed; refusing INT4 export")
```

- [ ] **Step 2: Build INT4 package**

Invoke ONNX Runtime GenAI builder against the trained checkpoint directory, not the original Hugging Face model:

```bash
python -m onnxruntime_genai.models.builder \
  -m /kaggle/working/h33-qwen-partial/final \
  -o /kaggle/working/h33-qwen-partial-int4 \
  -p int4 \
  -e cpu \
  --extra_options hf_remote=false hf_token=false
```

- [ ] **Step 3: Smoke test the quantized model**

Run arithmetic, Arabic response, and explicit session-context tests before packaging.

- [ ] **Step 4: Make trained model source explicit in CI**

Add a workflow input such as `model_source`. The stable default remains `Qwen/Qwen2.5-0.5B-Instruct`; a trained checkpoint source must be supplied explicitly. Do not silently switch stable users to experimental weights.

- [ ] **Step 5: Build APK and run Android QA**

Build `assembleDebug`, run static APK audit, install on emulator, and repeat:
- session memory,
- three-candidate flow,
- corrections/preferences,
- local answer sanity.

- [ ] **Step 6: Review gates**

Run CodeRabbit, SonarQube when available, PR review, GH Review Loop, and final verification before any merge into the stable mobile branch.

