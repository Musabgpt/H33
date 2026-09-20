# H33 partial original-Qwen training

This directory is for controlled training experiments on selected **original**
`Qwen/Qwen2.5-0.5B-Instruct` parameters. It is intentionally separate from
the Android INT4 inference package.

## Kaggle first-run rule

Do **not** upgrade or replace Torch, Transformers, PyArrow, Pandas, or other
core packages before recording the environment Kaggle already provides.

Run first:

```bash
python training/check_environment.py | tee /kaggle/working/h33-environment.json
pip freeze > /kaggle/working/h33-kaggle-freeze.txt
```

Keep both files with the experiment logs. Package changes should only be made
after comparing this snapshot with the training script requirements.

## Prepare preference data

Export learning data from H33 on Android. When a turn was revised multiple
times, pass SFT exports in chronological order; later valid rows win.

```bash
python training/prepare_preferences.py \
  --sft /kaggle/input/h33/H33_SFT_old.jsonl \
  --sft /kaggle/input/h33/H33_SFT_latest.jsonl \
  --preferences /kaggle/input/h33/H33_DPO_latest.jsonl \
  --output-dir /kaggle/working/h33-prepared
```

Outputs:

- `train_sft.jsonl`: canonical prompt/response pairs.
- `train_preferences.jsonl`: canonical chosen answer against unique rejected
  machine answers.
- `eval.jsonl`: held-out turns kept completely out of training.
- `prepare_report.json`: counts, rejected malformed rows, IDs and seed.

The split uses seed `3407`. A dataset with fewer than two effective turns is
rejected instead of pretending to provide a meaningful train/eval split.

## Weight semantics

Storing a preference or correction does **not** update Qwen weights. The later
trainer will start from the original FP16 model, freeze everything, then
unfreeze only a controlled subset of original Qwen parameters. Adapter-only
training must be labeled separately if it is ever added.


## Train selected original Qwen weights

The first trainer opens only the final two original decoder blocks plus the
final norm. The tied embedding / LM-head tensor remains frozen.

Example first run:

```bash
python training/train_partial_sft.py \
  --train-sft /kaggle/working/h33-prepared/train_sft.jsonl \
  --output-dir /kaggle/working/h33-partial \
  --epochs 3 \
  --gradient-accumulation-steps 16 \
  --checkpoint-every-steps 50
```

For a short controlled smoke run, add for example:

```bash
--max-optimizer-steps 2
```

Checkpoints are published only after a completed `optimizer.step()`. Each
checkpoint contains model/tokenizer state, optimizer, scheduler, Python/Torch/
CUDA RNG state, an exact data cursor, and a training fingerprint.

## Resume exactly from a checkpoint

Use the same dataset and training-defining settings:

```bash
python training/train_partial_sft.py \
  --train-sft /kaggle/working/h33-prepared/train_sft.jsonl \
  --output-dir /kaggle/working/h33-partial \
  --resume-from-checkpoint /kaggle/working/h33-partial/checkpoint-00000050 \
  --epochs 3 \
  --gradient-accumulation-steps 16 \
  --checkpoint-every-steps 50
```

The resume guard hashes the training JSONL and verifies the base model, seed,
opened layer count, sequence length, learning rate, weight decay, gradient
clip and accumulation settings. A mismatch fails before further training.

`--max-optimizer-steps` and `--checkpoint-every-steps` are operational
limits and are intentionally not part of the training fingerprint, so a later
session can extend a previously shorter run without changing the mathematical
training configuration.

## Evaluate before mobile export

The evaluator compares the untouched baseline and trained checkpoint on:

- 20 stable deterministic general-capability cases;
- held-out H33 chosen-vs-rejected preference pairs using mean assistant-token
  log probability.

No current political office-holder answer is frozen into the static suite.

```bash
python training/evaluate_partial.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --trained-model /kaggle/working/h33-partial/final \
  --eval /kaggle/working/h33-prepared/eval.jsonl \
  --output /kaggle/working/h33-partial/evaluation_report.json
```

The gate fails if general pass rate drops by more than 5 percentage points or,
when held-out preference pairs exist, if the trained model's mean preference
margin is worse than baseline.

## Export INT4 only after a passing gate

Do not quantize the mobile model directly during training. After evaluation
passes:

```bash
python training/export_mobile_int4.py \
  --trained-model /kaggle/working/h33-partial/final \
  --evaluation-report /kaggle/working/h33-partial/evaluation_report.json \
  --output-dir /kaggle/working/h33-partial-int4
```

The exporter refuses a failed or internally inconsistent evaluation report,
runs the ORT GenAI INT4 builder against the **trained local checkpoint**, and
then runs the H33 Qwen smoke test against that exact exported directory via
`H33_MODEL_DIR`.

Only after those checks pass should the generated INT4 directory be considered
for integration into the Android build.

## What is and is not learning

- `PreferenceStore` and `CorrectionMemory` are local persistence, not weight
  updates.
- `train_partial_sft.py` modifies selected original Qwen parameters.
- INT4 conversion happens after training and evaluation.
- LoRA/adapter experiments, if added later, must remain explicitly labeled as
  adapter training and must not be described as original-weight updates.
