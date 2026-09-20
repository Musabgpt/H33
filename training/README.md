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
