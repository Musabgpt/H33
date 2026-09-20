#!/usr/bin/env python3
"""Partial SFT for selected original Qwen2.5 weights.

The first experiment intentionally trains only the final decoder blocks and
final RMSNorm. Qwen2.5-0.5B-Instruct ties lm_head.weight to the input embedding
matrix, so the tied head/embedding tensor remains frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SEED = 3407


@dataclass(frozen=True)
class ResumeCursor:
    global_step: int
    epoch: int
    next_order_offset: int

    def __post_init__(self):
        if self.global_step < 0 or self.epoch < 0 or self.next_order_offset < 0:
            raise ValueError("resume cursor values must be non-negative")

    def to_dict(self) -> dict:
        return {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "next_order_offset": int(self.next_order_offset),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "ResumeCursor":
        if not isinstance(value, dict):
            raise ValueError("resume cursor must be an object")
        return cls(
            global_step=int(value.get("global_step", 0)),
            epoch=int(value.get("epoch", 0)),
            next_order_offset=int(value.get("next_order_offset", 0)),
        )


def build_epoch_order(example_count: int, seed: int, epoch: int) -> list[int]:
    if example_count <= 0:
        raise ValueError("example_count must be positive")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    order = list(range(example_count))
    random.Random(int(seed) + int(epoch)).shuffle(order)
    return order


def cursor_after_optimizer_step(
    *,
    global_step: int,
    epoch: int,
    completed_order_offset: int,
    order_length: int,
) -> ResumeCursor:
    if order_length <= 0:
        raise ValueError("order_length must be positive")
    if completed_order_offset < 0 or completed_order_offset >= order_length:
        raise ValueError("completed_order_offset outside order")
    if global_step < 0 or epoch < 0:
        raise ValueError("global_step/epoch must be non-negative")

    next_offset = completed_order_offset + 1
    if next_offset >= order_length:
        return ResumeCursor(
            global_step=global_step,
            epoch=epoch + 1,
            next_order_offset=0,
        )
    return ResumeCursor(
        global_step=global_step,
        epoch=epoch,
        next_order_offset=next_offset,
    )


def _atomic_write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def save_optimizer_checkpoint(
    *,
    output_dir: Path,
    cursor: ResumeCursor,
    model,
    tokenizer,
    optimizer,
    scheduler,
    torch,
) -> Path:
    """Publish a complete checkpoint only after an optimizer boundary."""
    if cursor.global_step <= 0:
        raise ValueError("checkpoint requires a completed optimizer step")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    name = f"checkpoint-{cursor.global_step:08d}"
    final_dir = output / name
    tmp_dir = output / f".{name}.tmp"

    if final_dir.exists():
        raise FileExistsError(f"checkpoint already exists: {final_dir}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        model_dir = tmp_dir / "model"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")

        rng_state = {
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else []
            ),
        }
        torch.save(rng_state, tmp_dir / "rng_state.pt")

        manifest = {
            "schema_version": 1,
            "cursor": cursor.to_dict(),
        }
        _atomic_write_json(tmp_dir / "trainer_state.json", manifest)

        os.replace(tmp_dir, final_dir)

        # Persist the directory rename when the platform supports fsync on dirs.
        try:
            fd = os.open(str(output), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

        return final_dir
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def read_checkpoint_cursor(checkpoint_dir: Path) -> ResumeCursor:
    path = Path(checkpoint_dir) / "trainer_state.json"
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("unsupported checkpoint schema")
    return ResumeCursor.from_dict(manifest.get("cursor"))


def restore_optimizer_checkpoint(
    *,
    checkpoint_dir: Path,
    optimizer,
    scheduler,
    torch,
    map_location="cpu",
) -> ResumeCursor:
    """Restore optimizer/scheduler/RNG state from a completed checkpoint."""
    checkpoint = Path(checkpoint_dir)
    cursor = read_checkpoint_cursor(checkpoint)

    optimizer_path = checkpoint / "optimizer.pt"
    scheduler_path = checkpoint / "scheduler.pt"
    rng_path = checkpoint / "rng_state.pt"

    for path in (optimizer_path, scheduler_path, rng_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint file: {path}")

    optimizer_state = torch.load(
        optimizer_path,
        map_location=map_location,
        weights_only=False,
    )
    scheduler_state = torch.load(
        scheduler_path,
        map_location=map_location,
        weights_only=False,
    )
    rng_state = torch.load(
        rng_path,
        map_location="cpu",
        weights_only=False,
    )

    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)

    python_state = rng_state.get("python_random_state")
    torch_state = rng_state.get("torch_rng_state")
    cuda_states = rng_state.get("cuda_rng_state_all", [])

    if python_state is None or torch_state is None:
        raise ValueError("checkpoint RNG state is incomplete")

    random.setstate(python_state)
    torch.set_rng_state(torch_state)

    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)

    return cursor


def _unique_parameters(model) -> list:
    seen = set()
    out = []
    for param in model.parameters():
        key = id(param)
        if key in seen:
            continue
        seen.add(key)
        out.append(param)
    return out


def _set_requires_grad(module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad = value


def select_trainable_parameters(
    model,
    last_n_blocks: int = 2,
    max_ratio: float = 0.20,
) -> dict:
    """Freeze the model, then open only selected original decoder weights.

    This function is intentionally framework-light so its safety contract can
    be tested without downloading PyTorch or the 494M model in CI.
    """
    if last_n_blocks <= 0:
        raise ValueError("last_n_blocks must be >= 1")
    if not (0.0 < float(max_ratio) < 1.0):
        raise ValueError("max_ratio must be between 0 and 1")

    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    norm = getattr(backbone, "norm", None)
    if layers is None or norm is None:
        raise ValueError("unexpected Qwen model structure: model.layers/norm missing")
    if last_n_blocks > len(layers):
        raise ValueError(
            f"last_n_blocks={last_n_blocks} exceeds layer count={len(layers)}"
        )

    all_params = _unique_parameters(model)
    if not all_params:
        raise ValueError("model has no parameters")

    for param in all_params:
        param.requires_grad = False

    for block in layers[-last_n_blocks:]:
        _set_requires_grad(block, True)
    _set_requires_grad(norm, True)

    # Explicitly keep the shared input/output embedding tensor frozen.
    embed_tokens = getattr(backbone, "embed_tokens", None)
    if embed_tokens is not None:
        _set_requires_grad(embed_tokens, False)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        _set_requires_grad(lm_head, False)

    total = sum(int(p.numel()) for p in all_params)
    trainable_params = [p for p in all_params if bool(p.requires_grad)]
    trainable = sum(int(p.numel()) for p in trainable_params)
    ratio = (trainable / total) if total else 0.0

    if trainable <= 0:
        raise ValueError("no trainable parameters selected")

    if ratio >= max_ratio:
        # Fail closed: never leave an unexpectedly broad trainable set active.
        for param in all_params:
            param.requires_grad = False
        raise ValueError(
            f"unsafe trainable ratio: {ratio:.4f} >= {max_ratio:.4f}"
        )

    names = []
    named_parameters = getattr(model, "named_parameters", None)
    if callable(named_parameters):
        names = [
            name
            for name, param in named_parameters()
            if bool(param.requires_grad)
        ]

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_tensors": len(trainable_params),
        "ratio": ratio,
        "trainable_names": names,
        "last_n_blocks": last_n_blocks,
    }


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("response", "")).strip()
            if not prompt or not response:
                raise ValueError(
                    f"missing prompt/response at {path}:{line_number}"
                )
            rows.append(
                {
                    "id": str(row.get("id", line_number)).strip(),
                    "prompt": prompt,
                    "response": response,
                }
            )
    if not rows:
        raise ValueError(f"no valid SFT rows in {path}")
    return rows


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_masked_sequence(
    prompt_ids,
    full_ids,
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Left-truncate while computing loss only on surviving assistant tokens."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    prompt = list(prompt_ids)
    full = list(full_ids)
    if len(full) <= len(prompt):
        raise ValueError("assistant response is missing from tokenized sequence")

    # Chat-template generation prompt should be the prefix immediately before
    # assistant content. Fail instead of silently training on a mismatched mask.
    prefix_len = min(len(prompt), len(full))
    if full[:prefix_len] != prompt[:prefix_len]:
        raise ValueError("chat template prompt is not a prefix of full sequence")

    removed_from_left = max(0, len(full) - max_length)
    trimmed = full[removed_from_left:]

    surviving_prompt = max(0, len(prompt) - removed_from_left)
    surviving_prompt = min(surviving_prompt, len(trimmed))

    labels = list(trimmed)
    for index in range(surviving_prompt):
        labels[index] = -100

    if not labels or all(label == -100 for label in labels):
        raise ValueError("assistant response vanished after truncation")

    return trimmed, labels


def _build_example(tokenizer, row: dict, max_length: int, torch) -> dict:
    user_messages = [{"role": "user", "content": row["prompt"]}]
    full_messages = [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["response"]},
    ]

    prompt_ids = tokenizer.apply_chat_template(
        user_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    input_ids, labels = build_masked_sequence(
        prompt_ids=prompt_ids,
        full_ids=full_ids,
        max_length=max_length,
    )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _collate_one(example: dict, pad_token_id: int, torch) -> dict:
    input_ids = example["input_ids"].unsqueeze(0)
    labels = example["labels"].unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def run_training(args) -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Run check_environment.py first; "
            "do not blindly replace Kaggle's package stack."
        ) from exc

    _seed_everything(args.seed, torch)

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "CUDA GPU is required by default. Pass --allow-cpu only for a tiny smoke test."
        )

    dtype = torch.bfloat16 if (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ) else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
    )
    model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    report = select_trainable_parameters(
        model,
        last_n_blocks=args.last_n_blocks,
        max_ratio=args.max_trainable_ratio,
    )
    print(json.dumps({"trainable_selection": report}, ensure_ascii=False, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    rows = _load_jsonl(args.train_sft)
    examples = [
        _build_example(tokenizer, row, args.max_length, torch)
        for row in rows
    ]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        random.Random(args.seed + epoch).shuffle(order)

        for example_index in order:
            batch = _collate_one(
                examples[example_index],
                tokenizer.pad_token_id,
                torch,
            )
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch={epoch} example={example_index}"
                )
            loss.backward()
            micro_step += 1

            should_step = (
                micro_step % args.gradient_accumulation_steps == 0
                or example_index == order[-1]
            )
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "example_index": example_index,
                        "loss": float(loss.detach().cpu())
                        * args.gradient_accumulation_steps,
                    },
                    ensure_ascii=False,
                )
            )

            if args.max_optimizer_steps and global_step >= args.max_optimizer_steps:
                break

        if args.max_optimizer_steps and global_step >= args.max_optimizer_steps:
            break

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    summary = {
        "base_model": args.base_model,
        "seed": args.seed,
        "optimizer_steps": global_step,
        "trainable_selection": report,
        "output": str(final_dir),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--last-n-blocks", type=int, default=2)
    parser.add_argument("--max-trainable-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
