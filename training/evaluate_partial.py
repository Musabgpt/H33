#!/usr/bin/env python3
"""Evaluate a partially trained Qwen checkpoint before mobile INT4 export.

The gate compares the trained checkpoint against the untouched baseline on:
1) stable, deterministic general-capability cases; and
2) held-out H33 preference pairs using length-normalized assistant log-probability.

No current political office-holder answer is frozen into this evaluation set.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import unicodedata
from pathlib import Path
from statistics import mean

from training.train_partial_sft import BASE_MODEL, build_masked_sequence


_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


def normalize_answer(value) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ARABIC_DIGITS)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.lower()
    text = re.sub(r"[_*~]+", " ", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def case_passes(output: str, case: dict) -> bool:
    check = case.get("check") or {}
    check_type = str(check.get("type", "")).strip()
    values = check.get("values") or []
    normalized_output = normalize_answer(output)
    normalized_values = [
        normalize_answer(value)
        for value in values
        if normalize_answer(value)
    ]

    if not normalized_values:
        raise ValueError(f"evaluation case has no expected values: {case!r}")

    if check_type == "exact_any":
        return normalized_output in normalized_values
    if check_type == "contains_any":
        return any(value in normalized_output for value in normalized_values)

    raise ValueError(f"unsupported evaluation check type: {check_type!r}")


def compute_gate(
    *,
    baseline_general_pass_rate: float,
    trained_general_pass_rate: float,
    baseline_preference_margin: float,
    trained_preference_margin: float,
    preference_pairs: int,
    max_general_drop: float = 0.05,
) -> dict:
    if preference_pairs < 0:
        raise ValueError("preference_pairs must be non-negative")
    if not (0.0 <= max_general_drop <= 1.0):
        raise ValueError("max_general_drop must be between 0 and 1")

    baseline_general = float(baseline_general_pass_rate)
    trained_general = float(trained_general_pass_rate)
    general_drop = baseline_general - trained_general
    general_ok = general_drop <= float(max_general_drop) + 1e-12

    baseline_margin = float(baseline_preference_margin)
    trained_margin = float(trained_preference_margin)
    margin_delta = trained_margin - baseline_margin
    preference_ok = preference_pairs == 0 or margin_delta >= -1e-12

    return {
        "max_general_drop": float(max_general_drop),
        "general_drop": general_drop,
        "preference_margin_delta": margin_delta,
        "general_ok": bool(general_ok),
        "preference_ok": bool(preference_ok),
        "passes_gate": bool(general_ok and preference_ok),
    }


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _load_general_cases(path: Path) -> list[dict]:
    rows = _load_jsonl(path)
    if not rows:
        raise ValueError("general evaluation set is empty")
    for index, row in enumerate(rows, 1):
        if not str(row.get("id", "")).strip():
            raise ValueError(f"general case {index} is missing id")
        if not str(row.get("prompt", "")).strip():
            raise ValueError(f"general case {index} is missing prompt")
        check = row.get("check") or {}
        if check.get("type") not in {"exact_any", "contains_any"}:
            raise ValueError(f"general case {index} has invalid check type")
        if not check.get("values"):
            raise ValueError(f"general case {index} has no expected values")
    return rows


def _load_preference_pairs(path: Path) -> list[dict]:
    rows = _load_jsonl(path)
    pairs = []
    for row in rows:
        prompt = str(row.get("prompt", "")).strip()
        chosen = str(row.get("response", "")).strip()
        rejected = row.get("rejected") or []
        if isinstance(rejected, str):
            rejected = [rejected]
        if not prompt or not chosen:
            continue
        seen = set()
        for answer in rejected:
            value = str(answer).strip()
            if not value or value == chosen or value in seen:
                continue
            seen.add(value)
            pairs.append(
                {
                    "id": str(row.get("id", "")).strip(),
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": value,
                }
            )
    return pairs


def _messages_for_prompt(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def _generate_answer(
    *,
    model,
    tokenizer,
    prompt: str,
    torch,
    device,
    max_new_tokens: int,
) -> str:
    input_ids = tokenizer.apply_chat_template(
        _messages_for_prompt(prompt),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    continuation = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(
        continuation,
        skip_special_tokens=True,
    ).strip()


def _mean_assistant_logprob(
    *,
    model,
    tokenizer,
    prompt: str,
    answer: str,
    torch,
    device,
    max_length: int,
) -> float:
    prompt_ids = tokenizer.apply_chat_template(
        _messages_for_prompt(prompt),
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )
    input_ids, labels = build_masked_sequence(
        prompt_ids=prompt_ids,
        full_ids=full_ids,
        max_length=max_length,
    )

    input_tensor = torch.tensor(
        [input_ids], dtype=torch.long, device=device
    )
    label_tensor = torch.tensor(
        [labels], dtype=torch.long, device=device
    )
    attention_mask = torch.ones_like(input_tensor)

    with torch.inference_mode():
        logits = model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
        ).logits

    shift_logits = logits[:, :-1, :].float()
    shift_labels = label_tensor[:, 1:]
    valid = shift_labels.ne(-100)
    if not bool(valid.any()):
        raise ValueError("no assistant tokens available for preference scoring")

    safe_labels = shift_labels.masked_fill(~valid, 0)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    selected = token_log_probs.masked_select(valid)
    return float(selected.mean().detach().cpu())


def _resolve_dtype(torch, has_cuda: bool):
    if not has_cuda:
        return torch.float32
    return (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )


def _evaluate_model_source(
    *,
    model_source: str,
    general_cases: list[dict],
    preference_pairs: list[dict],
    max_length: int,
    max_new_tokens: int,
    allow_cpu: bool,
) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Evaluation dependencies are missing. "
            "Inspect the Kaggle environment before installing replacements."
        ) from exc

    has_cuda = bool(torch.cuda.is_available())
    if not has_cuda and not allow_cpu:
        raise SystemExit(
            "CUDA GPU is required by default. Pass --allow-cpu only for smoke evaluation."
        )

    device = torch.device("cuda" if has_cuda else "cpu")
    dtype = _resolve_dtype(torch, has_cuda)

    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=dtype,
    )
    model.config.use_cache = True
    model.to(device)
    model.eval()

    general_details = []
    general_passes = 0
    for case in general_cases:
        output = _generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=str(case["prompt"]),
            torch=torch,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        passed = case_passes(output, case)
        general_passes += int(passed)
        general_details.append(
            {
                "id": str(case["id"]),
                "passed": bool(passed),
                "output": output,
            }
        )

    preference_details = []
    margins = []
    wins = 0
    for pair in preference_pairs:
        chosen_score = _mean_assistant_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=pair["prompt"],
            answer=pair["chosen"],
            torch=torch,
            device=device,
            max_length=max_length,
        )
        rejected_score = _mean_assistant_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=pair["prompt"],
            answer=pair["rejected"],
            torch=torch,
            device=device,
            max_length=max_length,
        )
        margin = chosen_score - rejected_score
        wins += int(margin > 0.0)
        margins.append(margin)
        preference_details.append(
            {
                "id": pair["id"],
                "chosen_logprob": chosen_score,
                "rejected_logprob": rejected_score,
                "margin": margin,
                "win": bool(margin > 0.0),
            }
        )

    result = {
        "model_source": model_source,
        "general_cases": len(general_cases),
        "general_passes": general_passes,
        "general_pass_rate": general_passes / len(general_cases),
        "preference_pairs": len(preference_pairs),
        "preference_wins": wins,
        "preference_win_rate": (
            wins / len(preference_pairs)
            if preference_pairs
            else 0.0
        ),
        "preference_mean_margin": mean(margins) if margins else 0.0,
        "general_details": general_details,
        "preference_details": preference_details,
    }

    del model
    del tokenizer
    gc.collect()
    if has_cuda:
        torch.cuda.empty_cache()
    return result


def evaluate(
    *,
    base_model: str,
    trained_model: str,
    eval_path: Path,
    general_cases_path: Path,
    max_length: int,
    max_new_tokens: int,
    max_general_drop: float,
    allow_cpu: bool,
) -> dict:
    general_cases = _load_general_cases(general_cases_path)
    preference_pairs = _load_preference_pairs(eval_path)

    baseline = _evaluate_model_source(
        model_source=base_model,
        general_cases=general_cases,
        preference_pairs=preference_pairs,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        allow_cpu=allow_cpu,
    )
    trained = _evaluate_model_source(
        model_source=trained_model,
        general_cases=general_cases,
        preference_pairs=preference_pairs,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        allow_cpu=allow_cpu,
    )

    if baseline["preference_pairs"] != trained["preference_pairs"]:
        raise RuntimeError("preference pair counts differ across evaluations")

    gate = compute_gate(
        baseline_general_pass_rate=baseline["general_pass_rate"],
        trained_general_pass_rate=trained["general_pass_rate"],
        baseline_preference_margin=baseline["preference_mean_margin"],
        trained_preference_margin=trained["preference_mean_margin"],
        preference_pairs=trained["preference_pairs"],
        max_general_drop=max_general_drop,
    )

    return {
        "schema_version": 1,
        "baseline_general_pass_rate": baseline["general_pass_rate"],
        "trained_general_pass_rate": trained["general_pass_rate"],
        "baseline_preference_win_rate": baseline["preference_win_rate"],
        "trained_preference_win_rate": trained["preference_win_rate"],
        "baseline_preference_margin": baseline["preference_mean_margin"],
        "trained_preference_margin": trained["preference_mean_margin"],
        "preference_pairs": trained["preference_pairs"],
        **gate,
        "baseline": baseline,
        "trained": trained,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--trained-model", required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument(
        "--general-cases",
        type=Path,
        default=Path(__file__).with_name("eval_cases.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-general-drop", type=float, default=0.05)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = evaluate(
        base_model=args.base_model,
        trained_model=args.trained_model,
        eval_path=args.eval,
        general_cases_path=args.general_cases,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        max_general_drop=args.max_general_drop,
        allow_cpu=args.allow_cpu,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passes_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
