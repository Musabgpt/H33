#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 33
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "MBZUAI/Dialectal-Arabic-MMLU"
LABELS = ["A", "B", "C", "D"]
JUDGE_LABELS = ["A", "B", "C", "N"]


def norm_dialect(value: str) -> str:
    s = re.sub(r"[^a-z]", "", (value or "").lower())
    if s in {"en", "eng", "english"} or "english" in s:
        return "EN"
    if s in {"msa", "modernstandardarabic"} or "modernstandard" in s:
        return "MSA"
    if s in {"syr", "syrian", "syrianarabic"} or "syrian" in s:
        return "SYR"
    return (value or "").strip().upper()


def build_mcq_prompt(row, instruction_lang: str) -> str:
    if instruction_lang == "ar":
        lead = "اختر الإجابة الصحيحة. أجب بحرف واحد فقط: A أو B أو C أو D."
    else:
        lead = "Choose the correct option. Reply with one letter only: A, B, C, or D."
    choices = "\n".join(f"{LABELS[i]}. {c}" for i, c in enumerate(row["choices"]))
    return f"{lead}\n\n{row['question']}\n{choices}\n\nAnswer:"


def encode_chat(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not isinstance(ids, torch.Tensor):
        try:
            ids = ids["input_ids"]
        except (KeyError, TypeError):
            pass
    if not isinstance(ids, torch.Tensor):
        raise TypeError(f"chat template returned {type(ids).__name__}, expected input_ids Tensor")
    return ids


def single_token_ids(tokenizer, labels):
    out = {}
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            ids2 = tokenizer.encode(" " + label, add_special_tokens=False)
            if len(ids2) == 1:
                ids = ids2
        if len(ids) != 1:
            raise RuntimeError(f"Decision label {label!r} is not single-token: {ids}")
        out[label] = ids[0]
    return out


def choose_from_logits(model, tokenizer, prompt, token_ids, ordered_labels):
    ids = encode_chat(tokenizer, prompt)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits[0, -1]
    vals = torch.tensor([float(logits[token_ids[x]]) for x in ordered_labels])
    probs = torch.softmax(vals, dim=0).cpu().numpy()
    idx = int(np.argmax(probs))
    return ordered_labels[idx], float(probs[idx]), probs.tolist()


def mcq_eval(model, tokenizer, rows, instruction_lang, token_ids):
    correct = []
    latencies = []
    preds = []
    for row in rows:
        t0 = time.perf_counter()
        pred, conf, probs = choose_from_logits(
            model, tokenizer, build_mcq_prompt(row, instruction_lang), token_ids, LABELS
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        gold = LABELS[int(row["answer"])]
        correct.append(int(pred == gold))
        preds.append({"pred": pred, "gold": gold, "confidence": conf, "probs": probs})
    return {
        "accuracy": float(np.mean(correct)) if correct else 0.0,
        "n": len(correct),
        "latency_ms_mean": float(np.mean(latencies)) if latencies else 0.0,
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "correct": correct,
        "predictions": preds,
    }


def bootstrap_diff(a, b, rounds=4000):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b) or len(a) == 0:
        return {"diff": None, "ci95": [None, None]}
    rng = np.random.default_rng(SEED)
    diffs = np.empty(rounds, dtype=np.float64)
    n = len(a)
    for i in range(rounds):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.mean(a[idx] - b[idx])
    return {
        "diff": float(np.mean(a - b)),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    }


def ece(confidences, correct, bins=10):
    conf = np.asarray(confidences, dtype=np.float64)
    cor = np.asarray(correct, dtype=np.float64)
    if len(conf) == 0:
        return 0.0
    total = 0.0
    for lo in np.linspace(0.0, 0.9, bins):
        hi = lo + 0.1
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not np.any(mask):
            continue
        total += np.mean(mask) * abs(float(np.mean(conf[mask])) - float(np.mean(cor[mask])))
    return float(total)


def build_judge_cases(en_rows, count=160):
    rng = random.Random(SEED + 1)
    cases = []
    for idx, row in enumerate(en_rows[:count]):
        correct_idx = int(row["answer"])
        correct_text = row["choices"][correct_idx]
        wrongs = [c for i, c in enumerate(row["choices"]) if i != correct_idx]
        if idx % 4 == 0:
            candidates = rng.sample(wrongs, 3)
            gold = "N"
            evidence = "The provided evidence is insufficient to establish which candidate is correct."
        else:
            candidates = [correct_text] + rng.sample(wrongs, 2)
            rng.shuffle(candidates)
            gold = JUDGE_LABELS[candidates.index(correct_text)]
            evidence = f"Verified evidence states that the correct answer is: {correct_text}"
        cases.append({"question": row["question"], "candidates": candidates, "evidence": evidence, "gold": gold})
    return cases


def build_judge_prompt(case):
    a, b, c = case["candidates"]
    return (
        "Use only the supplied evidence. Pick the candidate best supported by it. "
        "If the evidence does not support any candidate, choose N. "
        "Reply with one letter only: A, B, C, or N.\n\n"
        f"Question: {case['question']}\n"
        f"Evidence: {case['evidence']}\n"
        f"Candidate A: {a}\nCandidate B: {b}\nCandidate C: {c}\n"
        "Candidate N: None of the candidates is supported by the evidence.\n\nDecision:"
    )


def parse_generated(text):
    m = re.search(r"\b([ABCN])\b", text.upper())
    return m.group(1) if m else "?"


def judge_eval(model, tokenizer, cases, judge_token_ids):
    logit_correct, gen_correct = [], []
    confs = []
    logit_ms, gen_ms = [], []
    abstain_logit = []
    rows = []
    for case in cases:
        prompt = build_judge_prompt(case)
        t0 = time.perf_counter()
        lp, conf, probs = choose_from_logits(model, tokenizer, prompt, judge_token_ids, JUDGE_LABELS)
        logit_ms.append((time.perf_counter() - t0) * 1000)

        ids = encode_chat(tokenizer, prompt)
        t1 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                ids,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ms.append((time.perf_counter() - t1) * 1000)
        generated = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        gp = parse_generated(generated)

        gold = case["gold"]
        logit_correct.append(int(lp == gold))
        gen_correct.append(int(gp == gold))
        confs.append(conf)
        if gold == "N":
            abstain_logit.append(int(lp == "N"))
        rows.append({"gold": gold, "logit": lp, "generated": gp, "confidence": conf, "probs": probs})

    logit_acc = float(np.mean(logit_correct))
    gen_acc = float(np.mean(gen_correct))
    abstain_acc = float(np.mean(abstain_logit)) if abstain_logit else 0.0
    lmean = float(np.mean(logit_ms))
    gmean = float(np.mean(gen_ms))
    return {
        "n": len(cases),
        "logit_accuracy": logit_acc,
        "generated_accuracy": gen_acc,
        "abstention_accuracy": abstain_acc,
        "logit_latency_ms_mean": lmean,
        "generated_latency_ms_mean": gmean,
        "speedup_fraction": 1.0 - (lmean / gmean if gmean else 1.0),
        "ece": ece(confs, logit_correct),
        "rows": rows,
    }


def stratified_parallel_sample(ds, per_domain=8):
    groups = defaultdict(dict)
    dialect_values = set()
    for row in ds:
        d = norm_dialect(row["dialect"])
        dialect_values.add((row["dialect"], d))
        groups[(row["domain"], int(row["qid"]))][d] = row
    by_domain = defaultdict(list)
    for key, variants in groups.items():
        if all(x in variants for x in ("EN", "MSA", "SYR")):
            by_domain[key[0]].append((key, variants))
    rng = random.Random(SEED)
    selected = []
    for domain in sorted(by_domain):
        items = sorted(by_domain[domain], key=lambda x: x[0][1])
        rng.shuffle(items)
        selected.extend(items[:per_domain])
    return selected, sorted(dialect_values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/h33-bilingual-benchmark")
    ap.add_argument("--per-domain", type=int, default=8)
    ap.add_argument("--judge-cases", type=int, default=160)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_ID, split="test")
    selected, dialect_values = stratified_parallel_sample(ds, args.per_domain)
    if len(selected) < 100:
        raise RuntimeError(f"Too few parallel EN/MSA/SYR rows: {len(selected)}; dialects={dialect_values}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()

    mcq_token_ids = single_token_ids(tokenizer, LABELS)
    judge_token_ids = single_token_ids(tokenizer, JUDGE_LABELS)

    en_rows = [variants["EN"] for _, variants in selected]
    msa_rows = [variants["MSA"] for _, variants in selected]
    syr_rows = [variants["SYR"] for _, variants in selected]

    modes = {
        "en_en_instruction": (en_rows, "en"),
        "msa_ar_instruction": (msa_rows, "ar"),
        "msa_en_instruction": (msa_rows, "en"),
        "syr_ar_instruction": (syr_rows, "ar"),
        "syr_en_instruction": (syr_rows, "en"),
    }
    evals = {}
    for name, (rows, lang) in modes.items():
        print(f"EVAL {name} n={len(rows)}", flush=True)
        evals[name] = mcq_eval(model, tokenizer, rows, lang, mcq_token_ids)

    syr_diff = bootstrap_diff(evals["en_en_instruction"]["correct"], evals["syr_ar_instruction"]["correct"])
    msa_diff = bootstrap_diff(evals["en_en_instruction"]["correct"], evals["msa_ar_instruction"]["correct"])
    pivot_gate = bool(syr_diff["diff"] is not None and syr_diff["diff"] >= 0.03 and syr_diff["ci95"][0] > 0.0)

    judge_cases = build_judge_cases(en_rows, min(args.judge_cases, len(en_rows)))
    judge = judge_eval(model, tokenizer, judge_cases, judge_token_ids)
    judge_gate = bool(
        judge["logit_accuracy"] >= 0.80
        and judge["abstention_accuracy"] >= 0.70
        and judge["logit_accuracy"] + 0.02 >= judge["generated_accuracy"]
        and judge["speedup_fraction"] >= 0.25
    )

    compact_evals = {
        k: {x: v[x] for x in ("accuracy", "n", "latency_ms_mean", "latency_ms_p95")}
        for k, v in evals.items()
    }
    result = {
        "seed": SEED,
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "dataset_rows": len(ds),
        "parallel_sample_n": len(selected),
        "dialect_values": dialect_values,
        "decision_token_ids": {"mcq": mcq_token_ids, "judge": judge_token_ids},
        "language": compact_evals,
        "paired_diffs": {"en_minus_syr_ar": syr_diff, "en_minus_msa_ar": msa_diff},
        "pivot_gate_passed": pivot_gate,
        "judge": {k: v for k, v in judge.items() if k != "rows"},
        "judge_gate_passed": judge_gate,
    }
    (outdir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# H33 Bilingual / Judge Benchmark",
        "",
        f"- Model: {MODEL_ID}",
        f"- Dataset: {DATASET_ID} ({len(ds)} rows)",
        f"- Parallel EN/MSA/SYR sample: {len(selected)}",
        "",
        "## Language accuracy",
        "",
    ]
    for name, value in compact_evals.items():
        lines.append(f"- {name}: {value['accuracy']:.4f} (n={value['n']}, mean {value['latency_ms_mean']:.1f} ms)")
    lines += [
        "",
        f"- EN minus SYR-AR: {syr_diff['diff']:.4f}, 95% CI {syr_diff['ci95']}",
        f"- EN minus MSA-AR: {msa_diff['diff']:.4f}, 95% CI {msa_diff['ci95']}",
        f"- PIVOT_GATE: {'PASS' if pivot_gate else 'FAIL'}",
        "",
        "## Judge",
        "",
        f"- Logit accuracy: {judge['logit_accuracy']:.4f}",
        f"- Generated accuracy: {judge['generated_accuracy']:.4f}",
        f"- Abstention accuracy: {judge['abstention_accuracy']:.4f}",
        f"- Mean logit latency: {judge['logit_latency_ms_mean']:.1f} ms",
        f"- Mean generated latency: {judge['generated_latency_ms_mean']:.1f} ms",
        f"- Speedup fraction: {judge['speedup_fraction']:.4f}",
        f"- Raw-confidence ECE: {judge['ece']:.4f}",
        f"- JUDGE_GATE: {'PASS' if judge_gate else 'FAIL'}",
        "",
        "The gates above use thresholds fixed before the run; do not reinterpret them after seeing the results.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
