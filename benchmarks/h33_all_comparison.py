#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sacrebleu.metrics import CHRF
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 33
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "MBZUAI/Dialectal-Arabic-MMLU"
MCQ_LABELS = ["A", "B", "C", "D"]
JUDGE_LABELS = ["A", "B", "C", "N"]

EN_STOP = {
    "the","a","an","is","are","was","were","of","to","in","on","for","and","or",
    "what","which","who","how","many","much","there","this","that","these","those"
}

def norm_dialect(value):
    s = re.sub(r"[^a-z]", "", (value or "").lower())
    if s in {"en","eng","english"} or "english" in s:
        return "EN"
    if s in {"msa","modernstandardarabic"} or "modernstandard" in s:
        return "MSA"
    if s in {"syr","syrian","syrianarabic"} or "syrian" in s:
        return "SYR"
    return (value or "").strip().upper()

def paired_sample(ds, per_domain):
    grouped = defaultdict(dict)
    for row in ds:
        grouped[(row["domain"], int(row["qid"]))][norm_dialect(row["dialect"])] = row
    by_domain = defaultdict(list)
    for key, variants in grouped.items():
        if all(x in variants for x in ("EN","MSA","SYR")):
            by_domain[key[0]].append((key, variants))
    rng = random.Random(SEED)
    selected = []
    for domain in sorted(by_domain):
        items = sorted(by_domain[domain], key=lambda x: x[0][1])
        rng.shuffle(items)
        selected.extend(items[:per_domain])
    return selected

def prepare(out_path, per_domain):
    ds = load_dataset(DATASET_ID, split="test")
    selected = paired_sample(ds, per_domain)
    rows = []
    for (domain, qid), variants in selected:
        row = {"domain": domain, "qid": qid, "answer": int(variants["EN"]["answer"])}
        for d in ("EN","MSA","SYR"):
            row[d.lower()] = {
                "question": variants[d]["question"],
                "choices": list(variants[d]["choices"]),
            }
        rows.append(row)

    probes = [
        {"id":"digits_1","text":"في عام 2026 استخدم OpenAI نموذج GPT-5 في لندن."},
        {"id":"digits_2","text":"سعر الجهاز 1299 جنيه ورقم الإصدار 4.2.1."},
        {"id":"url_1","text":"افتح https://example.com ثم اكتب H33 و Qwen2.5."},
        {"id":"name_1","text":"اجتمع أحمد مع شركة Microsoft في Birmingham يوم 20/09/2026."},
    ]
    payload = {
        "seed": SEED,
        "dataset_id": DATASET_ID,
        "dataset_rows": len(ds),
        "sample_n": len(rows),
        "rows": rows,
        "probes": probes,
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"prepared": str(p), "sample_n": len(rows), "dataset_rows": len(ds)}))

def render_chat(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role":"user","content":prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

def option_prompt(question, choices, instruction):
    lines = "\n".join(f"{MCQ_LABELS[i]}. {c}" for i,c in enumerate(choices))
    if instruction == "ar":
        lead = "اختر الإجابة الصحيحة. أجب بحرف واحد فقط: A أو B أو C أو D."
    else:
        lead = "Choose the correct option. Reply with one letter only: A, B, C, or D."
    return f"{lead}\n\n{question}\n{lines}\n\nAnswer:"

def token_ids(tokenizer, labels):
    out = {}
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            ids = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Label {label!r} is not single-token: {ids}")
        out[label] = ids[0]
    return out

def batch_choice_eval(model, tokenizer, items, labels, instruction, batch_size=8):
    tids = token_ids(tokenizer, labels)
    preds, confs, correct_flags, latencies = [], [], [], []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start+batch_size]
        prompts = [render_chat(tokenizer, option_prompt(x["question"], x["choices"], instruction)) for x in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(**enc).logits[:, -1, :]
        elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.extend([elapsed / max(1, len(chunk))] * len(chunk))
        idxs = torch.tensor([tids[x] for x in labels], dtype=torch.long)
        selected = logits[:, idxs]
        probs = torch.softmax(selected, dim=-1)
        best = torch.argmax(probs, dim=-1).tolist()
        for j, item in enumerate(chunk):
            pred = labels[best[j]]
            preds.append(pred)
            confs.append(float(probs[j, best[j]]))
            gold = MCQ_LABELS[int(item["answer"])]
            correct_flags.append(int(pred == gold))
    return {
        "n": len(items),
        "accuracy": float(np.mean(correct_flags)) if correct_flags else 0.0,
        "correct": correct_flags,
        "predictions": preds,
        "confidence": confs,
        "latency_ms_mean": float(np.mean(latencies)) if latencies else 0.0,
        "latency_ms_p95": float(np.percentile(latencies,95)) if latencies else 0.0,
    }

def bootstrap_diff(a, b, rounds=3000):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return {"diff":None,"ci95":[None,None]}
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(rounds):
        idx = rng.integers(0, len(a), len(a))
        diffs.append(float(np.mean(a[idx] - b[idx])))
    return {
        "diff": float(np.mean(a-b)),
        "ci95":[float(np.percentile(diffs,2.5)), float(np.percentile(diffs,97.5))]
    }

def tokenize_en(text):
    toks = re.findall(r"[a-z0-9][a-z0-9._/-]*", (text or "").lower())
    return [t for t in toks if len(t)>1 and t not in EN_STOP]

def lexical_scores(evidence, candidates):
    ev = set(tokenize_en(evidence))
    scores = []
    for c in candidates:
        ct = set(tokenize_en(c))
        if not ct or not ev:
            scores.append(0.0)
        else:
            scores.append(len(ev & ct) / max(1, len(ct)))
    return scores

def rules_pick(evidence, candidates):
    scores = lexical_scores(evidence, candidates)
    order = sorted(range(len(scores)), key=lambda i:scores[i], reverse=True)
    best = scores[order[0]]
    second = scores[order[1]] if len(order)>1 else 0.0
    if best < 0.35 or (best-second) < 0.10:
        return "N", scores
    return JUDGE_LABELS[order[0]], scores

def judge_prompt(case):
    a,b,c = case["candidates"]
    return (
        "Use only the supplied evidence. Choose the candidate best supported by the evidence. "
        "If none is supported, choose N. Reply with one letter only: A, B, C, or N.\n\n"
        f"Question: {case['question']}\nEvidence: {case['evidence']}\n"
        f"Candidate A: {a}\nCandidate B: {b}\nCandidate C: {c}\n"
        "Candidate N: None of the candidates is supported.\n\nDecision:"
    )

def judge_logits(model, tokenizer, cases, batch_size=8):
    tids = token_ids(tokenizer, JUDGE_LABELS)
    idxs = torch.tensor([tids[x] for x in JUDGE_LABELS], dtype=torch.long)
    preds, probs_all, latency = [], [], []
    for start in range(0,len(cases),batch_size):
        chunk = cases[start:start+batch_size]
        prompts = [render_chat(tokenizer, judge_prompt(c)) for c in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        t0=time.perf_counter()
        with torch.inference_mode():
            logits = model(**enc).logits[:, -1, :]
        elapsed=(time.perf_counter()-t0)*1000.0
        latency.extend([elapsed/max(1,len(chunk))]*len(chunk))
        probs=torch.softmax(logits[:,idxs],dim=-1)
        best=torch.argmax(probs,dim=-1).tolist()
        for j in range(len(chunk)):
            preds.append(JUDGE_LABELS[best[j]])
            probs_all.append([float(x) for x in probs[j].tolist()])
    return preds, probs_all, latency

def judge_generate(model, tokenizer, cases, batch_size=8):
    preds, latency = [], []
    for start in range(0,len(cases),batch_size):
        chunk=cases[start:start+batch_size]
        prompts=[render_chat(tokenizer, judge_prompt(c)) for c in chunk]
        enc=tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        t0=time.perf_counter()
        with torch.inference_mode():
            out=model.generate(
                **enc,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed=(time.perf_counter()-t0)*1000.0
        latency.extend([elapsed/max(1,len(chunk))]*len(chunk))
        new=out[:,enc["input_ids"].shape[1]:]
        texts=tokenizer.batch_decode(new, skip_special_tokens=True)
        for txt in texts:
            m=re.search(r"[ABCN]", txt.upper())
            preds.append(m.group(0) if m else "?")
    return preds, latency

def ece(conf, correct, bins=10):
    conf=np.asarray(conf,dtype=float)
    cor=np.asarray(correct,dtype=float)
    if not len(conf): return 0.0
    out=0.0
    for i in range(bins):
        lo=i/bins; hi=(i+1)/bins
        mask=(conf>=lo)&(conf<hi if i<bins-1 else conf<=hi)
        if np.any(mask):
            out += float(np.mean(mask))*abs(float(np.mean(conf[mask]))-float(np.mean(cor[mask])))
    return out

def preservation_ratio(src, dst):
    important = re.findall(r"https?://\S+|\b\d+(?:[./-]\d+)*\b|\b[A-Za-z][A-Za-z0-9._-]*\b", src or "")
    if not important:
        return 1.0
    hit=sum(1 for x in important if x in (dst or ""))
    return hit/len(important)

def chrf_mean(hyps, refs):
    metric=CHRF(word_order=2)
    vals=[]
    for h,r in zip(hyps,refs):
        vals.append(metric.sentence_score(h,[r]).score/100.0)
    return float(np.mean(vals)) if vals else 0.0

def evaluate(input_path, translations_path, outdir):
    source=json.loads(Path(input_path).read_text(encoding="utf-8"))
    trans=json.loads(Path(translations_path).read_text(encoding="utf-8"))
    tmap={(x["domain"],int(x["qid"])):x for x in trans["rows"] if x.get("ok",False)}

    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=False
    )
    model.eval()
    torch.set_num_threads(max(1,min(8,os.cpu_count() or 4)))

    rows=source["rows"]
    def make_items(kind):
        out=[]
        for r in rows:
            key=(r["domain"],int(r["qid"]))
            if kind=="en":
                q=r["en"]["question"]; choices=r["en"]["choices"]
            elif kind=="msa":
                q=r["msa"]["question"]; choices=r["msa"]["choices"]
            elif kind=="syr":
                q=r["syr"]["question"]; choices=r["syr"]["choices"]
            elif kind=="msa_mlkit":
                if key not in tmap: continue
                q=tmap[key]["msa"]["question_en"]; choices=tmap[key]["msa"]["choices_en"]
            elif kind=="syr_mlkit":
                if key not in tmap: continue
                q=tmap[key]["syr"]["question_en"]; choices=tmap[key]["syr"]["choices_en"]
            else:
                raise KeyError(kind)
            out.append({"domain":r["domain"],"qid":r["qid"],"answer":r["answer"],"question":q,"choices":choices})
        return out

    modes={
        "english_human": batch_choice_eval(model,tokenizer,make_items("en"),MCQ_LABELS,"en"),
        "msa_direct_ar": batch_choice_eval(model,tokenizer,make_items("msa"),MCQ_LABELS,"ar"),
        "syrian_direct_ar": batch_choice_eval(model,tokenizer,make_items("syr"),MCQ_LABELS,"ar"),
        "msa_direct_en_instruction": batch_choice_eval(model,tokenizer,make_items("msa"),MCQ_LABELS,"en"),
        "syrian_direct_en_instruction": batch_choice_eval(model,tokenizer,make_items("syr"),MCQ_LABELS,"en"),
        "msa_mlkit_en": batch_choice_eval(model,tokenizer,make_items("msa_mlkit"),MCQ_LABELS,"en"),
        "syrian_mlkit_en": batch_choice_eval(model,tokenizer,make_items("syr_mlkit"),MCQ_LABELS,"en"),
    }

    paired_syr=bootstrap_diff(modes["syrian_mlkit_en"]["correct"],modes["syrian_direct_ar"]["correct"])
    paired_msa=bootstrap_diff(modes["msa_mlkit_en"]["correct"],modes["msa_direct_ar"]["correct"])

    translation_rows=trans["rows"]
    expected=len(rows)
    ok_count=sum(1 for x in translation_rows if x.get("ok",False))
    failure_rate=1.0-(ok_count/max(1,expected))
    preserve=[]
    en_similarity_msa=[]; en_similarity_syr=[]; back_similarity_msa=[]; back_similarity_syr=[]
    src_en_q=[]; ml_msa_q=[]; ml_syr_q=[]; src_msa_q=[]; back_msa_q=[]; src_syr_q=[]; back_syr_q=[]
    source_map={(r["domain"],int(r["qid"])):r for r in rows}
    for tr in translation_rows:
        if not tr.get("ok",False): continue
        r=source_map[(tr["domain"],int(tr["qid"]))]
        for d in ("msa","syr"):
            preserve.append(preservation_ratio(r[d]["question"],tr[d]["question_en"]))
            preserve.append(preservation_ratio(r[d]["question"],tr[d]["question_back_ar"]))
        src_en_q.append(r["en"]["question"])
        ml_msa_q.append(tr["msa"]["question_en"]); ml_syr_q.append(tr["syr"]["question_en"])
        src_msa_q.append(r["msa"]["question"]); back_msa_q.append(tr["msa"]["question_back_ar"])
        src_syr_q.append(r["syr"]["question"]); back_syr_q.append(tr["syr"]["question_back_ar"])

    translation_metrics={
        "expected_rows":expected,
        "ok_rows":ok_count,
        "failure_rate":failure_rate,
        "preservation_ratio":float(np.mean(preserve)) if preserve else 0.0,
        "msa_to_human_en_chrf":chrf_mean(ml_msa_q,src_en_q),
        "syr_to_human_en_chrf":chrf_mean(ml_syr_q,src_en_q),
        "msa_roundtrip_chrf":chrf_mean(back_msa_q,src_msa_q),
        "syr_roundtrip_chrf":chrf_mean(back_syr_q,src_syr_q),
        "probe_results":trans.get("probes",[]),
    }

    # Judge cases use ML Kit-translated MSA correct option as evidence and human English choices as candidates.
    rng=random.Random(SEED+9)
    judge_cases=[]
    for idx,r in enumerate(rows):
        tr=tmap.get((r["domain"],int(r["qid"])))
        if not tr: continue
        gold_idx=int(r["answer"])
        evidence=tr["msa"]["choices_en"][gold_idx]
        all_choices=list(r["en"]["choices"])
        wrong=[c for i,c in enumerate(all_choices) if i!=gold_idx]
        if idx % 4 == 0:
            candidates=wrong[:3]
            gold="N"
        else:
            candidates=[all_choices[gold_idx]] + rng.sample(wrong,2)
            rng.shuffle(candidates)
            gold=JUDGE_LABELS[candidates.index(all_choices[gold_idx])]
        judge_cases.append({
            "question":r["en"]["question"],
            "evidence":evidence,
            "candidates":candidates,
            "gold":gold,
        })

    lp, lprobs, llat = judge_logits(model,tokenizer,judge_cases)
    gp, glat = judge_generate(model,tokenizer,judge_cases)
    rp=[]; rscores=[]
    for c in judge_cases:
        pred,scores=rules_pick(c["evidence"],c["candidates"])
        rp.append(pred); rscores.append(scores)

    ep=[]
    for c, probs, lex in zip(judge_cases,lprobs,rscores):
        bestlex=max(lex) if lex else 0.0
        scores=[
            0.55*probs[0]+0.45*lex[0],
            0.55*probs[1]+0.45*lex[1],
            0.55*probs[2]+0.45*lex[2],
            0.55*probs[3]+0.45*(1.0-bestlex),
        ]
        ep.append(JUDGE_LABELS[int(np.argmax(scores))])

    gold=[c["gold"] for c in judge_cases]
    def acc(pred):
        return float(np.mean([int(a==b) for a,b in zip(pred,gold)])) if gold else 0.0
    def abst(pred):
        idx=[i for i,g in enumerate(gold) if g=="N"]
        return float(np.mean([int(pred[i]=="N") for i in idx])) if idx else 0.0
    logit_correct=[int(a==b) for a,b in zip(lp,gold)]
    conf=[max(p) for p in lprobs]
    judge={
        "n":len(gold),
        "rules_accuracy":acc(rp),
        "logit_accuracy":acc(lp),
        "generated_accuracy":acc(gp),
        "ensemble_accuracy":acc(ep),
        "rules_abstention":abst(rp),
        "logit_abstention":abst(lp),
        "generated_abstention":abst(gp),
        "ensemble_abstention":abst(ep),
        "logit_latency_ms_mean":float(np.mean(llat)) if llat else 0.0,
        "generated_latency_ms_mean":float(np.mean(glat)) if glat else 0.0,
        "logit_speedup_fraction":1.0-(float(np.mean(llat))/float(np.mean(glat))) if glat and np.mean(glat)>0 else 0.0,
        "logit_ece":ece(conf,logit_correct),
    }

    pivot_gate=bool(
        paired_syr["diff"] is not None
        and paired_syr["diff"] >= 0.03
        and paired_syr["ci95"][0] > 0.0
        and paired_msa["diff"] is not None
        and paired_msa["diff"] >= -0.02
        and failure_rate <= 0.02
        and translation_metrics["preservation_ratio"] >= 0.98
    )
    judge_gate=bool(
        judge["logit_accuracy"] >= 0.80
        and judge["logit_abstention"] >= 0.70
        and judge["logit_accuracy"] + 0.02 >= judge["generated_accuracy"]
        and judge["logit_speedup_fraction"] >= 0.25
    )
    ensemble_gate=bool(
        judge["ensemble_accuracy"] >= max(judge["rules_accuracy"],judge["logit_accuracy"]) + 0.03
        and judge["ensemble_abstention"] >= 0.70
    )

    result={
        "seed":SEED,
        "model_id":MODEL_ID,
        "dataset_id":DATASET_ID,
        "sample_n":len(rows),
        "language":{
            k:{x:v[x] for x in ("n","accuracy","latency_ms_mean","latency_ms_p95")}
            for k,v in modes.items()
        },
        "paired_diffs":{
            "mlkit_syr_minus_direct_syr":paired_syr,
            "mlkit_msa_minus_direct_msa":paired_msa,
        },
        "translation":translation_metrics,
        "judge":judge,
        "gates":{
            "mlkit_pivot":pivot_gate,
            "litjev_logit":judge_gate,
            "rules_plus_litjev_ensemble":ensemble_gate,
        },
        "implementation_decision":{
            "local_reasoning_path":"mlkit_ar_to_en_then_qwen" if pivot_gate else "direct_input_language",
            "judge_path":"rules_plus_litjev" if ensemble_gate else ("litjev_secondary" if judge_gate else "deterministic_evidence_rules_only"),
        },
    }
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=[
        "# H33 All-In-One Comparison",
        "",
        f"- sample: {len(rows)} parallel EN/MSA/SYR questions",
        "",
        "## Language paths",
    ]
    for k,v in result["language"].items():
        lines.append(f"- {k}: {v['accuracy']:.4f} (n={v['n']}, mean {v['latency_ms_mean']:.1f} ms)")
    lines += [
        "",
        f"- ML Kit SYR - direct SYR: {paired_syr['diff']:.4f}, 95% CI {paired_syr['ci95']}",
        f"- ML Kit MSA - direct MSA: {paired_msa['diff']:.4f}, 95% CI {paired_msa['ci95']}",
        f"- Translation failure rate: {failure_rate:.4f}",
        f"- Preservation ratio: {translation_metrics['preservation_ratio']:.4f}",
        f"- MSA→human EN chrF: {translation_metrics['msa_to_human_en_chrf']:.4f}",
        f"- SYR→human EN chrF: {translation_metrics['syr_to_human_en_chrf']:.4f}",
        f"- MSA round-trip chrF: {translation_metrics['msa_roundtrip_chrf']:.4f}",
        f"- SYR round-trip chrF: {translation_metrics['syr_roundtrip_chrf']:.4f}",
        f"- PIVOT_GATE: {'PASS' if pivot_gate else 'FAIL'}",
        "",
        "## Judge paths",
        f"- rules accuracy: {judge['rules_accuracy']:.4f}",
        f"- logit accuracy: {judge['logit_accuracy']:.4f}",
        f"- generated accuracy: {judge['generated_accuracy']:.4f}",
        f"- ensemble accuracy: {judge['ensemble_accuracy']:.4f}",
        f"- logit abstention: {judge['logit_abstention']:.4f}",
        f"- ensemble abstention: {judge['ensemble_abstention']:.4f}",
        f"- logit speedup fraction: {judge['logit_speedup_fraction']:.4f}",
        f"- LITJEV_GATE: {'PASS' if judge_gate else 'FAIL'}",
        f"- ENSEMBLE_GATE: {'PASS' if ensemble_gate else 'FAIL'}",
        "",
        "## Build decision",
        f"- local reasoning: {result['implementation_decision']['local_reasoning_path']}",
        f"- judge: {result['implementation_decision']['judge_path']}",
        "",
        "Thresholds are fixed in this script before the run and are not changed after results.",
    ]
    (out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))

def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("prepare")
    p.add_argument("--out",required=True)
    p.add_argument("--per-domain",type=int,default=2)
    e=sub.add_parser("evaluate")
    e.add_argument("--input",required=True)
    e.add_argument("--translations",required=True)
    e.add_argument("--outdir",required=True)
    args=ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if args.cmd=="prepare":
        prepare(args.out,args.per_domain)
    else:
        evaluate(args.input,args.translations,args.outdir)

if __name__=="__main__":
    main()
