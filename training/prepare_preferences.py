#!/usr/bin/env python3
"""Normalize H33 exported SFT/DPO JSONL into deterministic training splits."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence


@dataclass
class InputStats:
    total_lines: int = 0
    valid_sft_rows: int = 0
    valid_preference_rows: int = 0
    malformed_lines: int = 0
    empty_or_invalid_rows: int = 0


@dataclass
class PreparedData:
    sft: Dict[str, dict]
    preferences: Dict[str, List[dict]]
    stats: InputStats


@dataclass
class SplitData:
    train_ids: List[str]
    eval_ids: List[str]


def _clean(value) -> str:
    return "" if value is None else str(value).strip()


def _iter_jsonl(path: Path, stats: InputStats) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stats.total_lines += 1
            raw = line.strip()
            if not raw:
                stats.empty_or_invalid_rows += 1
                continue
            try:
                obj = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                stats.malformed_lines += 1
                continue
            if not isinstance(obj, dict):
                stats.empty_or_invalid_rows += 1
                continue
            yield obj


def _valid_sft(row: Mapping) -> dict | None:
    turn_id = _clean(row.get("id") or row.get("turn_id"))
    prompt = _clean(row.get("prompt"))
    response = _clean(row.get("response"))
    if not turn_id or not prompt or not response:
        return None
    return {"id": turn_id, "prompt": prompt, "response": response}


def _valid_preference(row: Mapping) -> dict | None:
    turn_id = _clean(row.get("id") or row.get("turn_id"))
    prompt = _clean(row.get("prompt"))
    chosen = _clean(row.get("chosen"))
    rejected = _clean(row.get("rejected"))
    if not turn_id or not prompt or not chosen or not rejected:
        return None
    if chosen == rejected:
        return None
    return {
        "id": turn_id,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def prepare_from_paths(
    sft_paths: Sequence[Path],
    preference_paths: Sequence[Path],
) -> PreparedData:
    """Load exports in supplied order; later valid SFT rows supersede earlier ones."""
    stats = InputStats()
    effective_sft: Dict[str, dict] = {}
    raw_preferences: Dict[str, List[dict]] = {}

    for path in sft_paths:
        for row in _iter_jsonl(Path(path), stats):
            normalized = _valid_sft(row)
            if normalized is None:
                stats.empty_or_invalid_rows += 1
                continue
            stats.valid_sft_rows += 1
            effective_sft[normalized["id"]] = normalized

    for path in preference_paths:
        for row in _iter_jsonl(Path(path), stats):
            normalized = _valid_preference(row)
            if normalized is None:
                stats.empty_or_invalid_rows += 1
                continue
            stats.valid_preference_rows += 1
            raw_preferences.setdefault(normalized["id"], []).append(normalized)

    canonical_preferences: Dict[str, List[dict]] = {}
    for turn_id, sft_row in effective_sft.items():
        canonical = sft_row["response"]
        prompt = sft_row["prompt"]

        # Every machine answer seen for this turn is a potential rejected answer.
        variants: List[str] = []
        seen = set()
        for row in raw_preferences.get(turn_id, []):
            for answer in (row["chosen"], row["rejected"]):
                answer = _clean(answer)
                if answer and answer != canonical and answer not in seen:
                    seen.add(answer)
                    variants.append(answer)

        if variants:
            canonical_preferences[turn_id] = [
                {
                    "id": turn_id,
                    "prompt": prompt,
                    "chosen": canonical,
                    "rejected": rejected,
                }
                for rejected in variants
            ]

    return PreparedData(
        sft=effective_sft,
        preferences=canonical_preferences,
        stats=stats,
    )


def split_effective_records(
    sft: Mapping[str, dict],
    preferences: Mapping[str, List[dict]],
    seed: int = 3407,
) -> SplitData:
    ids = sorted(_clean(key) for key in sft.keys() if _clean(key))
    if len(ids) < 2:
        raise ValueError("not enough data for train/eval split")

    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    eval_count = max(2, round(n * 0.10)) if n >= 20 else 1
    eval_ids = sorted(ids[:eval_count])
    eval_set = set(eval_ids)
    train_ids = sorted(turn_id for turn_id in ids if turn_id not in eval_set)
    return SplitData(train_ids=train_ids, eval_ids=eval_ids)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
    tmp.replace(path)


def _rows_for_ids(mapping: Mapping[str, dict], ids: Sequence[str]) -> Iterable[dict]:
    for turn_id in ids:
        row = mapping.get(turn_id)
        if row:
            yield row


def _preference_rows_for_ids(
    mapping: Mapping[str, List[dict]],
    ids: Sequence[str],
) -> Iterable[dict]:
    for turn_id in ids:
        for row in mapping.get(turn_id, []):
            yield row


def write_prepared_outputs(
    prepared: PreparedData,
    output_dir: Path,
    seed: int = 3407,
) -> dict:
    split = split_effective_records(prepared.sft, prepared.preferences, seed=seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_jsonl(out / "train_sft.jsonl", _rows_for_ids(prepared.sft, split.train_ids))
    _write_jsonl(
        out / "train_preferences.jsonl",
        _preference_rows_for_ids(prepared.preferences, split.train_ids),
    )

    eval_rows = []
    for turn_id in split.eval_ids:
        sft = prepared.sft[turn_id]
        rejected = [
            row["rejected"]
            for row in prepared.preferences.get(turn_id, [])
            if _clean(row.get("rejected"))
        ]
        eval_rows.append(
            {
                "id": turn_id,
                "prompt": sft["prompt"],
                "response": sft["response"],
                "rejected": rejected,
            }
        )
    _write_jsonl(out / "eval.jsonl", eval_rows)

    report = {
        "seed": seed,
        "train_turns": len(split.train_ids),
        "eval_turns": len(split.eval_ids),
        "train_preference_pairs": sum(
            len(prepared.preferences.get(i, [])) for i in split.train_ids
        ),
        "stats": asdict(prepared.stats),
        "train_ids": split.train_ids,
        "eval_ids": split.eval_ids,
    }
    (out / "prepare_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sft",
        type=Path,
        action="append",
        required=True,
        help="H33 SFT JSONL export. Repeat to apply later revisions in order.",
    )
    parser.add_argument(
        "--preferences",
        type=Path,
        action="append",
        default=[],
        help="H33 DPO/preference JSONL export. Repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    prepared = prepare_from_paths(args.sft, args.preferences)
    report = write_prepared_outputs(prepared, args.output_dir, seed=args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
