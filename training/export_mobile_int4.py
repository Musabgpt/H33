#!/usr/bin/env python3
"""Export a passing partial-Qwen checkpoint to ORT GenAI INT4 for H33."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def require_passing_evaluation(report: dict) -> dict:
    if not isinstance(report, dict):
        raise ValueError("evaluation report must be an object")
    if int(report.get("schema_version", 0)) != 1:
        raise ValueError("unsupported evaluation report schema")

    passes = bool(report.get("passes_gate", False))
    general_ok = bool(report.get("general_ok", False))
    preference_ok = bool(report.get("preference_ok", False))

    if passes and not (general_ok and preference_ok):
        raise ValueError("evaluation report is inconsistent")
    if not passes:
        raise ValueError("evaluation gate failed; refusing INT4 export")
    return report


def require_original_weight_update(summary: dict) -> bool:
    if not isinstance(summary, dict):
        raise ValueError("training summary must be an object")

    steps = int(summary.get("optimizer_steps", 0))
    changed = bool(summary.get("original_weights_changed", False))
    if steps <= 0 or not changed:
        raise ValueError(
            "no proof that selected original Qwen weights changed; refusing export"
        )
    return True


def load_training_summary(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    require_original_weight_update(summary)
    return summary


def load_evaluation_report(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    return require_passing_evaluation(report)


def build_builder_command(
    *,
    python_executable: str,
    trained_model: Path,
    output_dir: Path,
) -> list[str]:
    return [
        str(python_executable),
        "-m",
        "onnxruntime_genai.models.builder",
        "-m",
        str(Path(trained_model)),
        "-o",
        str(Path(output_dir)),
        "-p",
        "int4",
        "-e",
        "cpu",
        "--extra_options",
        "hf_remote=false",
        "hf_token=false",
    ]


def run_export(args) -> int:
    report = load_evaluation_report(args.evaluation_report)

    trained_model = Path(args.trained_model)
    training_summary_path = (
        Path(args.training_summary)
        if args.training_summary is not None
        else trained_model.parent / "training_summary.json"
    )
    training_summary = load_training_summary(training_summary_path)
    if not trained_model.is_dir():
        raise FileNotFoundError(f"trained model directory missing: {trained_model}")
    if not (trained_model / "config.json").is_file():
        raise FileNotFoundError(
            f"trained model config.json missing: {trained_model}"
        )

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output directory already exists: {output_dir}; "
                "pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    command = build_builder_command(
        python_executable=sys.executable,
        trained_model=trained_model,
        output_dir=output_dir,
    )
    print(json.dumps({"builder_command": command}, ensure_ascii=False))
    subprocess.run(command, check=True)

    genai_config = output_dir / "genai_config.json"
    if not genai_config.is_file():
        raise RuntimeError("ORT GenAI export missing genai_config.json")

    smoke_env = os.environ.copy()
    smoke_env["H33_MODEL_DIR"] = str(output_dir.resolve())
    subprocess.run(
        [sys.executable, "scripts/smoke_qwen_genai.py"],
        check=True,
        env=smoke_env,
    )

    summary = {
        "schema_version": 1,
        "trained_model": str(trained_model.resolve()),
        "output_dir": str(output_dir.resolve()),
        "evaluation_report": str(Path(args.evaluation_report).resolve()),
        "training_summary": str(training_summary_path.resolve()),
        "original_weights_changed": bool(
            training_summary["original_weights_changed"]
        ),
        "evaluation_general_ok": bool(report["general_ok"]),
        "evaluation_preference_ok": bool(report["preference_ok"]),
        "evaluation_passes_gate": bool(report["passes_gate"]),
        "smoke_tested": True,
    }
    summary_path = output_dir.parent / "mobile_export_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-model", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return run_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
