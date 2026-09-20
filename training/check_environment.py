#!/usr/bin/env python3
"""Print a machine-readable Kaggle/runtime compatibility snapshot before installs."""

from __future__ import annotations

import importlib.metadata
import json
import platform


PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "huggingface-hub",
    "numpy",
    "pandas",
    "pyarrow",
    "onnx",
    "onnxruntime",
    "onnxruntime-genai",
]


def version_or_none(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_report() -> dict:
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version_or_none(name) for name in PACKAGES},
        "cuda_available": False,
        "cuda": None,
        "gpu_count": 0,
        "gpus": [],
    }

    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda"] = torch.version.cuda
        report["gpu_count"] = int(torch.cuda.device_count())
        report["gpus"] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]
    except Exception as exc:
        report["torch_probe_error"] = f"{type(exc).__name__}: {exc}"

    return report


def main() -> int:
    print(json.dumps(build_report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
