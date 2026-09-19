#!/usr/bin/env python3
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

MODEL_ID = "aubmindlab/aragpt2-base"
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app" / "src" / "main" / "assets"
EXPORT_DIR = ROOT / ".onnx_export"
ASSETS.mkdir(parents=True, exist_ok=True)
shutil.rmtree(EXPORT_DIR, ignore_errors=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Exporting with Hugging Face Optimum ONNX:", MODEL_ID, flush=True)
cmd = [
    "optimum-cli", "export", "onnx",
    "--model", MODEL_ID,
    "--task", "text-generation",
    "--monolith",
    "--opset", "17",
    str(EXPORT_DIR),
]
subprocess.run(cmd, check=True)

onnx_files = sorted(EXPORT_DIR.glob("*.onnx"), key=lambda p: p.stat().st_size, reverse=True)
if not onnx_files:
    raise SystemExit("Optimum did not produce an ONNX model")
source = onnx_files[0]
print("Optimum model:", source, round(source.stat().st_size / 1024 / 1024, 1), "MB", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tokenizer.save_pretrained(ASSETS)

int8 = ASSETS / "aragpt2.int8.onnx"
if int8.exists():
    int8.unlink()

print("Quantizing to INT8:", int8, flush=True)
quantize_dynamic(
    source.as_posix(),
    int8.as_posix(),
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
    op_types_to_quantize=["MatMul", "Gemm"],
)

if not int8.exists() or int8.stat().st_size < 50 * 1024 * 1024:
    raise SystemExit("INT8 model was not produced correctly")
print("INT8 size MB:", round(int8.stat().st_size / 1024 / 1024, 1), flush=True)

print("Running ONNX smoke test...", flush=True)
sess = ort.InferenceSession(int8.as_posix(), providers=["CPUExecutionProvider"])
print("Inputs:", [(x.name, x.shape, x.type) for x in sess.get_inputs()], flush=True)
print("Outputs:", [(x.name, x.shape, x.type) for x in sess.get_outputs()], flush=True)

ids = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
mask = np.ones_like(ids)
pos = np.arange(ids.shape[1], dtype=np.int64)[None, :]
feed = {}
for inp in sess.get_inputs():
    name = inp.name
    if name == "input_ids":
        feed[name] = ids
    elif name == "attention_mask":
        feed[name] = mask
    elif name == "position_ids":
        feed[name] = pos
    elif name == "token_type_ids":
        feed[name] = np.zeros_like(ids)
    else:
        raise SystemExit(f"Unexpected required ONNX input: {name}")

outputs = sess.run(None, feed)
logits = outputs[0]
if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != ids.shape[1]:
    raise SystemExit(f"Unexpected logits shape: {logits.shape}")
if not np.isfinite(logits).all():
    raise SystemExit("Non-finite logits in ONNX smoke test")

print("Smoke test OK:", logits.shape, flush=True)
shutil.rmtree(EXPORT_DIR, ignore_errors=True)
