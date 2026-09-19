#!/usr/bin/env python3
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "aubmindlab/aragpt2-base"
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app" / "src" / "main" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

print("Loading", MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()

class LastTokenWithHidden(torch.nn.Module):
    """Return only next-token logits and the last hidden vector to keep Android RAM low."""
    def __init__(self, m):
        super().__init__()
        self.transformer = m.transformer
        self.lm_head = m.lm_head

    def forward(self, input_ids, attention_mask):
        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state[:, -1, :]
        logits = self.lm_head(hidden)
        return logits, hidden

wrapper = LastTokenWithHidden(model)
ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
mask = torch.ones_like(ids)
fp32 = ASSETS / "aragpt2.fp32.onnx"
print("Exporting", fp32)
with torch.no_grad():
    torch.onnx.export(
        wrapper,
        (ids, mask),
        fp32.as_posix(),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits", "hidden"],
        dynamic_axes={
            "input_ids": {1: "sequence"},
            "attention_mask": {1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

tokenizer.save_pretrained(ASSETS)

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8 = ASSETS / "aragpt2.int8.onnx"
    print("Quantizing", int8)
    quantize_dynamic(
        fp32.as_posix(),
        int8.as_posix(),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print("INT8 size MB:", round(int8.stat().st_size / 1024 / 1024, 1))
except Exception as e:
    print("INT8 quantization failed; FP32 will still be usable:", repr(e))

try:
    import numpy as np
    import onnxruntime as ort
    candidate = ASSETS / "aragpt2.int8.onnx"
    if not candidate.exists():
        candidate = fp32
    sess = ort.InferenceSession(candidate.as_posix(), providers=["CPUExecutionProvider"])
    out = sess.run(None, {
        "input_ids": ids.numpy().astype(np.int64),
        "attention_mask": mask.numpy().astype(np.int64),
    })
    assert out[0].shape == (1, model.config.vocab_size), out[0].shape
    assert out[1].shape == (1, model.config.n_embd), out[1].shape
    assert np.isfinite(out[0]).all() and np.isfinite(out[1]).all()
    print("Smoke test OK:", out[0].shape, out[1].shape)
except Exception as e:
    raise SystemExit("ONNX smoke test failed: " + repr(e))
