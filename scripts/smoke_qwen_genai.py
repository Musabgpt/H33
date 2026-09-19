#!/usr/bin/env python3
from pathlib import Path
import sys

import onnxruntime_genai as og

MODEL_DIR = Path("app/src/main/assets/model")
if not (MODEL_DIR / "genai_config.json").exists():
    raise SystemExit("genai_config.json missing")

SYSTEM = (
    "أنت مساعد عربي محلي. أجب مباشرة وبوضوح وبقدر السؤال. "
    "إذا لم تعرف الإجابة فقل إنك لا تعرف، ولا تختلق معلومات."
)
QUESTION = "ما عاصمة فرنسا؟"
PROMPT = (
    "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
    "<|im_start|>user\n" + QUESTION + "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

print("Loading Qwen INT4 GenAI model...", flush=True)
model = og.Model(str(MODEL_DIR))
tokenizer = og.Tokenizer(model)
stream = tokenizer.create_stream()

input_tokens = tokenizer.encode(PROMPT)
params = og.GeneratorParams(model)
params.set_search_options(
    max_length=min(512, len(input_tokens) + 32),
    do_sample=False,
    repetition_penalty=1.05,
)

generator = og.Generator(model, params)
generator.append_tokens(input_tokens)

answer = ""
while not generator.is_done():
    generator.generate_next_token()
    token = generator.get_next_tokens()[0]
    answer += stream.decode(token)

answer = answer.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
print("SMOKE ANSWER:", answer, flush=True)

if not answer:
    raise SystemExit("Qwen smoke test produced empty output")

lower = answer.lower()
if "باريس" not in answer and "paris" not in lower:
    raise SystemExit("Qwen smoke test did not answer the factual Arabic prompt as expected")

print("QWEN_SMOKE_OK", flush=True)
