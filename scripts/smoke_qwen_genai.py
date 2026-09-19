#!/usr/bin/env python3
from pathlib import Path

import onnxruntime_genai as og

MODEL_DIR = Path("app/src/main/assets/model")
FABLE = Path("app/src/main/assets/fable_for_qwen_v1.txt")

if not (MODEL_DIR / "genai_config.json").exists():
    raise SystemExit("genai_config.json missing")

SYSTEM = (
    FABLE.read_text(encoding="utf-8").strip()
    if FABLE.exists()
    else "أنت مساعد عربي محلي. أجب مباشرة ولا تختلق معلومات."
)

def run_prompt(prompt: str, extra_tokens: int = 48) -> str:
    model = og.Model(str(MODEL_DIR))
    tokenizer = og.Tokenizer(model)
    stream = tokenizer.create_stream()
    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=min(1900, len(input_tokens) + extra_tokens),
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

    return answer.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

print("Loading Qwen INT4 GenAI model...", flush=True)

# Basic Arabic factual test.
prompt = (
    "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
    "<|im_start|>user\nما ناتج 7 ضرب 8؟ أجب بالرقم فقط.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
answer = run_prompt(prompt, 32)
print("SMOKE ANSWER:", answer, flush=True)
if not answer:
    raise SystemExit("Qwen smoke test produced empty output")
if "56" not in answer:
    raise SystemExit("Qwen smoke test did not answer the arithmetic prompt as expected")

# Multi-turn session-memory test.
memory_prompt = (
    "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
    "<|im_start|>user\nاحفظ داخل هذه المحادثة فقط: الرمز التجريبي هو زمرد 4827.<|im_end|>\n"
    "<|im_start|>assistant\nحسنًا، سأحتفظ به ضمن سياق هذه المحادثة.<|im_end|>\n"
    "<|im_start|>user\nما الرمز التجريبي الذي ذكرته قبل قليل؟<|im_end|>\n"
    "<|im_start|>assistant\n"
)
memory_answer = run_prompt(memory_prompt, 40)
print("SESSION MEMORY ANSWER:", memory_answer, flush=True)
if "4827" not in memory_answer and "زمرد" not in memory_answer:
    raise SystemExit("Qwen multi-turn session-memory smoke test failed")

print("QWEN_SMOKE_OK", flush=True)
print("SESSION_MEMORY_SMOKE_OK", flush=True)
