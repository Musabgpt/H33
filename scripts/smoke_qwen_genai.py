#!/usr/bin/env python3
from pathlib import Path
import os
import json

import onnxruntime_genai as og

MODEL_DIR = Path(os.environ.get("H33_MODEL_DIR", "app/src/main/assets/model"))
FABLE = Path("app/src/main/assets/fable_for_qwen_v1.txt")

if not (MODEL_DIR / "genai_config.json").exists():
    raise SystemExit("genai_config.json missing")

SYSTEM = (
    FABLE.read_text(encoding="utf-8").strip()
    if FABLE.exists()
    else "أنت مساعد عربي محلي. أجب مباشرة ولا تختلق معلومات."
)

print("Loading Qwen INT4 GenAI model...", flush=True)
model = og.Model(str(MODEL_DIR))
tokenizer = og.Tokenizer(model)

def run_messages(messages, extra_tokens=48):
    stream = tokenizer.create_stream()
    prompt = tokenizer.apply_chat_template(
        json.dumps(messages, ensure_ascii=False),
        add_generation_prompt=True,
    )
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

answer = run_messages([
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "ما ناتج 7 ضرب 8؟ أجب بالرقم فقط."},
], 32)
print("SMOKE ANSWER:", answer, flush=True)
if not answer:
    raise SystemExit("Qwen smoke test produced empty output")
if "56" not in answer:
    raise SystemExit("Qwen smoke test did not answer the arithmetic prompt as expected")

session_context = (
    SYSTEM
    + "\n\nSESSION_CONTEXT من نفس المحادثة. هذا السياق موثوق. "
      "إذا سأل المستخدم عن شيء ذُكر قبل قليل فاستخدمه مباشرة.\n"
      "المستخدم سابقًا: احفظ داخل هذه المحادثة فقط: الرمز التجريبي هو زمرد 4827.\n"
      "المساعد سابقًا: حسنًا، الرمز التجريبي في هذه المحادثة هو زمرد 4827."
)

memory_answer = run_messages([
    {"role": "system", "content": session_context},
    {"role": "user", "content": "ما الرمز التجريبي الذي ذكرته قبل قليل؟ أجب بالرمز فقط."},
], 48)
print("SESSION MEMORY ANSWER:", memory_answer, flush=True)
if "4827" not in memory_answer and "زمرد" not in memory_answer:
    raise SystemExit("Qwen explicit session-context smoke test failed")

print("QWEN_SMOKE_OK", flush=True)
print("SESSION_MEMORY_SMOKE_OK", flush=True)
