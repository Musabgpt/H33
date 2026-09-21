#!/usr/bin/env python3
"""Make the DeepSeek GPT-style regex compatible with ORT GenAI's Android regex engine."""
import json
import sys
from pathlib import Path

ASCII_GPT_PATTERN = (
    r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


def patch(value):
    changed = 0
    if isinstance(value, dict):
        for key, child in list(value.items()):
            if key == "Regex" and isinstance(child, str) and "\\p{" in child:
                value[key] = ASCII_GPT_PATTERN
                changed += 1
            else:
                replacement, count = patch(child)
                value[key] = replacement
                changed += count
    elif isinstance(value, list):
        for index, child in enumerate(value):
            value[index], count = patch(child)
            changed += count
    return value, changed


def main():
    tokenizer = Path(sys.argv[1])
    data = json.loads(tokenizer.read_text(encoding="utf-8"))
    data, changed = patch(data)
    if not changed:
        raise SystemExit("No incompatible Unicode regex was found; refusing a silent patch")
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if "\\\\p{" in encoded:
        raise SystemExit("Unsupported Unicode property regex remains")
    tokenizer.write_text(encoded, encoding="utf-8")
    print(f"Patched {changed} ORT-incompatible tokenizer regex pattern(s)")


if __name__ == "__main__":
    main()
