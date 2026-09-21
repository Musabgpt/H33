"""Validate llama-simple's stdout separately from runtime logs on stderr."""

PROMPT = "### Instruction:\nHi\n### Response:\n"


def generated_answer(stdout, returncode):
    if returncode:
        raise ValueError(f"Inference exited with status {returncode}")
    if PROMPT not in stdout:
        raise ValueError("Missing prompt boundary in llama-simple stdout")
    answer = stdout.split(PROMPT, 1)[1].strip()
    if not any(char.isalpha() for char in answer):
        raise ValueError("No generated text after the prompt")
    return answer


def main():
    import argparse
    import subprocess
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable")
    parser.add_argument("model")
    args = parser.parse_args()
    try:
        result = subprocess.run(
            [args.executable, "-m", args.model, "-ngl", "0", "-n", "8", PROMPT],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        sys.stderr.write(result.stderr)
        answer = generated_answer(result.stdout, result.returncode)
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        print(f"GGUF smoke test FAILED: {error}", file=sys.stderr)
        return 1
    print(answer)
    print("Host GGUF generation passed. Android/JNI execution remains unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
