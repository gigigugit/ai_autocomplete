"""
Convert a raw JSONL corpus into prompt/completion training pairs for autocomplete.

Usage:
    python make_autocomplete_dataset.py

Reads INPUT (one JSON object per line with a "text" field) and writes OUTPUT
with {"prompt": "...", "completion": "..."} pairs.
"""

import json
import random
from pathlib import Path

# --------------- configuration ---------------
INPUT = "input.jsonl"
OUTPUT = "autocomplete_train.jsonl"

MIN_WORDS = 6           # skip snippets shorter than this
SPLITS_PER_SNIPPET = 2  # how many prompt/completion pairs per snippet

# Prompt / completion length bounds (in whitespace-delimited words)
MIN_PROMPT_WORDS = 4
MAX_PROMPT_WORDS = 25
MIN_COMPLETION_WORDS = 2
MAX_COMPLETION_WORDS = 15
# -----------------------------------------------


def split_text(text: str):
    """Yield (prompt, completion) pairs from a single text snippet."""
    words = text.split()
    if len(words) < MIN_WORDS:
        return

    for _ in range(SPLITS_PER_SNIPPET):
        # Choose a split point that respects the length bounds
        lo = max(MIN_PROMPT_WORDS, 1)
        hi = min(MAX_PROMPT_WORDS, len(words) - MIN_COMPLETION_WORDS)
        if lo > hi:
            continue

        split_idx = random.randint(lo, hi)
        prompt = " ".join(words[:split_idx])
        completion = " " + " ".join(words[split_idx:])

        comp_words = len(words) - split_idx
        if comp_words > MAX_COMPLETION_WORDS:
            # Trim completion to MAX_COMPLETION_WORDS
            completion = " " + " ".join(words[split_idx : split_idx + MAX_COMPLETION_WORDS])

        yield prompt, completion


def main():
    input_path = Path(INPUT)
    if not input_path.exists():
        raise FileNotFoundError(
            f"{INPUT} not found. Place your raw JSONL corpus here first."
        )

    seen = set()
    count = 0

    with open(INPUT, encoding="utf-8") as fin, \
         open(OUTPUT, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if not text or text in seen:
                continue
            seen.add(text)

            for prompt, completion in split_text(text):
                row = {"prompt": prompt, "completion": completion}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"✓ Wrote {count} training examples to {OUTPUT}")


if __name__ == "__main__":
    main()
