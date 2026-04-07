"""
Inference script for autocomplete using a LoRA-adapted model.

Usage:
    python generate_autocomplete.py

Loads the LoRA adapter from MODEL_DIR and generates short continuations.
"""

import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# --------------- configuration ---------------
MODEL_DIR = "autocomplete-lora"
PROMPT = "Writers often revise the opening"

MAX_NEW_TOKENS = 16        # keep completions short for autocomplete
TEMPERATURE = 0.3          # low temperature → more predictable output
TOP_P = 0.9
REPETITION_PENALTY = 1.1
# -----------------------------------------------


def generate(prompt: str, **kwargs) -> str:
    """Return the model's completion for *prompt*."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", MAX_NEW_TOKENS),
            temperature=kwargs.get("temperature", TEMPERATURE),
            top_p=kwargs.get("top_p", TOP_P),
            repetition_penalty=kwargs.get("repetition_penalty", REPETITION_PENALTY),
            do_sample=True,
        )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return completion


def main():
    print(f"Prompt:     {PROMPT}")
    completion = generate(PROMPT)
    print(f"Completion: {completion}")
    print(f"Full text:  {PROMPT}{completion}")


if __name__ == "__main__":
    main()
