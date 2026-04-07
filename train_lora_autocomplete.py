"""
LoRA / QLoRA fine-tuning script for autocomplete.

Usage:
    python train_lora_autocomplete.py

Expects a JSONL file (DATA_FILE) with {"prompt": "...", "completion": "..."} rows.
Produces a LoRA adapter saved to OUTPUT_DIR.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# --------------- configuration ---------------
MODEL_ID = "Qwen/Qwen3-0.6B"       # small causal LM; swap for any HF model
DATA_FILE = "autocomplete_train.jsonl"
OUTPUT_DIR = "autocomplete-lora"

USE_4BIT = True                     # set False for standard LoRA (needs more VRAM)
LORA_R = 16                        # LoRA rank  (8–32 is a good range)
LORA_ALPHA = 32                    # LoRA alpha (commonly 2×r)
LORA_DROPOUT = 0.05

LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
PER_DEVICE_BATCH = 4
GRADIENT_ACCUM = 4
MAX_SEQ_LEN = 256
LOGGING_STEPS = 25
SAVE_STEPS = 200
# -----------------------------------------------


def formatting_func(example):
    """Concatenate prompt and completion into a single training string."""
    return example["prompt"] + example["completion"]


def main():
    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- quantisation (QLoRA) ---
    bnb_config = None
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # --- model ---
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # --- LoRA adapter ---
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # --- dataset ---
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # --- trainer ---
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRADIENT_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_seq_length=MAX_SEQ_LEN,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        formatting_func=formatting_func,
    )

    # --- train & save ---
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✓ LoRA adapter saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
