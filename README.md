# AI Autocomplete – LoRA / QLoRA Fine-Tuning

Fine-tune a small causal language model (0.5 B–3 B parameters) for inline
text autocomplete using **LoRA** or **QLoRA**, powered by
[TRL](https://huggingface.co/docs/trl) +
[PEFT](https://huggingface.co/docs/peft).

---

## Repository layout

| File | Purpose |
|---|---|
| `make_autocomplete_dataset.py` | Convert a raw `input.jsonl` corpus into prompt/completion training pairs |
| `train_lora_autocomplete.py` | Fine-tune the base model with SFTTrainer + LoRA/QLoRA |
| `generate_autocomplete.py` | Run inference with the trained adapter |
| `requirements.txt` | Python dependencies |

---

## Quick-start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `bitsandbytes` is needed only for QLoRA (4-bit quantisation) and
> requires a supported NVIDIA GPU.

### 2. Prepare your data

Place your raw corpus in `input.jsonl` — one JSON object per line with a
`"text"` field:

```jsonl
{"text": "The project began as a small internal tool before expanding into a public platform."}
{"text": "Writers often revise the opening sentence several times before settling on the right tone."}
```

Then generate prompt/completion pairs:

```bash
python make_autocomplete_dataset.py
```

This creates `autocomplete_train.jsonl` with rows like:

```jsonl
{"prompt": "The project began as a small", "completion": " internal tool before expanding into a public platform."}
```

### 3. Fine-tune

```bash
python train_lora_autocomplete.py
```

By default this performs **QLoRA** (4-bit) fine-tuning of
`Qwen/Qwen3-0.6B`. Edit the constants at the top of the script to change
the base model, hyperparameters, or switch to standard LoRA
(`USE_4BIT = False`).

The trained LoRA adapter is saved to `autocomplete-lora/`.

### 4. Run inference

```bash
python generate_autocomplete.py
```

Edit `PROMPT` in the script (or integrate the `generate()` function into
your own application) to get short autocomplete suggestions.

---

## Choosing LoRA vs QLoRA

| | LoRA | QLoRA |
|---|---|---|
| VRAM required | Higher | Lower (4-bit base model) |
| Setup complexity | Simpler | Needs `bitsandbytes` |
| Best when | You have ≥ 16 GB VRAM | Consumer GPU / tight VRAM |

Toggle `USE_4BIT` in `train_lora_autocomplete.py`.

---

## Recommended hyperparameters

| Parameter | Starting value |
|---|---|
| Learning rate | `2e-4` |
| Epochs | 1–3 |
| LoRA rank (`r`) | 8–32 |
| LoRA alpha | 2× rank |
| Batch size × grad accum | 16+ effective |
| Max sequence length | 256 |

---

## How much data?

| Amount | Expectation |
|---|---|
| ~1 k examples | Enough to verify the pipeline works |
| 5 k–10 k | Noticeable style adaptation |
| 50 k–100 k+ | Strong autocomplete quality |

Quality matters more than quantity — aim for **consistent style** and
**clean text**.

---

## License

This project is provided as-is for educational and experimental use.
