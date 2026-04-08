# Model Resources — Candidate Models for Autocomplete

This document summarizes candidate base models for LoRA / QLoRA fine-tuning
in the autocomplete pipeline. All models are causal language models hosted on
Hugging Face and are compatible with the existing training script
(`train_lora_autocomplete.py`) — only the `MODEL_ID` constant needs to change.

## Selection criteria

| Criterion | Requirement |
|---|---|
| Parameter range | 0.5 B – 3 B (up to ~3.8 B with QLoRA) |
| Framework | HuggingFace Transformers + PEFT |
| Quantisation | Must support `bitsandbytes` 4-bit (QLoRA) |
| LoRA target | `"all-linear"` injection |
| Task type | `CAUSAL_LM` (autoregressive text completion) |
| Inference budget | ≤ 16 new tokens, 256 max sequence length |

## Candidate model summary

### Tier 1 — Best Fit (0.5 B – 1.5 B)

| Model | Params | HF Hub ID | Key strengths | Notes |
|---|---|---|---|---|
| Qwen3 0.6B | 0.6 B | `Qwen/Qwen3-0.6B` | Current default. Strong multilingual causal LM, good tokenizer, well-supported by PEFT. | Already integrated as baseline. |
| Qwen2.5 0.5B | 0.5 B | `Qwen/Qwen2.5-0.5B` | Proven stability, very fast inference. | Good for comparison with Qwen3. |
| Qwen2.5 1.5B | 1.5 B | `Qwen/Qwen2.5-1.5B` | Step up in quality with modest VRAM increase. | Fits QLoRA on 8 GB GPUs. |
| TinyLlama 1.1B | 1.1 B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Trained on 3 T tokens (unusually high for its size), very fast. | Good autocomplete baseline. |
| StableLM 2 1.6B | 1.6 B | `stabilityai/stablelm-2-1_6b` | Strong English text generation, compact and efficient. | — |

### Tier 2 — Higher Quality (up to 3 B, more VRAM)

| Model | Params | HF Hub ID | Key strengths | Notes |
|---|---|---|---|---|
| Qwen2.5 3B | 3 B | `Qwen/Qwen2.5-3B` | Top of target range; noticeably better generation quality. | QLoRA fits on 10–12 GB GPUs. |
| Phi-3.5 Mini | 3.8 B | `microsoft/phi-3.5-mini-instruct` | Exceptional quality; strong at structured completions. | Slightly above range but QLoRA brings VRAM to ~6–8 GB. |
| Phi-2 | 2.7 B | `microsoft/phi-2` | Exceptional quality-to-size ratio, strong at text completion. | Well-tested with PEFT; fits QLoRA on 8–12 GB GPUs. |
| Gemma 2 2B | 2.6 B | `google/gemma-2-2b` | Efficient architecture, good text continuation. | HF name is "2b" but actual param count is 2.6 B. |
| Llama 3.2 1B | 1.2 B | `meta-llama/Llama-3.2-1B` | Excellent tokenizer efficiency, strong general text quality. | Best quality-to-speed ratio at ~1 B scale. |
| Llama 3.2 3B | 3 B | `meta-llama/Llama-3.2-3B` | Best quality at 3 B from Meta. | Requires Llama licence acceptance on HF. |

### Tier 3 — Specialised / Niche Alternatives

| Model | Params | HF Hub ID | Key strengths | Notes |
|---|---|---|---|---|
| StarCoderBase 1B | 1 B | `bigcode/starcoderbase-1b` | Optimised for code completion. | Only suitable if autocomplete targets code. |
| Pythia 1B (deduped) | 1 B | `EleutherAI/pythia-1b-deduped` | Well-documented training data, good research baseline. | Useful for reproducibility studies. |
| OpenELM 1.1B | 1.1 B | `apple/OpenELM-1_1B` | Layer-wise scaling architecture. | Interesting efficiency trade-offs. |

## Top 3 recommendations

If benchmarking a shortlist, these three span the full parameter budget and
give a clear picture of the quality-vs-speed trade-off:

1. **Qwen/Qwen3-0.6B** — current baseline, keep for comparison
2. **meta-llama/Llama-3.2-1B** — best quality-to-speed ratio at ~1 B
3. **Qwen/Qwen2.5-3B** — best quality at the top of the target range

## Usage

Change the `MODEL_ID` constant in `train_lora_autocomplete.py`:

```python
MODEL_ID = "meta-llama/Llama-3.2-1B"   # swap for any model above
```

No other code changes are required.
