import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen3-0.6B"   # example small causal LM
DATA_FILE = "autocomplete_train.jsonl"
OUTPUT_DIR = "autocomplete-lora"

use_4bit = True
