# Lecture 03: LoRA + Adapter Patterns

## What LoRA does

Instead of fine-tuning all 7B params of a model, train a low-rank update:

```
W_new = W_base + (A · B)
```

where A is `[d × r]` and B is `[r × d]`, and r << d (typically 16-64).

For a 7B model with r=16, you train ~4M params instead of 7B — 1750× less.

## Why it matters operationally

1. **Cheap fine-tuning**: 1 GPU instead of 8; hours instead of days
2. **Multi-tenant serving**: one base model + many small adapters; hot-swap at inference time
3. **Cheap experimentation**: try 50 fine-tunes for the cost of 1 full fine-tune

## QLoRA

QLoRA = quantize the base model to 4-bit + LoRA on top. Pushes 13B fine-
tuning to a single 24GB GPU.

```python
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf",
                                              quantization_config=bnb, device_map="auto")
model = get_peft_model(model, LoraConfig(r=64, lora_alpha=16))
```

## Serving with hot-swap

vLLM serves base model + N LoRA adapters; clients pick by name:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-lora \
  --lora-modules sql=adapters/sql legal=adapters/legal medical=adapters/medical \
  --max-loras 4 --max-lora-rank 64
```

Requests with `"model": "sql"` use the SQL adapter; with `"model": "legal"`
use the legal adapter. Single base model fits in memory.

## Companion

[engineer-solutions/mod-110 ex-05 (finetuning-at-scale)](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-110-llm-infrastructure/exercise-05-finetuning-at-scale) — LoRA + QLoRA + DDP examples.
