# Lecture 01: Quantization

## Why quantize

A 7B fp16 model = ~14 GB. At int8 = 7 GB. At int4 (AWQ/GPTQ) = ~4 GB. Smaller
weights → fits in cheaper GPUs + faster memory transfer.

## The numeric trade-off

- fp32: training default; full precision
- bf16: training + inference; same range as fp32, half precision
- fp16: inference; smaller range than bf16 (gradient underflow risk in training)
- int8: weight + activation quantization; modest accuracy hit
- int4 (AWQ, GPTQ): weights only; near-zero accuracy hit on large models
- int2 / int3: experimental; substantial accuracy loss

## Methods

| Method | Type | Hardware | Accuracy hit |
|---|---|---|---|
| Dynamic quantization | post-train, int8 | CPU | ~1pp |
| Static quantization | post-train w/ calibration | CPU/GPU | ~0.5pp |
| QAT (quant-aware training) | train-time | any | < 0.2pp |
| AWQ | activation-aware weight-only | GPU | < 0.2pp on large models |
| GPTQ | one-shot weight quantization | GPU | < 0.3pp on large models |

## Tooling

```bash
# AWQ
pip install autoawq
python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
model = AutoAWQForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
model.quantize(tok, quant_config={'zero_point': True, 'q_group_size': 128, 'w_bit': 4, 'version': 'GEMM'})
model.save_quantized('mistral-7b-awq')
"
```

vLLM loads AWQ models directly with `--quantization awq`.

## When NOT to quantize

- Small models (< 1B params): accuracy hit can be significant
- Specialty tasks (medical, legal, code): test thoroughly
- Anything with extreme precision needs (e.g., scientific computing)
