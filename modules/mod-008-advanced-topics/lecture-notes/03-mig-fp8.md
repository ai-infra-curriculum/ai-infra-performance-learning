# Lecture 03: MIG + FP8

## MIG (Multi-Instance GPU)

A100/H100 GPUs can partition into 7 isolated instances. Each:
- Has its own memory + compute
- Is hardware-isolated from other instances
- Looks like an independent GPU to the OS

Use for:
- Multi-tenant dev/notebook clusters
- Mixing serving (need isolation) + batch on the same node
- Better utilization for many small models

Trade-offs: smaller per-instance memory; can't use full GPU for one workload.

## FP8

H100 + B100 support fp8 tensors via Transformer Engine. Two formats:
- E4M3: range [-448, 448], finer precision
- E5M2: range [-57344, 57344], wider range for gradients

Use cases:
- Training: 30-50% speedup over bf16 with minimal accuracy loss (use TE)
- Inference: 1.5-2× speedup over fp16/bf16

The catch: fp8 dynamic range is narrow; per-tensor scaling factors are required.
Transformer Engine + Megatron-LM handle this; rolling your own is risky.

## When FP8

- H100/B100 GPUs (older HW lacks the precision)
- Transformer architectures with TE
- Training acceleration on critical-path models

When NOT:
- Non-transformer architectures (no TE support)
- Tiny models (won't benefit much)
- Sensitive accuracy domains (medical, financial) without thorough testing
