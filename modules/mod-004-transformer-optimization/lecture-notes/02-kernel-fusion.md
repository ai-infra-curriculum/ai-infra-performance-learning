# Lecture 02: Kernel Fusion

## Why fusion matters

A transformer block has ~30 ops. Each unfused op = a kernel launch + a global
memory round-trip. Fusion combines multiple ops into one kernel — fewer
launches, intermediate values stay in registers / shared memory.

## How to get fusion

| Approach | When |
|---|---|
| `torch.compile` | first try; fuses what it can automatically |
| `torch.jit.script` (older) | legacy systems |
| Triton | write fused kernel in Python |
| Hand-written CUDA | last resort; for hottest kernels |

## torch.compile

```python
model = torch.compile(model, mode="reduce-overhead")
```

`mode="reduce-overhead"` enables CUDA graphs + maximum fusion. Often 1.3-2×
speedup with one line.

## Triton example

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

Triton's strength: write GPU-fast kernels in Python; let the compiler handle
launch config + memory layout. Used inside vLLM, FlashAttention, etc.

## Common fusion opportunities

- Elementwise chains: `bias + GELU + dropout` → one kernel
- LayerNorm + linear projection
- Attention forward (FlashAttention does this)
- Softmax + dropout
