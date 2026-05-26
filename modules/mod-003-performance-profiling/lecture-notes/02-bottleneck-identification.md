# Lecture 02: Bottleneck Identification

## The roofline model

Plot kernel performance against arithmetic intensity (FLOPS/byte). A kernel
hits one of two ceilings:
- **Memory-bound**: below the slope (bandwidth limit)
- **Compute-bound**: at the horizontal ceiling (peak FLOPS)

If you're memory-bound, more FLOPS won't help — optimize memory access.
If you're compute-bound, more bandwidth won't help — optimize the math
(tensor cores, fp16/bf16, better algorithm).

ncu reports arithmetic intensity directly.

## Common bottlenecks

| Symptom (nsys) | Diagnosis | Fix |
|---|---|---|
| Long CUDA API calls | sync between host + device | reduce `cudaMemcpy`; use streams |
| Small kernel, called many times | launch overhead dominates | batch larger; use CUDA graphs |
| One kernel takes 80%+ | true hotspot; ncu it | per-kernel optimization |
| Many tiny kernels in sequence | kernel fusion opportunity | torch.compile / custom kernel |
| Long memcpy time | host↔device transfer is the bottleneck | overlap with compute via streams |

## Speed-of-Light interpretation

ncu's "GPU Speed of Light" section reports:
- % of peak FLOPS
- % of peak memory bandwidth

If both are low → kernel isn't using the GPU well → typically too few warps
launched, or warp divergence.

## NVTX ranges

Decorate your code:

```python
with torch.cuda.nvtx.range("forward"):
    out = model(x)
with torch.cuda.nvtx.range("backward"):
    out.sum().backward()
```

Now nsys shows your high-level structure on the timeline. Critical for
finding which Python-level phase is the hotspot.
