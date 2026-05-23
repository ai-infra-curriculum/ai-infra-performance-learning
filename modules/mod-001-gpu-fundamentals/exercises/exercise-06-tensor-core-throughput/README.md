# Exercise 6 — Tensor-core throughput prediction

> **Targets learning objectives:** 2, 4, 6
> **Time:** ~60 min
> **Requires:** Python 3.10+. No GPU needed.
> **References:** lecture-notes/05-roofline-model.md (roofline, ridge), lecture-notes/06-gpu-generations.md (A100 / H100 specs).

## What you'll do

Given a GEMM (matrix multiply) size and a target GPU (A100 or
H100), predict on the FP16 tensor-core path:

1. The **arithmetic intensity** of the GEMM.
2. Whether the GEMM is **memory-bound or compute-bound** on that
   GPU's tensor cores.
3. The **realistic attainable TFLOPS**, given the roofline.

This pulls together every concept from the module: peak compute,
peak bandwidth, ridge point, arithmetic intensity, roofline.

You will use **only the verified NVIDIA datasheet values** from
Lecture 1.6. No new numbers, no estimates. The point is to see
which decisions follow purely from the published specs.

## Why this skill matters

"Should we use A100 or H100 for this workload?" is a six-figure
question for any real ML team. The answer hinges on whether your
dominant GEMMs are compute-bound (where H100 dramatically wins)
or memory-bound (where H100 helps less). The roofline analysis
is exactly this prediction.

You'll also see this pattern when:

- Sizing batches. Bigger batches = bigger GEMMs = higher AI =
  more likely to be on the compute side of the ridge. The
  roofline tells you the smallest batch that still saturates the
  tensor cores.
- Picking model architectures. A model with mostly large GEMMs
  (a wide transformer with big d_model) is compute-bound.
  A model with mostly small GEMMs (lots of layernorm, small
  attention heads) is memory-bound.
- Choosing precision. FP16 tensor-core peak is much higher than
  FP32. Moving from FP32 to FP16 raises the *compute* ceiling
  but doesn't change the *memory* ceiling — so it only helps
  for workloads that were close to or above the FP32 ridge.

## The model

### Step 1: Compute the GEMM's AI

For an `M x K` times `K x N` GEMM at precision with `bytes_per_elem`
bytes per element (e.g., 4 for FP32, 2 for FP16):

```
total_flops = 2 * M * N * K          # 2 FLOPS per MAC, M*N*K MACs
total_bytes = bytes_per_elem * (M*K + K*N + M*N)
AI          = total_flops / total_bytes
```

For square M=N=K with FP16 (bytes_per_elem = 2):

```
AI = (2 * K^3) / (2 * 3 * K^2)
   = K / 3
```

So a 4096^3 FP16 GEMM has AI = 4096/3 ≈ 1365 FLOP/byte. (Same as
FP32 4096^3, because the per-element size is half but the FLOPS
are also unchanged.)

Wait — that's not right. Let me redo:

```
FP32 4096^3: total_flops = 2 * 4096^3, total_bytes = 4 * 3 * 4096^2
           AI = 2 * 4096 / (4 * 3) = 8192 / 12 ≈ 683

FP16 4096^3: total_flops = 2 * 4096^3, total_bytes = 2 * 3 * 4096^2
           AI = 2 * 4096 / (2 * 3) = 8192 / 6 ≈ 1365
```

FP16 GEMM has *higher* AI than FP32 GEMM at the same matrix
shape — because the bytes get smaller (2 vs 4) while the FLOPS
stay the same. The compute-to-bandwidth ratio of the work
itself shifts.

### Step 2: Compare to the tensor-core ridge

The tensor-core peak FLOPS is *much* higher than FP32 CUDA-core
peak. From Lecture 1.6:

| GPU | FP32 peak | FP16 tensor peak (dense) |
|---|---|---|
| A100 SXM4 80GB | 19.5 TFLOPS | 312 TFLOPS |
| H100 SXM5 80GB | 67 TFLOPS | 989 TFLOPS |

The ridge point on the tensor-core path:

```
A100 FP16 TC ridge = 312_000 GFLOPS / 2039 GB/s ≈ 153 FLOP/byte
H100 FP16 TC ridge = 989_000 GFLOPS / 3350 GB/s ≈ 295 FLOP/byte
```

These are *very* high. Many GEMMs that look compute-bound on the
FP32 path become *memory-bound* on the tensor-core path, because
the tensor cores can devour FLOPS so fast that HBM can't feed
them.

### Step 3: Predict attainable TFLOPS

```
peak_flops_gflops = peak_tc_tflops * 1000
memory_ceiling_gflops = peak_bw_gbs * ai
attainable_gflops = min(peak_flops_gflops, memory_ceiling_gflops)
attainable_tflops = attainable_gflops / 1000
```

Apply a "realistic utilization" factor of 0.85 (typical for
well-tuned cuBLAS / cuDNN tensor-core GEMMs):

```
realistic_tflops = attainable_tflops * 0.85
```

The 0.85 factor accounts for end-of-tile waste, instruction
overhead, and warp scheduling — it's not perfect 100% peak even
on compute-bound kernels in practice.

### Worked example: 4096^3 FP16 GEMM on A100 vs H100

```
AI = 4096 / 3 = 1365.33 FLOP/byte

A100:
  AI=1365, ridge=153. AI >> ridge -> compute-bound.
  attainable = min(312000, 2039 * 1365) = min(312000, 2.78M)
             = 312000 GFLOPS = 312 TFLOPS
  realistic = 312 * 0.85 ≈ 265 TFLOPS

H100:
  AI=1365, ridge=295. AI >> ridge -> compute-bound.
  attainable = min(989000, 3350 * 1365) = min(989000, 4.57M)
             = 989000 GFLOPS = 989 TFLOPS
  realistic = 989 * 0.85 ≈ 841 TFLOPS

H100 / A100 speedup = 841 / 265 ≈ 3.17x
```

A 3.17× speedup matches what real training benchmarks show for
large FP16 GEMMs on H100 vs A100. The model recovers it from
first principles.

### Worked example: 1024^3 FP16 GEMM (smaller)

```
AI = 1024 / 3 = 341 FLOP/byte

A100:
  AI=341, ridge=153. AI > ridge -> compute-bound (barely above).
  attainable = min(312000, 2039 * 341) = min(312000, 695K)
             = 312000 GFLOPS
  realistic = ~265 TFLOPS

H100:
  AI=341, ridge=295. AI > ridge -> compute-bound (very close to ridge).
  attainable = min(989000, 3350 * 341) = min(989000, 1.14M)
             = 989000 GFLOPS
  realistic = ~841 TFLOPS

H100 / A100 = 3.17x again.
```

Same shape because both GPUs are still compute-bound.

### Worked example: a hypothetical small-ish, low-K GEMM

`M = N = 4096, K = 512` (think: a thin attention projection). FP16.

```
AI = 2 * M*N*K / (2 * (M*K + K*N + M*N))
   = 4096*4096*512 / (4096*512 + 512*4096 + 4096*4096)
   = 4096*4096*512 / (4096^2 + 2*4096*512)
   = 4096*4096*512 / (4096 * (4096 + 1024))
   = 4096 * 512 / 5120
   ≈ 410 FLOP/byte
```

Still compute-bound on A100 (ridge 153) and H100 (ridge 295).
The K=512 doesn't drop the AI as much as you'd think for square
GEMMs — the relevant dimension here is the smaller of M*K, K*N,
M*N (the "shape" of bytes vs FLOPs).

### A memory-bound GEMM

`M = N = 32, K = 32`. Tiny GEMM. FP16.

```
AI = 2 * 32^3 / (2 * 3 * 32^2)
   = 32 / 3
   ≈ 10.7 FLOP/byte
```

Below both ridges. Memory-bound.

```
A100 attainable = min(312000, 2039 * 10.7) = 21,800 GFLOPS = 21.8 TFLOPS
H100 attainable = min(989000, 3350 * 10.7) = 35,800 GFLOPS = 35.8 TFLOPS
```

7% of A100 peak. The bandwidth gain from H100 (1.64×) translates
roughly 1:1 to the speedup, much smaller than the 3× compute
speedup. Small GEMMs are bandwidth-limited on tensor cores.

## What to submit

Edit `starter.py`. Implement three functions:

```python
def gemm_arithmetic_intensity(M: int, N: int, K: int, bytes_per_elem: int) -> float: ...
def classify_gemm(M, N, K, bytes_per_elem, gpu: str) -> str: ...
    # returns "memory-bound" or "compute-bound"
def predict_tflops(M, N, K, bytes_per_elem, gpu: str, util: float = 0.85) -> float: ...
```

The `gpu` argument must be `"A100"` or `"H100"`. The starter
provides the spec constants.

Then run:

```bash
python check.py
```

The autograder runs five cases that exercise compute-bound and
memory-bound GEMMs on both GPUs.

## Hints

### Implementation skeleton

```python
A100_TC_FP16_TFLOPS = 312.0
A100_BW_GBS = 2039.0
H100_TC_FP16_TFLOPS = 989.0
H100_BW_GBS = 3350.0

def _spec(gpu: str) -> tuple[float, float]:
    if gpu == "A100":
        return A100_TC_FP16_TFLOPS, A100_BW_GBS
    if gpu == "H100":
        return H100_TC_FP16_TFLOPS, H100_BW_GBS
    raise ValueError(f"unknown GPU: {gpu!r} (use 'A100' or 'H100')")


def gemm_arithmetic_intensity(M, N, K, bytes_per_elem):
    flops = 2 * M * N * K
    bytes_moved = bytes_per_elem * (M*K + K*N + M*N)
    return flops / bytes_moved


def classify_gemm(M, N, K, bytes_per_elem, gpu):
    peak_tflops, peak_bw = _spec(gpu)
    ai = gemm_arithmetic_intensity(M, N, K, bytes_per_elem)
    ridge = (peak_tflops * 1000) / peak_bw
    return "memory-bound" if ai < ridge else "compute-bound"


def predict_tflops(M, N, K, bytes_per_elem, gpu, util=0.85):
    peak_tflops, peak_bw = _spec(gpu)
    ai = gemm_arithmetic_intensity(M, N, K, bytes_per_elem)
    compute_ceiling_gflops = peak_tflops * 1000
    memory_ceiling_gflops = peak_bw * ai
    attainable = min(compute_ceiling_gflops, memory_ceiling_gflops)
    return util * attainable / 1000  # TFLOPS
```

### Spec values you must use

These are the values quoted in Lecture 1.6 (verified against
NVIDIA datasheets). Do NOT use other sources or make up numbers.

```
A100 SXM4 80GB:
  - FP16 tensor-core peak (dense): 312 TFLOPS
  - HBM2e bandwidth: 2039 GB/s

H100 SXM5 80GB:
  - FP16 tensor-core peak (dense): 989 TFLOPS
  - HBM3 bandwidth: 3350 GB/s
```

### Why FP16 specifically

The autograder uses FP16 because:

- It's the most common training precision for AI workloads.
- The tensor-core peak is well-documented for both GPUs.
- BF16 tensor peak is the same as FP16 on A100/H100.
- FP8 would be H100-only, complicating the comparison.

A real model might use a mix of precisions, but for the
exercise FP16 is the cleanest choice.

### The 0.85 utilization factor

It's a "what cuBLAS typically achieves" number. Real-world
peak tensor-core utilization on large compute-bound GEMMs:

- cuBLAS on A100: ~85–90% of peak.
- cuBLAS on H100: ~80–88% of peak (slightly lower because
  Transformer Engine overhead).

The autograder accepts answers within ±10% of the 0.85-utilization
prediction.

### What to do for very small GEMMs

For M=N=K=32 (the smallest case the autograder tests):

```
AI = 2*32^3 / (2 * 3 * 32^2) = 32/3 ≈ 10.67
A100 ridge = 312000/2039 ≈ 153 -> memory-bound
A100 attainable = 2039 * 10.67 = 21,754 GFLOPS = 21.75 TFLOPS
A100 realistic = 21.75 * 0.85 ≈ 18.5 TFLOPS

H100 ridge = 989000/3350 ≈ 295 -> memory-bound
H100 attainable = 3350 * 10.67 = 35,745 GFLOPS = 35.75 TFLOPS
H100 realistic = 35.75 * 0.85 ≈ 30.4 TFLOPS
```

In practice cuBLAS doesn't even achieve 85% on very small GEMMs
(kernel launch overhead, tile-edge effects), but the model
ignores those — they're sub-Module-1 concerns. The autograder
tolerates 10% deviation.

## Common pitfalls

### 1. Mixing FP32 and FP16 byte counts

Make sure to pass `bytes_per_elem=2` for FP16, not 4. FP32 GEMM
would also be on the FP32 path (CUDA cores), not tensor cores —
the autograder doesn't test FP32, but if you were to extend the
model you'd need to use the FP32 peak (19.5 / 67 TFLOPS) and
4-byte elements.

### 2. Using "with sparsity" tensor TFLOPS

NVIDIA's FP16 tensor-core peak *with sparsity* is 624 TFLOPS for
A100 (2× the dense 312) and 1979 TFLOPS for H100. Use the **dense**
number. Workloads without 2:4 structured sparsity should plan
around dense throughput.

### 3. Forgetting to multiply by util

The model predicts the *attainable* throughput at the roofline.
Real cuBLAS achieves ~85% of that. The autograder expects you
to apply the 0.85 factor.

### 4. Off by 1000 (TFLOPS vs GFLOPS)

The `peak_tflops * 1000` conversion is what lets you compare to
`peak_bw_gbs * ai` (which is in GFLOPS). Forget it and your
ridge will be off by 1000x.

### 5. Treating non-square GEMMs as if they were square

For non-square M, N, K, you must compute the bytes-moved with all
three pairings:

```
bytes = bytes_per_elem * (M*K + K*N + M*N)
```

Don't shortcut with `3 * K^2` unless M = N = K.

## Grading rubric

The autograder runs 5 cases:

| Case | M | N | K | bytes_per_elem | GPU | Expected class | Expected TFLOPS (±10%) |
|---|---|---|---|---|---|---|---|
| Large FP16 on A100 (compute-bound) | 4096 | 4096 | 4096 | 2 | A100 | compute-bound | ~265 |
| Large FP16 on H100 (compute-bound) | 4096 | 4096 | 4096 | 2 | H100 | compute-bound | ~841 |
| Tiny FP16 on A100 (memory-bound) | 32 | 32 | 32 | 2 | A100 | memory-bound | ~18.5 |
| Tiny FP16 on H100 (memory-bound) | 32 | 32 | 32 | 2 | H100 | memory-bound | ~30.4 |
| Non-square (4096, 4096, 512) FP16 on H100 | 4096 | 4096 | 512 | 2 | H100 | compute-bound | ~841 |

Plus:
- `gemm_arithmetic_intensity(4096, 4096, 4096, 2)` ≈ 682.7
  (within 1%).
- Invalid GPU name raises `ValueError`.

If all checks pass, prints `PASS`.

## What "right" looks like

After this exercise you should be able to:

- Compute AI of a GEMM in 30 seconds given (M, N, K, bytes).
- Compare to the relevant ridge point.
- Predict A100 vs H100 speedup ratio without benchmarking.
- Explain why memory-bound kernels see less speedup on H100 than
  compute-bound ones.

If you can do that fluently, you can also answer the
"A100 vs H100 for our workload?" question in front of an
exec sponsor.

## Related material

- Lecture 1.5 — Roofline model (the model behind this exercise).
- Lecture 1.6 — A100 / H100 specs (the numbers you used).
- NVIDIA cuBLAS performance documentation — real-world tensor-
  core utilization data.
- Williams et al. 2009 — original roofline paper.

## Extended case studies

### Case A: Llama-class attention head GEMM

The dominant kernel in transformer attention is the (Q × K^T) and
(softmax × V) matmuls. For a typical Llama-class layer with hidden
dim 4096, head dim 128, sequence length 2048:

```
Q × K^T: shape (seq, head_d) x (head_d, seq) -> (seq, seq)
        M=2048, N=2048, K=128, FP16

AI = 2 * M*N*K / (2 * (M*K + K*N + M*N))
   = 2 * 2048 * 2048 * 128 / (2 * (2048*128 + 128*2048 + 2048*2048))
   = 2 * 2048 * 128 / (128 + 128 + 2048)   (cancel one 2048 from num&denom)
   = 524288 / 2304
   ≈ 227.6 FLOP/byte
```

A100 ridge for FP16 TC: 153. 227 > 153 → **compute-bound on A100**.
H100 ridge: 295. 227 < 295 → **memory-bound on H100**.

Same kernel, different ceilings:

```
A100: attainable = 312 TFLOPS (compute-bound)
      realistic = 312 * 0.85 ≈ 265 TFLOPS

H100: attainable = 3350 * 227 / 1000 ≈ 761 TFLOPS (memory-bound)
      realistic = 761 * 0.85 ≈ 647 TFLOPS

H100 / A100 speedup ≈ 2.44x
```

Less than the headline 3.17× compute ratio, because H100's ridge
moved up faster than its bandwidth. Memory-bound attention kernels
get a smaller H100 win than fully compute-bound large GEMMs.

This is one of the reasons FlashAttention matters so much: by
keeping the (M, N) = (seq, seq) intermediate matrix out of HBM,
it raises the *effective* AI of the kernel and pushes it more
firmly into the compute-bound regime, which is where H100's
compute advantage actually pays off.

### Case B: tiny attention head (long context)

For very long context (sequence 32768), the (Q × K^T) becomes a
huge matrix:

```
M=32768, N=32768, K=128, FP16

AI = 2 * 32768 * 128 / (128 + 128 + 32768)
   = 8.39e6 / 33024
   ≈ 254 FLOP/byte
```

Same AI shape as before — it's dominated by `2K / (something with
seq^2 / seq^2 = 1)`. The AI doesn't grow with sequence length for
Q × K^T; the *bytes* grow quadratically alongside the FLOPs.

What does grow quadratically: the *absolute* memory pressure.
A 32768^2 attention matrix in FP16 is 2 GB. That blows past
L2 cache and competes for HBM bandwidth with everything else
in the model.

So long context isn't an "AI" problem; it's a *capacity* problem.
The roofline analysis stays the same; the optimization moves to
chunking and recomputation (FlashAttention v2, ring attention,
etc.).

### Case C: comparing A100 vs L40S for inference

L40S has 91.6 FP32 TFLOPS, but its FP16 tensor-core peak is 366
TFLOPS (less than H100's 989, more than A100's 312). Memory
bandwidth is 864 GB/s (less than A100's 2039). FP16 ridge:

```
L40S FP16 ridge = 366 * 1000 / 864 ≈ 423 FLOP/byte
```

Compare to A100 (153) and H100 (295). L40S has the *highest*
ridge — most compute-rich relative to bandwidth.

For our 4096^3 FP16 GEMM (AI = 1365):
- L40S: AI > ridge, compute-bound. attainable = 366 TFLOPS.
  Realistic = 311 TFLOPS at 85% util.
- A100: AI > ridge, compute-bound. attainable = 312 TFLOPS.
  Realistic = 265 TFLOPS.
- Ratio L40S/A100 = 1.17x. Marginal on this workload.

For the attention Q × K^T (AI = 227):
- L40S: AI < ridge (227 < 423), memory-bound. attainable = 864 *
  227 / 1000 = 196 TFLOPS. Realistic = 167 TFLOPS.
- A100: AI > ridge (227 > 153), compute-bound. attainable = 312
  TFLOPS. Realistic = 265 TFLOPS.
- Ratio L40S/A100 = 0.63x. **L40S is slower on this kernel.**

So which is "faster" depends entirely on the workload. The
roofline analysis is what tells you so. For large compute-bound
GEMM workloads L40S is competitive with A100; for attention-heavy
workloads at typical context lengths, A100's bandwidth still
wins.

This is why benchmarks comparing "GPUs" are misleading without
specifying the workload. The roofline is the analytical answer
to "which GPU should I buy."

### Beyond Module 1: the model's blind spots

The first-order model in this exercise doesn't handle:

- **Sparsity.** Real cuBLAS achieves more than the dense peak
  when the data has structured sparsity. NVIDIA's 2:4 sparsity
  doubles throughput.
- **Mixed-precision GEMM.** Some kernels read FP16 weights, dequantize
  to FP32, and accumulate in FP32. The bytes-per-element shift
  during the kernel; the model assumes one uniform precision.
- **Quantization (INT8, FP8, FP4).** Each has its own tensor-core
  throughput. The arithmetic is the same shape; you just plug in
  different peak TFLOPS.
- **Power throttling.** A 700W H100 in a thermally constrained rack
  may run at 80% of its boost clock, dropping effective peak.
- **Multi-GPU scaling.** Adding GPUs raises peak proportionally,
  but inter-GPU bandwidth (NVLink/PCIe) becomes the new ceiling.

These all extend the same analytical framework. Modules 4 and 5
of the curriculum build them up explicitly. For Module 1 the
single-GPU FP16 roofline is enough.

## Module 1 wrap-up

This is the last exercise in Module 1. If it passes — along with
exercises 1–5 — you've internalized:

- SIMT, warps, divergence, shuffles.
- Memory hierarchy, coalescing, bank conflicts.
- Occupancy and the three resource caps.
- Arithmetic intensity, the roofline, ridge point.
- How to read an NVIDIA datasheet and predict throughput.

Module 2 picks up from these primitives and starts writing actual
CUDA kernels.
