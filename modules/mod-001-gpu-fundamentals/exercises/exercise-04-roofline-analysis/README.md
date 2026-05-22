# Exercise 4 — Roofline analysis

> **Targets learning objectives:** 2, 3, 4
> **Time:** ~45 min
> **Requires:** Python 3.10+, matplotlib. No GPU needed.

## What you'll do

You will compute the arithmetic intensity of six concrete kernels, place
them on an A100 roofline, and classify each as **memory-bound** or
**compute-bound** without running them. Your script also plots the
roofline as a PNG so you have a visual artifact.

This is the diagnostic that, in practice, tells you where to look first
on any new kernel. "Compute-bound + 40% of peak" is a different
optimization problem from "memory-bound + 80% of peak bandwidth" — and
the roofline tells you which conversation you are in.

## The six kernels

You will work with these kernels. The numbers below are the FLOPS per
output element and bytes moved per output element — you will derive
each kernel's arithmetic intensity from these.

| Kernel | What it does | FLOPS/element | Bytes/element (FP32) |
|---|---|---|---|
| `vector_add` | `c[i] = a[i] + b[i]` | 1 | 12 (read 4+4, write 4) |
| `vector_fma` | `c[i] = a[i]*b[i] + d[i]` | 2 | 16 (read 4+4+4, write 4) |
| `axpy_scaled` | `y[i] = alpha*x[i] + y[i]` | 2 | 12 (read 4+4, write 4) |
| `relu` | `y[i] = max(0, x[i])` | 1 | 8 (read 4, write 4) |
| `gemm_8192` | dense FP32 GEMM, M=N=K=8192 | 2·M·N·K | 4·(M·K + K·N + M·N) |
| `softmax_row` | row-wise softmax over a (B=512, D=4096) FP32 tensor | ~5·B·D | 8·B·D (one read + one write) |

For each, compute the arithmetic intensity in FLOP/byte. For `gemm_8192`
and `softmax_row`, the per-element numbers do not factor cleanly — you
need the **total** FLOPS and **total** bytes for the kernel, then divide.

## The roofline

You will plot the A100 roofline:

- Peak FP32: 19.5 TFLOPS
- Peak HBM bandwidth: 2039 GB/s
- Ridge point: `AI_ridge = 19500 / 2039` ≈ 9.56 FLOP/byte

The two lines:

```
memory ceiling: throughput = bandwidth * AI       (a line through origin)
compute ceiling: throughput = peak_flops          (a horizontal line)
attainable:      min(memory_ceiling, compute_ceiling)
```

Then plot each of the six kernels at its arithmetic intensity on the
memory ceiling (assuming peak bandwidth — the *upper bound* of
achievable performance for that AI).

## What to submit

Edit `starter.py`. Implement:

```python
def arithmetic_intensity(kernel: str) -> float: ...
def classify(ai: float, peak_flops_tflops: float, peak_bw_gbs: float) -> str: ...
    # returns "memory-bound" or "compute-bound"
def attainable_throughput_tflops(ai: float, peak_flops_tflops: float, peak_bw_gbs: float) -> float: ...
def plot_roofline(out_path: str) -> None: ...
```

Then run:

```bash
python check.py
```

The autograder verifies the six AI values, the classifications, and
that `plot_roofline` produces a PNG with the right structure (two
ceilings + six markers).

## Hints

- Convert TFLOPS to GFLOPS (multiply by 1000) for the comparison with
  bandwidth-derived throughput in GB/s — units have to match.
- For `gemm_8192`: total FLOPS = 2 × 8192³ ≈ 1.1 × 10¹². Total bytes
  = 4 × (8192² + 8192² + 8192²) = 12 × 8192² ≈ 8.05 × 10⁸. AI ≈ 1364
  FLOP/byte. (This is deeply compute-bound on FP32 hardware.)
- For `softmax_row`: 5 FLOPS per element is an estimate (one exp, one
  add to running sum, one divide, plus the max trick). Bytes = 8
  per element (read + write). AI ≈ 0.625 FLOP/byte.
- Use `matplotlib` with `loglog` axes to make the roofline shape
  legible — both AI and throughput span several orders of magnitude.

## What "right" looks like

Your `roofline.png` should show vector_add, vector_fma, axpy_scaled,
relu, and softmax_row clustered well below the ridge point (deeply
memory-bound), and gemm_8192 well above (deeply compute-bound).
That visual gap is the punchline of the whole module: not all kernels
on the same GPU are bottlenecked by the same resource, and the
arithmetic intensity tells you which.
