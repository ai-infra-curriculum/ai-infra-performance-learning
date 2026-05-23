# Exercise 4 — Roofline analysis

> **Targets learning objectives:** 2, 3, 4
> **Time:** ~45–60 min
> **Requires:** Python 3.10+, matplotlib. No GPU needed.
> **References:** lecture-notes/05-roofline-model.md (the model, ridge point, worked examples).

## What you'll do

You will compute the arithmetic intensity of six concrete kernels,
place them on an A100 roofline, and classify each as
**memory-bound** or **compute-bound** without running them. Your
script also plots the roofline as a PNG so you have a visual
artifact.

This is the diagnostic that, in practice, tells you where to look
first on any new kernel. "Compute-bound + 40% of peak" is a
different optimization problem from "memory-bound + 80% of peak
bandwidth" — and the roofline tells you which conversation you're
in.

## Why this skill matters

The roofline is the single most-used diagnostic in GPU performance
engineering. Every team that takes performance seriously has
something like it pinned up on a wall. Here's why:

- **Avoids wasted optimization.** "I tried to make the FMA inner
  loop faster on my vector_add kernel and it didn't help." Of
  course not — vector_add is memory-bound; the FMAs already had
  plenty of slack. The roofline tells you that *before* you
  start optimizing.
- **Communicates with non-experts.** A roofline plot with a
  kernel marker below the memory ceiling tells a manager
  "we're bandwidth-bound; throwing more compute at it won't
  help" in one image.
- **Catches measurement errors.** A kernel reporting 5 TFLOPS at
  AI=0.1 on an A100 (where the memory ceiling at AI=0.1 is
  204 GFLOPS) is reporting something wrong. The roofline is
  also a sanity check.
- **Decides which GPU to use.** A workload close to the ridge on
  A100 might be far above the ridge on a 4090 (different ratios).
  Roofline analysis on candidate GPUs is the framework for that
  decision.

After this exercise, you should be able to look at any kernel
description ("I'm doing fused matmul + bias + gelu on FP16
tensors of shape (4096, 1024)") and predict, in 30 seconds,
roughly which side of the ridge it's on and what the peak
attainable throughput is.

## The six kernels

You will work with these kernels. The numbers below are the FLOPS
per output element and bytes moved per output element — you will
derive each kernel's arithmetic intensity from these.

| Kernel | What it does | FLOPS/element | Bytes/element (FP32) |
|---|---|---|---|
| `vector_add` | `c[i] = a[i] + b[i]` | 1 | 12 (read 4+4, write 4) |
| `vector_fma` | `c[i] = a[i]*b[i] + d[i]` | 2 | 16 (read 4+4+4, write 4) |
| `axpy_scaled` | `y[i] = alpha*x[i] + y[i]` | 2 | 12 (read 4+4, write 4) |
| `relu` | `y[i] = max(0, x[i])` | 1 | 8 (read 4, write 4) |
| `gemm_8192` | dense FP32 GEMM, M=N=K=8192 | 2·M·N·K | 4·(M·K + K·N + M·N) |
| `softmax_row` | row-wise softmax over a (B=512, D=4096) FP32 tensor | ~5·B·D | 8·B·D (one read + one write) |

For each, compute the arithmetic intensity in FLOP/byte. For
`gemm_8192` and `softmax_row`, the per-element numbers do not
factor cleanly — you need the **total** FLOPS and **total** bytes
for the kernel, then divide.

### Worked AI for each kernel

| Kernel | FLOPs | Bytes | AI = FLOP/byte |
|---|---|---|---|
| `vector_add` | 1 | 12 | 0.0833 |
| `vector_fma` | 2 | 16 | 0.125 |
| `axpy_scaled` | 2 | 12 | 0.1667 |
| `relu` | 1 | 8 | 0.125 |
| `gemm_8192` (total) | 2 * 8192^3 | 12 * 8192^2 | 8192*2/12 ≈ 1365.3 |
| `softmax_row` | 5 | 8 | 0.625 |

(`gemm_8192` simplifies: `2 * M * N * K / (4 * (M*K + K*N + M*N))`.
For square M=N=K, that's `2K / 12 = K/6`. K=8192 gives ~1365.3.)

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

Then plot each of the six kernels at its arithmetic intensity on
the memory ceiling (assuming peak bandwidth — the *upper bound*
of achievable performance for that AI).

### Reading the plot

On log-log axes:

- The **memory ceiling** is a straight line going up and to the
  right, slope 1.
- The **compute ceiling** is a horizontal line.
- The **ridge point** is where they meet.
- Each kernel appears as a marker. Five of the six (vector_add,
  vector_fma, axpy_scaled, relu, softmax_row) cluster well below
  the ridge — they're all memory-bound. `gemm_8192` sits at the
  far right, hitting the compute ceiling.

The visual gap between the two clusters is the punchline of the
whole module: not all kernels on the same GPU are bottlenecked by
the same resource. The arithmetic intensity tells you which.

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

The autograder verifies the six AI values, the classifications,
and that `plot_roofline` produces a PNG with the right structure
(two ceilings + six markers).

## Hints

### Implementation strategy

For `arithmetic_intensity`, use a dispatch dict:

```python
def arithmetic_intensity(kernel: str) -> float:
    if kernel == "vector_add":
        return 1 / 12
    elif kernel == "vector_fma":
        return 2 / 16
    elif kernel == "axpy_scaled":
        return 2 / 12
    elif kernel == "relu":
        return 1 / 8
    elif kernel == "gemm_8192":
        M = N = K = 8192
        flops = 2 * M * N * K
        bytes_moved = 4 * (M*K + K*N + M*N)
        return flops / bytes_moved
    elif kernel == "softmax_row":
        return 5 / 8
    raise ValueError(f"unknown kernel: {kernel}")
```

For `classify`:

```python
def classify(ai, peak_flops_tflops=..., peak_bw_gbs=...):
    # Convert peak_flops_tflops to GFLOPS so units match peak_bw_gbs.
    peak_flops_gflops = peak_flops_tflops * 1000
    ridge_ai = peak_flops_gflops / peak_bw_gbs
    if ai < ridge_ai:
        return "memory-bound"
    return "compute-bound"
```

For `attainable_throughput_tflops`:

```python
def attainable_throughput_tflops(ai, peak_flops_tflops, peak_bw_gbs):
    memory_ceiling_gflops = peak_bw_gbs * ai
    compute_ceiling_gflops = peak_flops_tflops * 1000
    return min(memory_ceiling_gflops, compute_ceiling_gflops) / 1000
```

### For the plot

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_roofline(out_path):
    ai_range = np.logspace(-2, 4, 200)
    mem_ceiling = ai_range * A100_PEAK_BW_GBS / 1000  # in TFLOPS
    cmp_ceiling = np.full_like(ai_range, A100_PEAK_FP32_TFLOPS)
    attain = np.minimum(mem_ceiling, cmp_ceiling)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(ai_range, mem_ceiling, '--', label='memory ceiling')
    ax.loglog(ai_range, cmp_ceiling, '--', label='compute ceiling')
    ax.loglog(ai_range, attain, '-', label='attainable')

    for k in KERNEL_NAMES:
        ai = arithmetic_intensity(k)
        tput = attainable_throughput_tflops(ai, A100_PEAK_FP32_TFLOPS, A100_PEAK_BW_GBS)
        ax.scatter([ai], [tput], label=k, zorder=5)

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Throughput (TFLOPS)")
    ax.set_title("A100 Roofline")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
```

You don't need to copy this exactly; the autograder only checks
that the PNG exists and is non-trivial in size.

### Unit conversions

This is where bugs hide. Be consistent:

- `peak_flops_tflops` (TFLOPS) and `peak_bw_gbs` (GB/s) are in
  different units. Don't compare them directly.
- Convert TFLOPS to GFLOPS (multiply by 1000) or GB/s to TB/s
  (divide by 1000) — pick one and stick with it.
- AI is in FLOP/byte (dimensionless ratio).
- `peak_bw_gbs * ai` has units of `GB/s * FLOP/byte = GFLOPS`.
- For `attainable_throughput_tflops` you want TFLOPS, so divide
  GFLOPS by 1000.

### For the GEMM AI

You can derive it without plugging in 8192:

```
AI(GEMM, square M=N=K) = 2*K^3 / (4 * 3 * K^2)
                       = 2*K / 12
                       = K / 6
```

So GEMM AI scales linearly with K. For K=4096: 683. For K=8192:
1365. For K=16384: 2731. The AI grows linearly with K — bigger
GEMMs are more compute-bound.

### For `softmax_row`

Per element: 5 FLOPS is an estimate covering:
- One max-tracking comparison per pass (~1 FLOP).
- One subtract (`x - max`) per pass (~1 FLOP).
- One exp (~1 FLOP, but actually quite a few internally).
- One add to running sum (~1 FLOP).
- One divide (`exp / sum`, ~1 FLOP).

The autograder tolerates up to 10% deviation from this estimate.

Bytes per element: 4 (read) + 4 (write) = 8. The actual
implementation reads each element ~3 times (max pass, sum pass,
divide pass) but those re-reads come from cache, so they don't
count against HBM bandwidth.

## Common pitfalls

### 1. Mixing TFLOPS and GFLOPS

```
WRONG: ridge_ai = 19.5 / 2039             # = 0.0096, way off
RIGHT: ridge_ai = 19500 / 2039 ≈ 9.56     # GFLOPS / GB/s = FLOP/byte
```

### 2. Forgetting log scale

A linear roofline plot crushes everything below AI=10 into a
sliver near the y-axis. Use `loglog` to see structure across 4+
orders of magnitude.

### 3. Plotting AI as a "value" rather than a coordinate

Each kernel marker should be at `(AI, attainable_throughput(AI))`.
Don't plot all markers at the same y-value — they should land
*on* the attainable curve at their respective AIs.

### 4. Wrong AI for GEMM

The total-FLOPS / total-bytes formulation matters. The
per-element AI doesn't make sense for GEMM because each output
element does K work but the bytes don't divide cleanly.

### 5. PNG file not large enough

If the autograder complains about a "tiny file (<5000 bytes)",
your plot is probably empty. Check that you're calling
`plt.savefig` after adding artists, and `plt.close(fig)` after
saving.

## Grading rubric

The autograder checks:

1. **AI for each of 6 kernels** within 10% of the reference value.
   Generous tolerance because softmax FLOPS-per-element is an
   estimate.
2. **`classify`** correctly bucketing AI=0.1, AI=1/12, AI=1000,
   AI=20 as memory-bound or compute-bound (the ridge is ~9.56).
3. **`attainable_throughput_tflops`** within 2% at AI=0.1
   (memory ceiling) and AI=20 (compute ceiling).
4. **`plot_roofline(out_path)`** writes a PNG file of at least
   5000 bytes with the correct PNG magic header.

If all four pass, the kernel exits 0 with `PASS`.

## What "right" looks like

Your `roofline.png` should show vector_add, vector_fma,
axpy_scaled, relu, and softmax_row clustered well below the
ridge point (deeply memory-bound), and gemm_8192 well above
(deeply compute-bound). That visual gap is the punchline of the
whole module: not all kernels on the same GPU are bottlenecked
by the same resource, and the arithmetic intensity tells you
which.

Bonus: re-plot on an H100 (peak 67 TFLOPS, bw 3350 GB/s, ridge
~20). Notice how `softmax_row` (AI=0.625) is *still* memory-bound,
but `vector_add` is even *more* below the ridge in proportional
terms — that's why H100's bandwidth advantage helps memory-bound
kernels less than its compute advantage helps compute-bound
ones.

## Extended worked examples

### Example 1: comparing kernels at the same throughput

You measure two kernels both at exactly 200 GFLOPS on an A100:

- Kernel A is `vector_add` (AI = 0.083).
- Kernel B is a fused `relu(matmul + bias)` with AI = 5 FLOP/byte.

Which kernel is closer to its ceiling?

```
Kernel A: attainable = min(19500, 2039 * 0.083) = 169.2 GFLOPS
          Achieved 200 / 169.2 = 118% of attainable.
          IMPOSSIBLE -- the measurement or the AI is wrong.
          (Likely the AI is wrong: the kernel might be exploiting L2
           cache hits, effectively raising AI above the algorithmic
           value.)

Kernel B: attainable = min(19500, 2039 * 5) = 10195 GFLOPS
          Achieved 200 / 10195 = 2% of attainable.
          Lots of headroom; this kernel is poorly tuned.
```

Same raw throughput, completely different stories. Roofline is the
context.

### Example 2: deciding between fusion and not

You have two kernels that run back-to-back: a matmul (AI = 1000)
that produces a tensor T, and a softmax over T (AI = 0.625).
Without fusion, T is written to HBM and re-read.

```
Without fusion:
  matmul:     compute-bound, runs at ~17 TFLOPS (cuBLAS efficiency)
  write T:    bandwidth-bound, runs at ~peak BW for write
  read T:     bandwidth-bound, runs at ~peak BW for read
  softmax:    bandwidth-bound, runs at ~1.3 TFLOPS

With fusion (FlashAttention-style):
  fused op:   T never goes to HBM
              effective AI = matmul AI (mostly), still compute-bound
              total wall time = matmul time only (softmax cost amortized)
```

The fusion saves the write+read round trip on T. For a (512, 4096)
intermediate, that's 16 MB of writes + 16 MB of reads = 32 MB at
2 TB/s = 16 µs saved per kernel. On a small batch, that's a 10-20%
speedup. On a large batch, the savings compound.

The roofline tells you the fusion is *worth it* before you write
the fused kernel: the unfused softmax was at 1.3 TFLOPS (deeply
memory-bound) and the matmul leaves bandwidth slack.

### Example 3: when AI changes with implementation

Naive matmul (no tiling) reads each row of A and column of B
once per output element. For M=N=K=4096:

```
total_FLOPS = 2 * 4096^3 ≈ 1.37e11
naive bytes = 2 * 4096^3 * 4   (each output reads K elements of A + K of B, plus output)
            = 5.5e11 bytes      (massive over-fetch)
AI_naive    = 1.37e11 / 5.5e11 = 0.25 FLOP/byte
```

That's deeply memory-bound. AI_naive << ridge.

Tiled matmul with tile size 64 reads each tile of A once per K/tile
output tiles in its row, and similarly for B:

```
tiled bytes = 4 * (M*K + K*N + M*N)   (each input element touched O(1) times)
            ≈ 2e8 bytes
AI_tiled    = 1.37e11 / 2e8 ≈ 686 FLOP/byte
```

686 >> ridge. Now compute-bound.

Same algorithm, same FLOPs, dramatically different AI — because
*implementation* changed how many bytes cross the HBM boundary.
This is why tiling matters and why the AI you analyze (algorithm)
is often a lower bound on the AI you achieve (implementation).

### Example 4: when not even fusion saves you

A pure elementwise pipeline: read X → relu → write Y, all
operating on a single tensor. AI = 1 FLOP / 8 bytes = 0.125
FLOP/byte. No amount of optimization makes this compute-bound.

```
Best attainable = 2039 * 0.125 / 1000 = 0.255 TFLOPS
                = 1.3% of A100 FP32 peak
```

What can you do? Three options, none of which help on this kernel
alone:

1. **Fuse with adjacent ops.** If relu is part of a longer chain
   (linear → relu → linear → relu → ...), fuse the chain. Now
   the intermediate "Y" never exists; only the original X read
   and the final write count. AI rises.
2. **Move to lower precision.** FP16 halves the bytes/element,
   doubling AI to 0.25 FLOP/byte. Still memory-bound, but you've
   halved wall-clock time.
3. **Accept that you're at peak.** Bandwidth is the limit. Spend
   optimization budget elsewhere.

The roofline is *also* a tool for knowing when to stop optimizing.

## Extension exercise (not graded)

After the autograder passes, draw the H100 roofline (peak 67
TFLOPS, BW 3350 GB/s, ridge ~20). Plot the same six kernels.
Observe:

- The compute ceiling is 3.4× higher.
- The memory ceiling is 1.6× higher (slope is 1.6× steeper on
  log-log).
- The ridge moved right from 9.6 to 20.

For `softmax_row` (AI = 0.625):
- A100 attainable: 2039 * 0.625 / 1000 = 1.27 TFLOPS.
- H100 attainable: 3350 * 0.625 / 1000 = 2.09 TFLOPS.
- Speedup: 1.65× (matches the bandwidth ratio, since both are
  bandwidth-bound).

For `gemm_8192` (AI = 1365):
- A100 attainable: 19.5 TFLOPS (compute-bound).
- H100 attainable: 67 TFLOPS (compute-bound).
- Speedup: 3.4× (matches the compute ratio).

Same kernels, different speedups, because different ceilings bind.
This is why "H100 is X% faster than A100" is a meaningless headline
without knowing the kernel.

## Related material

- Lecture 1.5 — Roofline model (the full theory).
- Williams et al. 2009 — original roofline paper (in
  `resources.md`).
- Nsight Compute documentation — how to overlay measured kernel
  points on a roofline.
- "Hierarchical roofline" — extension model that adds L1/L2
  ceilings; lets you debug cache effects from the plot.

## Next exercise

Exercise 5 — Warp divergence analyzer. Quantify the cost of
branch divergence as a function of branch probability and per-
branch cost. Sister-skill to roofline analysis: predicting
kernel performance from kernel structure, without running it.
