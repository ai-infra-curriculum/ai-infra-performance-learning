# Lecture 1.5 — The Roofline Performance Model

> Lesson 5 of 6. Plan for ~75 minutes. The roofline is the single
> diagnostic you'll reach for first on any new kernel for the rest
> of your career as a performance engineer. Internalize it once;
> use it forever.

## Table of Contents

1. [Why we need the roofline](#why-we-need-the-roofline)
2. [Arithmetic intensity, formally](#arithmetic-intensity-formally)
3. [The model itself](#the-model-itself)
4. [Reading a roofline plot](#reading-a-roofline-plot)
5. [The ridge point](#the-ridge-point)
6. [Worked examples on an A100](#worked-examples-on-an-a100)
7. [Hierarchical rooflines](#hierarchical-rooflines)
8. [Limits and caveats](#limits-and-caveats)
9. [Practice problems](#practice-problems)

---

## Why we need the roofline

You write a kernel. It runs. You measure 800 GFLOPS. Is that good?

That question is unanswerable without context. 800 GFLOPS:

- Compared to A100's peak FP32 (19.5 TFLOPS): **4%** of peak. Bad.
- Compared to A100's peak HBM bandwidth × kernel AI (e.g., 0.083
  FLOP/byte for vector add → 169 GFLOPS ceiling): **473%** of
  peak. Impossible. Either the measurement is wrong or the AI
  estimate is wrong.

The same number — 800 GFLOPS — is wildly different depending on
*which ceiling* applies. The roofline model gives you the framework
to know which ceiling applies before you measure.

The Williams/Waterman/Patterson 2009 paper that introduced the
roofline (cited in `resources.md`) made one core observation: any
kernel's performance is bounded above by *either* peak compute *or*
peak memory bandwidth, whichever runs out first. The slower of the
two is what limits throughput.

Once you know which ceiling binds, you know which optimization
helps. Memory-bound? Reduce bytes moved, increase reuse, fix
coalescing. Compute-bound? Use faster math units (tensor cores),
reduce arithmetic, increase ILP.

---

## Arithmetic intensity, formally

The roofline rests on one number:

```
AI (arithmetic intensity) = FLOPs performed / bytes moved between HBM and SM
```

Units: FLOPS per byte. This is a property of the **algorithm**, not
of the implementation. Two implementations of the same algorithm
have the same AI (modulo dead code).

The "bytes moved between HBM and SM" is the subtle part. It does
*not* include:

- Bytes that stay in shared memory (already in the SM).
- Bytes that stay in registers.
- Bytes already cached in L1 or L2 on a re-read.

It *does* include:

- Initial loads from HBM into the SM.
- Stores from the SM out to HBM.
- Reads/writes to host memory over PCIe (if any).

In practice, for hand analysis we approximate AI as
"total FLOPs of the algorithm divided by total bytes touched at
least once." That undercounts reuse from caches; reality is
usually better. Use it as a *lower bound*.

### Examples worked out

**Vector add: `c[i] = a[i] + b[i]`, all FP32.**

For each output:
- FLOPs: 1 (one add)
- Bytes moved: 4 (read a) + 4 (read b) + 4 (write c) = 12

```
AI = 1 / 12 ≈ 0.083 FLOP/byte
```

This is one of the lowest AIs you'll see. Vector add is *the*
archetypal memory-bound kernel.

**FMA vector: `c[i] = a[i] * b[i] + d[i]`, all FP32.**

For each output:
- FLOPs: 2 (mul + add, fused into one FMA)
- Bytes moved: 4 + 4 + 4 + 4 = 16

```
AI = 2 / 16 = 0.125 FLOP/byte
```

**AXPY: `y[i] = alpha * x[i] + y[i]`, all FP32.**

For each output:
- FLOPs: 2 (mul + add)
- Bytes moved: 4 (read x) + 4 (read y) + 4 (write y) = 12
  (alpha is a scalar broadcast — comes from a register, doesn't
  contribute to HBM bytes)

```
AI = 2 / 12 ≈ 0.167 FLOP/byte
```

**Dense GEMM: `C = A @ B`, all FP32, square M=N=K.**

For the entire kernel:
- FLOPs: `2 * M * N * K` (M*N output elements, each costing 2*K)
- Bytes moved: `4 * (M*K + K*N + M*N)` (read A, read B, write C)

```
AI = 2*M*N*K / (4 * (M*K + K*N + M*N))
   = M*N*K / (2 * (M*K + K*N + M*N))
```

For M=N=K=4096:
```
AI = 4096^3 / (2 * (3 * 4096^2))
   = 4096 / 6
   ≈ 683 FLOP/byte
```

This is *very* high — orders of magnitude above the ridge point
on any current GPU. GEMM is the prototypical compute-bound kernel.
That's why GPUs are good at AI workloads (which are mostly GEMM
underneath).

**Softmax over a (batch, dim) tensor in FP32.**

For each output row:
- FLOPs per element: ~5 (max scan, subtract, exp, divide, plus a
  small constant for the running-sum reduction)
- Bytes per element: 4 (read) + 4 (write) = 8

```
AI ≈ 5 / 8 = 0.625 FLOP/byte
```

Memory-bound but not as deeply as vector add. Softmax often runs
at the HBM bandwidth ceiling on data-center GPUs.

### Per-element vs per-kernel AI

For element-wise kernels, AI is easy to compute per-element. For
kernels with reduction structure (GEMM, attention, prefix sum),
you need *total* FLOPs and *total* bytes — per-element doesn't
factor cleanly.

The rule of thumb: if every input byte is used to produce O(1)
output FLOPs, the kernel is memory-bound. If every input byte is
reused to produce O(N) output FLOPs (think GEMM's tile reuse),
the kernel can be compute-bound.

---

## The model itself

The roofline says: for any arithmetic intensity AI, the maximum
achievable throughput is:

```
attainable_throughput(AI) = min(peak_FLOPS, peak_bandwidth * AI)
```

Both terms in the `min` have units of FLOPS:

- `peak_FLOPS` is the GPU's peak compute throughput (FLOPS).
- `peak_bandwidth * AI` is bytes/sec × FLOPS/byte = FLOPS/sec.

The first term doesn't depend on AI (it's a flat ceiling). The
second is linear in AI (it goes through the origin).

```
            attainable throughput
                  ^
                  |
        peak FLOPS|  ---------------+------------------    compute
                  |               /                       ceiling
                  |              /
                  |             /
                  |            / (slope = peak BW)
                  |           /
                  |          /
                  |         /
                  |        /  <-- "memory ceiling"
                  |       /       linear in AI
                  |      /
                  |     /
                  |    /
                  |   /
                  +----+--------------------> arithmetic
                       |                       intensity
                  AI_ridge = peak_FLOPS / peak_BW
```

Where the two ceilings meet is the **ridge point**:

```
AI_ridge = peak_FLOPS / peak_bandwidth
```

Left of the ridge: bandwidth limits you. Right of the ridge: compute
limits you.

---

## Reading a roofline plot

A roofline plot is, almost always, on log-log axes:

- X axis: arithmetic intensity (FLOP/byte), log scale.
- Y axis: throughput (TFLOPS or GFLOPS), log scale.

Reasons for log scale:

- AI spans **four+ orders of magnitude** in practice (0.08 for
  vector add; 1000+ for large GEMM). Linear axes would crush
  everything into the corner.
- The "memory ceiling" line (`bw * AI`) becomes a straight line
  with slope 1 on log-log axes; the "compute ceiling" stays flat.
- Multiplicative differences ("this kernel achieves 10× more
  throughput than that one") show as equal vertical spacing.

A point on the plot represents a kernel:

- **At AI**: the kernel's arithmetic intensity (a property of the
  algorithm).
- **At throughput Y**: the kernel's actually-achieved throughput
  in your measurement.

Three diagnostic regions:

1. **Far below the memory ceiling line**: you're leaving bandwidth
   on the floor. Memory-bound kernel, achieving much less than its
   own ceiling. Look at: coalescing, cache hit rate, redundant loads.

2. **Right on the memory ceiling line**: bandwidth-limited and
   well-tuned. To go further, you must change the algorithm to
   raise AI.

3. **Well below the compute ceiling but right of the ridge**:
   compute-bound, but not saturating compute. Look at: register
   pressure, ILP, tensor-core utilization.

A roofline tells you *what to optimize next*, before the profiler
sends you down a rabbit hole.

---

## The ridge point

The ridge AI tells you "how much arithmetic, per byte, do I need
to make this GPU useful?"

| GPU | Peak FP32 | Peak BW | Ridge AI (FP32) |
|---|---|---|---|
| A100 SXM4 80GB | 19.5 TFLOPS | 2039 GB/s | ~9.6 FLOP/byte |
| H100 SXM5 80GB | 67 TFLOPS | 3350 GB/s | ~20 FLOP/byte |
| RTX 4090 | 82.6 TFLOPS | 1008 GB/s | ~82 FLOP/byte |

Two observations:

1. **H100's ridge point (20) is higher than A100's (9.6).** H100
   has more compute *relative to* bandwidth. For deeply
   memory-bound kernels (vector add etc.), the H100 doesn't help
   much — bandwidth scaled less than compute did. For
   compute-bound kernels (GEMM), H100 is dramatically faster.

2. **RTX 4090's ridge is ~82.** Consumer GPUs have very high
   compute and modest bandwidth, so the ridge is much higher.
   Many real workloads that are compute-bound on H100 are
   bandwidth-bound on a 4090 — even though the 4090's *peak*
   compute is higher.

When you read benchmark numbers, always ask: which side of the
ridge is the workload on, and which GPU are we comparing? Those
two questions usually predict the comparison.

### Ridge point for tensor cores

If you're using tensor cores instead of FP32 CUDA cores, the
"peak compute" is much higher (16× on A100 for FP16). The ridge
shifts right by the same factor:

```
A100 FP32 ridge:   ~9.6 FLOP/byte
A100 FP16 TC ridge: ~9.6 * 16 = ~150 FLOP/byte
```

So even GEMM that was compute-bound at FP32 becomes a different
optimization problem at FP16 tensor cores. This is one of the
reasons large LLMs run "on the bandwidth ceiling" for attention
even on H100 — the FP16 tensor-core ridge is so high that even
attention's substantial AI doesn't clear it.

---

## Worked examples on an A100

Pick A100 SXM4 80GB:
- Peak FP32: 19.5 TFLOPS = 19,500 GFLOPS
- Peak HBM bandwidth: 2039 GB/s
- Ridge AI: 19500 / 2039 ≈ 9.56 FLOP/byte

For each kernel, compute (a) AI, (b) which side of the ridge,
(c) the attainable throughput at peak bandwidth, (d) the realistic
percentage of peak you'd see.

### 1. Vector add, FP32, large N

```
AI = 1 / 12 ≈ 0.083 FLOP/byte
Ridge = 9.56. AI << Ridge -> deeply memory-bound.

Attainable throughput at peak BW:
  2039 GB/s * 0.083 FLOP/byte = 169.7 GFLOPS

Percentage of compute peak: 169.7 / 19500 ≈ 0.87%
```

Vector add achieves at most ~170 GFLOPS on a chip with 19,500
GFLOPS of FP32 peak. Almost two orders of magnitude below the
compute ceiling. Optimizing for compute on vector add is pointless;
the only knob is HBM bandwidth utilization (which you achieve via
coalesced reads, not via more arithmetic).

### 2. Dense GEMM, FP32, 8192^3

```
AI = (2 * 8192^3) / (12 * 8192^2) = 2 * 8192 / 12 ≈ 1365 FLOP/byte
Ridge = 9.56. AI >> Ridge -> deeply compute-bound.

Attainable throughput:
  min(19500, 2039 * 1365) = min(19500, ~2.78M) = 19,500 GFLOPS
  i.e. exactly peak FP32.
```

GEMM at this size is compute-bound. cuBLAS achieves ~85% of peak
FP32 in practice (the remaining 15% is non-FMA instruction overhead,
warp scheduling overhead, and end-of-tile waste). At FP16 with
tensor cores, it's even higher.

### 3. Softmax row, FP32, (B=512, D=4096)

```
AI ≈ 5 / 8 = 0.625 FLOP/byte
Ridge = 9.56. AI < Ridge -> memory-bound, but not deeply.

Attainable throughput at peak BW:
  2039 * 0.625 ≈ 1274 GFLOPS

Percentage of compute peak: 1274 / 19500 ≈ 6.5%
```

Softmax achieves at most ~1.3 TFLOPS even at peak HBM bandwidth.
This is still memory-bound; the only way to make it faster is
fusion with adjacent ops (so you don't write to HBM and re-read).
This is the intuition behind FlashAttention — fusing softmax
with the matmul above and below it to keep intermediates on-chip.

### 4. ReLU element-wise, FP32

```
AI = 1 / 8 = 0.125 FLOP/byte  (just a compare-and-select, one FLOP per element)
                              (4 bytes read + 4 bytes write = 8 bytes)
Ridge = 9.56. AI << Ridge -> memory-bound.

Attainable throughput at peak BW:
  2039 * 0.125 ≈ 255 GFLOPS

Percentage of compute peak: 255 / 19500 ≈ 1.3%
```

ReLU is essentially a bandwidth benchmark. It runs at HBM speed
if implemented competently. Fusing it with the matmul that
produced its input is the only way to "speed it up."

### 5. The pathological case: AXPY with non-coalesced access

```
AI = 2 / 12 ≈ 0.167 FLOP/byte (algorithm)

But if the access is uncoalesced (stride-32 reads), effective HBM
bandwidth is 1/8 of peak:
  effective_BW = 2039 / 8 ≈ 255 GB/s

Attainable at *effective* bandwidth:
  255 * 0.167 ≈ 42 GFLOPS

vs. naive expectation at peak BW:
  2039 * 0.167 ≈ 340 GFLOPS
```

The kernel is still memory-bound, but it's bound by *bad*
bandwidth, not by HBM peak. Fixing coalescing is an 8× speedup
here without touching the algorithm.

You'll compute the AI for six kernels and place them on a roofline
in **Exercise 4**.

---

## Hierarchical rooflines

Real GPUs have *multiple* memory levels, not just HBM. A hierarchical
roofline draws additional lines for each level:

- L2 cache: ~hundreds of GB/s (5–10× higher than HBM peak).
- L1 / shared: TB/s per SM (chip-wide is many TB/s).

```
                    ^
                    |  +-- compute ceiling
                    |  |
        peak FLOPS  +--+-------- (flat)
                    |   \  L1/shared ceiling (very steep)
                    |    \
                    |     \
                    |      \  L2 ceiling
                    |       \
                    |        \
                    |         \
                    |          \  HBM ceiling
                    |           \
                    +----------+----------> AI
                            ridge
```

Kernels that hit L2 or L1 ceilings (with high cache hit rates) can
run *above* the HBM ceiling on the plot. That's the visual
signature of a well-cached kernel: it lives between the HBM line
and the next level up.

This is also where the "Hopper's L2 cache residency" trick shows
up — for a KV cache that fits in L2, you can stay on the L2
ceiling instead of dropping to HBM, gaining ~10× effective
bandwidth for that part of the workload.

You won't need to construct hierarchical rooflines in this module,
but you should recognize them when you see them in NVIDIA's docs
and conference talks.

---

## Limits and caveats

The roofline model is a *first-order* tool. It doesn't account for:

- **Latency-not-bandwidth limited kernels.** Small kernels that
  finish before HBM is even saturated. The roofline assumes
  steady-state throughput.
- **Kernel launch overhead.** A 5-µs launch is invisible on a
  100-ms kernel but huge on a 50-µs kernel.
- **Synchronization overhead.** `__syncthreads()` calls, atomic
  contention, cross-block waits.
- **Tensor core vs CUDA core distinction.** A kernel using only
  CUDA cores is bounded by FP32 peak; switching to tensor cores
  with FP16 raises the compute ceiling 16× (on A100).
- **Multiple peak ceilings.** A kernel might be FP32-bounded for
  one phase, tensor-core-bounded for another. Roofline shows one
  ceiling at a time.
- **Power throttling.** A real chip running near thermal limits
  may not sustain peak. The "peak" in the formula is the
  manufacturer's number; the chip in your rack may deliver less.

In practice the roofline gives you the right intuition 80% of the
time. The other 20% is where Nsight Compute, hierarchical
rooflines, and detailed analysis come in — but you reach for
those *after* the roofline tells you which question to ask.

The most common failure mode of the roofline as an analysis tool
is forgetting that **AI depends on reuse**, which depends on
implementation. A naive matmul has AI ≈ K. A tiled matmul has the
same FLOPs but moves much less data, so its *effective* AI is
much higher. The roofline you analyze is the algorithm's AI; the
roofline you measure is the implementation's AI. They can differ
by 100×.

---

## Practice problems

**P1.** A kernel has AI = 4 FLOP/byte. On an A100 (peak FP32 19.5
TFLOPS, peak BW 2039 GB/s), is it memory-bound or compute-bound?
What's its peak attainable throughput?

**P2.** You measure a kernel at 1500 GFLOPS on an A100. Its AI is
1 FLOP/byte. What percentage of its attainable throughput is it
achieving?

**P3.** A100's FP32 ridge is ~9.6. H100's FP32 ridge is ~20. What
does this tell you about which GPU "favors" memory-bound vs
compute-bound workloads?

**P4.** A vector_add (AI = 0.083) runs at 1700 GB/s effective HBM
bandwidth. That's 83% of peak. Convert this to GFLOPS achieved
and as a percentage of the kernel's attainable throughput.

**P5.** You fuse a softmax (AI ≈ 0.6) with the matmul above it
(AI ≈ 1000). The fused kernel's AI is closer to which of these
extremes — and why?

---

### Answers

**A1.** Ridge is ~9.6. AI = 4 < 9.6, so **memory-bound**.
Attainable = `2039 * 4 = 8156 GFLOPS = 8.2 TFLOPS`.

**A2.** Attainable = `min(19500, 2039 * 1) = 2039 GFLOPS`. Achieved
1500 / 2039 = **73.6%** of attainable. Pretty good.

**A3.** H100's higher ridge (20 vs 9.6) means it needs more
arithmetic per byte to saturate compute. So H100 is **relatively**
better for compute-bound workloads — bandwidth grew less than
compute. Memory-bound workloads benefit less from H100 because the
bandwidth gain is smaller than the compute gain.

**A4.** GFLOPS = 1700 GB/s × 0.083 FLOP/byte ≈ 141 GFLOPS.
Attainable = 2039 × 0.083 = 169 GFLOPS. 141 / 169 = **83%** of
attainable. Same percentage as the bandwidth utilization, because
the kernel is bandwidth-bound — bandwidth utilization *is* the
throughput utilization for memory-bound kernels.

**A5.** Closer to the **softmax** AI (0.6), or actually higher because
fusion eliminates the intermediate write/read. The fused kernel
still has to load the matmul inputs from HBM (high cost, lots of
bytes). The fusion saves the *intermediate* write/read between
matmul and softmax. So total bytes go down a bit, total FLOPs stay
the same — AI goes up, but only by the fraction of bytes saved.
For typical sizes, fused AI ends up maybe 1.5–2× the softmax AI,
still memory-bound but a meaningful win. The matmul's high AI
doesn't transfer because most of those FLOPs were already
amortized over the matmul's reused tiles, not over the softmax.

---

*End of Lecture 1.5. Next: Lecture 1.6 — GPU Generations and Specs.*
