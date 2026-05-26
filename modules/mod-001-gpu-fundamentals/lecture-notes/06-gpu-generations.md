# Lecture 1.6 — GPU Generations and Specs

> Lesson 6 of 6. Plan for ~60 minutes. This is the lecture you'll
> reference when someone asks "should we buy A100 or H100 for
> training? L40S or 4090 for inference?" The skill is reading a
> datasheet without losing the plot.

## Table of Contents

1. [Why GPU SKUs matter](#why-gpu-skus-matter)
2. [The reference table](#the-reference-table)
3. [How to read an NVIDIA datasheet](#how-to-read-an-nvidia-datasheet)
4. [A100 — the workhorse](#a100--the-workhorse)
5. [H100 — the transformer engine](#h100--the-transformer-engine)
6. [L40S — the inference card](#l40s--the-inference-card)
7. [RTX 4090 — consumer GPUs at the data-center boundary](#rtx-4090--consumer-gpus-at-the-data-center-boundary)
8. [What changed in Blackwell (preview)](#what-changed-in-blackwell-preview)
9. [Procurement decision framework](#procurement-decision-framework)
10. [Practice problems](#practice-problems)

---

## Why GPU SKUs matter

Two reasons performance engineers need to understand GPU SKUs:

1. **Procurement decisions are huge.** A100s are ~$10k each at
   spot, H100s are $25–35k, B100/B200 are $30–40k. An 8-GPU node
   at ~$200k per node × dozens of nodes adds up fast. Picking
   the right SKU for the workload is a six-figure decision per
   rack.

2. **Workloads run differently on different GPUs.** A model that's
   compute-bound on A100 might be memory-bound on a 4090 (different
   ridge points). A model that fits in HBM on H100 (80 GB) might
   not fit on an L40S (48 GB). The same code, the same model — but
   the engineering work is different depending on hardware.

This lecture doesn't make you a SKU expert. It gives you the
five or six numbers per GPU that drive the analysis, plus a
framework for reading a new datasheet when NVIDIA launches the
next chip.

---

## The reference table

The numbers you'll cite for the rest of the curriculum. **All FP32
TFLOPS values are dense (no sparsity); all bandwidth values are
peak HBM/GDDR.** Sources are in `resources.md`.

| Spec | A100 SXM4 80GB | H100 SXM5 80GB | L40S | RTX 4090 |
|---|---|---|---|---|
| Architecture | Ampere | Hopper | Ada Lovelace | Ada Lovelace |
| Compute capability | 8.0 | 9.0 | 8.9 | 8.9 |
| SMs | 108 | 132 | 142 | 128 |
| FP32 CUDA cores | 6,912 | 16,896 | 18,176 | 16,384 |
| FP32 cores per SM | 64 | 128 | 128 | 128 |
| Tensor cores per SM | 4 (3rd gen) | 4 (4th gen) | 4 (4th gen) | 4 (4th gen) |
| Boost clock | ~1.41 GHz | ~1.98 GHz | ~2.52 GHz | ~2.52 GHz |
| Peak FP32 (TFLOPS) | 19.5 | 67 | 91.6 | 82.6 |
| Peak FP16 tensor (TFLOPS, dense) | 312 | 989 | 366 | 330 |
| Peak BF16 tensor (TFLOPS, dense) | 312 | 989 | 366 | 330 |
| Peak FP8 tensor (TFLOPS, dense) | n/a | 1,979 | 733 | 660 |
| Memory | 80 GB HBM2e | 80 GB HBM3 | 48 GB GDDR6 | 24 GB GDDR6X |
| Memory bandwidth (GB/s) | ~2,039 | ~3,350 | ~864 | ~1,008 |
| L2 cache | 40 MB | 50 MB | 96 MB | 72 MB |
| Max shared mem per block | 163 KB | 228 KB | 99 KB | 99 KB |
| Max registers per thread | 255 | 255 | 255 | 255 |
| Max threads per block | 1,024 | 1,024 | 1,024 | 1,024 |
| Max warps per SM | 64 | 64 | 48 | 48 |
| TDP | 400 W | 700 W | 350 W | 450 W |
| Form factor | SXM4 | SXM5 | PCIe | PCIe |
| NVLink? | Yes (12x 50 GB/s) | Yes (18x 50 GB/s) | No (PCIe Gen4 x16) | No |

A few things to internalize from this table alone:

- **CUDA-cores-per-SM changed in Hopper.** Ampere data-center
  (GA100) had 64 FP32 cores per SM. Hopper and Ada have 128.
  When you derive peak throughput, you need the right "per-SM"
  number for the chip you're analyzing.
- **Consumer Ampere (RTX 30-series) had 128 FP32 cores per SM
  too** — but only by reusing INT32 datapaths. The data-center
  A100 is the outlier with 64.
- **Bandwidth scales sub-linearly with compute generation-over-generation.**
  H100 has 3.4× the compute of A100 but only 1.6× the bandwidth.
  This is why the H100 ridge is higher (more compute per byte).
- **L40S has the most L2 of any current SKU** (96 MB). That
  matters for inference workloads where activations fit in L2 and
  stay there across layers.
- **Max warps per SM dropped from 64 (Ampere) to 48 (Ada).**
  Consumer-class chips have less concurrency capacity but the
  same per-warp throughput. A100/H100's 64-warp ceiling is part
  of why they remain compelling for training workloads.

---

## How to read an NVIDIA datasheet

NVIDIA publishes datasheets for every GPU SKU. They're 4–8 pages.
The numbers you care about, in order:

1. **SM count** and **FP32 CUDA cores total**. Page 1 usually.
2. **Boost clock**. Sometimes "GPU boost clock" or "Peak GPU clock."
3. **Memory size, type, bandwidth, bus width.** All four.
4. **Peak TFLOPS for FP32, FP16/BF16, FP8, INT8.** Look carefully
   at the "with sparsity" footnote — those are 2× cooked numbers.
5. **TDP** — both for power budgeting and for thinking about
   throttling.
6. **NVLink topology**, if any. Multi-GPU performance depends
   heavily on this.

What you can usually ignore:

- "Graphics features" (RT cores, optical-flow accelerators).
- Video encode/decode block counts.
- Display outputs.
- Most "AI TOPS" marketing numbers — they often mix sparsity,
  data types, and tensor-core generations in confusing ways.

Cross-check against the **CUDA C Programming Guide, Appendix H**,
for the architectural limits (max warps, registers, shared mem,
etc.). The Programming Guide is the authoritative source for
*architectural* limits; the datasheet is authoritative for the
*specific SKU's* clocks and bandwidth.

### Where sparsity hides

NVIDIA's "structured sparsity" — typically called 2:4 sparsity —
allows tensor cores to skip 2 out of every 4 values when they're
zero. The reported TFLOPS with sparsity is **exactly 2× the
dense number** for any data type. Tensor-core specs are usually
quoted both ways:

> H100 SXM5: 1,979 TFLOPS FP8 dense, 3,958 TFLOPS FP8 with sparsity

If your model isn't trained with structured sparsity, plan around
the **dense** number. Most workloads in 2026 still are not using
sparsity at training time; some inference flows use it. When you
see "TOPS" or "TFLOPS" in vendor marketing, default-assume sparsity
is included until you confirm otherwise.

---

## A100 — the workhorse

The A100 (Ampere, CC 8.0) launched in 2020 and remains, in 2026,
the most-deployed data-center GPU. Two memory variants: 40 GB and
80 GB. Two form factors: PCIe (250 W) and SXM4 (400 W).

**Why A100 still matters:**

- Mature software stack. Every CUDA-aware library has been
  optimized for it.
- 80 GB HBM2e is enough for many large models without
  partitioning.
- $7–10k street price (in 2026) — about 1/3 of an H100.
- Plenty of supply on the secondary / colocation market.

**What A100 is bad at:**

- No FP8. If your inference stack uses FP8 (more and more
  do), A100 falls off a cliff vs H100.
- No Transformer Engine (Hopper-exclusive). Loss of ~2× perf on
  modern attention implementations.
- Older NVLink (12× 50 GB/s = 600 GB/s aggregate). H100 has
  18× 50 GB/s = 900 GB/s.

**Architectural specifics:**

- 108 SMs × 64 FP32 cores per SM = 6,912 FP32 cores.
- 4 tensor cores per SM × 108 SMs = 432 third-gen tensor cores.
- 65,536 32-bit registers per SM.
- Up to 163 KB shared memory per block (with carveout).
- 40 MB L2 cache.

**The exception case for the "64 cores per SM" rule:** A100 (GA100)
is the only modern data-center SKU with 64 FP32 cores per SM
instead of 128. Other Ampere chips (GA102 / RTX 30-series) have
128. Consumer-vs-datacenter has been the determinant since
Ampere — datacenter prioritizes register file size and tensor-core
throughput; consumer prioritizes raw FP32 count.

A100 is still a smart procurement choice for: workloads that fit
in 80 GB HBM, don't need FP8, and are price-sensitive. Many
inference clusters at companies that aren't bleeding-edge are
still A100-based in 2026.

---

## H100 — the transformer engine

The H100 (Hopper, CC 9.0) launched in 2022 and is the dominant
training chip for frontier-scale models in 2026.

**What's new vs A100:**

- **FP8 tensor cores.** Fourth-gen tensor cores support FP8 at
  ~2× FP16 throughput. With Transformer Engine, FP8 training is
  practical at minimal accuracy loss for LLMs.
- **Transformer Engine.** Software + hardware co-design for
  automatic FP8 scaling. Critical for any production transformer
  training on H100.
- **TMA (Tensor Memory Accelerator).** Hardware-managed
  bulk-asynchronous data movement between HBM and shared memory.
  Lets one warp issue a multi-KB transfer and let it run in the
  background. Used by FlashAttention 2/3, cuBLAS, cuDNN.
- **Thread block clusters.** Multiple blocks can synchronize and
  share memory across SMs in the same cluster. Relaxes the
  "blocks are independent" constraint of every prior generation.
- **DPX instructions.** Hardware acceleration for dynamic
  programming (e.g., Smith-Waterman). Niche but real for
  genomics workloads.
- **Confidential computing.** Encrypted memory pages for
  multi-tenant deployments. Compliance-relevant.

**Numbers worth memorizing:**

- 132 SMs × 128 FP32 cores per SM = 16,896 FP32 cores.
- FP32 peak: 67 TFLOPS (~3.4× A100).
- HBM3 bandwidth: 3,350 GB/s (~1.65× A100).
- 50 MB L2 cache.
- TDP: 700 W (vs A100's 400 W). Density and cooling are real
  constraints in H100 deployments.

**What H100 doesn't help with:**

- Memory-bound kernels that don't decompose into MMAs (e.g.,
  sparse linear algebra, irregular graphs).
- Workloads dominated by host↔device transfer (PCIe didn't get
  faster).
- Anything that fits in <40 GB of HBM (you're paying for
  capacity you don't use).

H100 is the right choice for: frontier LLM training, attention-heavy
inference at scale, and anything where the workload runs in FP8
or with tensor-core-friendly arithmetic intensity.

---

## L40S — the inference card

The L40S (Ada Lovelace, CC 8.9) is NVIDIA's data-center inference
SKU, launched 2023. Same chip family as the RTX 4090 (AD102) but
with ECC memory, professional drivers, and a different cooling
profile.

**Why L40S exists:**

- L40 (no S) is more graphics-focused.
- A100 is great but pricey for inference-only workloads.
- H100 is great but absurdly expensive for inference where you
  don't need FP8 training and don't push HBM size limits.

**Specs that matter:**

- 142 SMs × 128 FP32 cores per SM = 18,176 FP32 cores. More raw
  FP32 than A100.
- FP32 peak: 91.6 TFLOPS (4.7× A100, 1.4× H100 — but consumer
  pricing, not data-center).
- 48 GB GDDR6 (no HBM, so memory is cheaper).
- Bandwidth: 864 GB/s — significantly lower than A100 (2 TB/s).
  This is the catch.
- 96 MB L2 cache — *largest of any current SKU*. Compensates
  partially for the GDDR bandwidth gap on cache-friendly inference
  loads.
- 350 W TDP.

**The L40S tradeoff:**

L40S is compute-rich and bandwidth-poor relative to A100. For
inference workloads that are compute-bound (most modern LLM
inference at typical batch sizes), L40S wins on perf-per-dollar.
For workloads that are bandwidth-bound (small-batch attention,
KV-cache-heavy decoding), A100 may still beat it.

The 96 MB L2 is genuinely impactful: a KV cache that fits in 96
MB stays in L2 across decode steps, giving you 5–10× effective
bandwidth over GDDR. NVIDIA's marketing for L40S leans heavily on
this.

L40S is the right choice for: high-throughput inference at
batch sizes where compute is the bottleneck, especially for models
that fit in 48 GB.

---

## RTX 4090 — consumer GPUs at the data-center boundary

The RTX 4090 (Ada Lovelace, CC 8.9) is a consumer card, $1,500–2,500
street price in 2026. It has the same AD102 chip as L40S but with
24 GB of GDDR6X, the consumer driver, and a fan-cooled card.

**Why anyone uses 4090s in production:**

- 24 GB is enough for many models (especially with quantization).
- Per-dollar FP32 throughput is *outstanding*. 82.6 TFLOPS for
  $2k is hard to match.
- Local-deployment / on-prem inference where data-center cooling
  doesn't apply.
- Hobbyist and small-business AI workloads.

**Why NVIDIA's EULA mostly forbids it:**

NVIDIA's consumer driver EULA includes a clause prohibiting use
in data centers (with carve-outs for crypto, blockchain, etc.).
Enforcement is uneven, but it's a real legal risk for commercial
deployments.

**Why 4090s aren't a great fit for serious work anyway:**

- **No ECC memory.** Bit flips are rare but not zero, and at
  inference scale (millions of tokens/day) you'll hit them.
- **No NVLink.** Multi-GPU on a 4090 is limited to PCIe (32 GB/s).
  H100/A100 NVLink is 600–900 GB/s aggregate.
- **24 GB ceiling.** A 70B-parameter model in FP16 won't fit in
  one card.
- **Consumer drivers** lack some MIG (multi-instance GPU) and
  virtualization features that data-center deployments rely on.

For the curriculum, the 4090 is a useful pedagogical case: it has
high compute, moderate bandwidth, and a *very* high ridge point
(~82 FLOP/byte). Workloads that look compute-bound on H100 are
often memory-bound on a 4090. Exercise 2 walks you through the
derivation of its peak numbers.

---

## What changed in Blackwell (preview)

Blackwell (CC 10.0, B100/B200) launched in 2024. Per-die throughput
figures from early decks were revised more than once before NVIDIA
published its final datasheets; this lecture deliberately stays
qualitative rather than quoting a single TFLOPS number that could
go stale. For procurement-grade numbers, always cross-check the
current [NVIDIA Blackwell datasheet](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
and your vendor's quote.

What's structurally new:

- **FP4 tensor cores.** New low-precision data type, primarily for
  inference. Doubles tensor-core throughput vs FP8 (8× vs FP16).
- **Two-die package.** The B200 is two GB100 dies on one package,
  connected by a chip-to-chip interconnect at TB/s scale. NVIDIA
  presents this as one "GPU" to the programmer.
- **HBM3e.** Same wide bus as HBM3 but higher per-pin clock,
  ~1.5× more bandwidth.
- **More L2 per die.** Larger on-chip cache.
- **NVLink 5.** 1.8 TB/s aggregate per GPU, vs 900 GB/s on H100.

What stays the same:

- Warp size is still 32.
- SM is still the scheduling unit.
- Memory hierarchy is still register / shared / L1 / L2 / global.
- CUDA programming model is backward-compatible.

For this curriculum we won't lean on Blackwell numbers — too
fluid still. Mentally tag B200 as "H100 but ~2× across the board
for AI workloads," and revisit when NVIDIA's datasheets stabilize.

---

## Procurement decision framework

A simple framework you can apply when asked "which GPU do we buy?":

**Step 1: Is the workload compute-bound or memory-bound at the
expected scale?**

Compute the AI of the kernel (or the dominant kernel). Compare
to the ridge points of candidate GPUs.

**Step 2: How big is the working set?**

If it doesn't fit in HBM, you'll spend wall-clock time on
host↔device transfer regardless of the GPU. If it fits in 24
GB, a 4090 / L40S is plausible. If it needs 80 GB, you're in
A100/H100 territory. If you need 141 GB, H200.

**Step 3: Does the workload use FP8 / sparsity / Transformer Engine?**

If yes, H100 or newer is the floor.

**Step 4: Is this training (sustained, multi-GPU) or inference
(latency-sensitive, often single-GPU)?**

Training: NVLink matters. SXM form factor matters. H100 SXM5 or
H200 SXM5 are the standard. A100 SXM4 if budget is constrained.

Inference: PCIe is usually fine. L40S, H100 PCIe, or A100 PCIe.
Sometimes 4090 if the constraints permit.

**Step 5: TCO / availability / driver-stack maturity.**

H100 supply has historically been constrained. A100 supply is
mature. L40S is the inference compromise. Blackwell is rolling
out in waves.

This framework doesn't give you a definitive answer — every real
procurement is more nuanced — but it gives you the shape of the
analysis. The numerical inputs all come from the table at the top
of this lecture.

---

## Practice problems

**P1.** A workload's dominant kernel has AI = 5 FLOP/byte. You're
choosing between A100 (ridge 9.6), H100 (ridge 20), and 4090
(ridge 82). On which GPUs is this workload memory-bound?

**P2.** A model uses 60 GB in FP16. Which of A100 80GB, H100 80GB,
L40S 48GB, RTX 4090 24GB can hold it in a single GPU's HBM?

**P3.** A datasheet quotes "5,000 TFLOPS FP8 with sparsity." What
peak should you plan around for a workload without structured
sparsity?

**P4.** Why is H100's ridge point higher than A100's, and what
does that mean for porting a memory-bound kernel from A100 to H100?

**P5.** You see an inference workload at 90% of A100 FP32 peak.
You move it to L40S (which has 4.7× the FP32 compute of A100).
Predict the speedup. (Hint: think about which ridge you're now
on.)

---

### Answers

**A1.** AI=5 is below A100's ridge (9.6) → memory-bound on A100.
Below H100's ridge (20) → memory-bound on H100. Below 4090's
ridge (82) → memory-bound on 4090. Memory-bound on all three.

**A2.** A100 80GB (yes), H100 80GB (yes), L40S 48GB (no — doesn't
fit), RTX 4090 24GB (no — needs to spill).

**A3.** 2,500 TFLOPS (half the sparsity number). The sparsity
number assumes 50% of weights are zero; without that, you're on
the dense path.

**A4.** H100 has 3.4× A100's compute but only 1.65× the bandwidth,
so compute grew faster than bandwidth. A memory-bound kernel
that achieves a constant fraction of peak bandwidth on both will
see only ~1.65× speedup on H100, not 3.4×. The relative gain on
H100 is highest for compute-bound kernels.

**A5.** Probably much less than 4.7×. The workload was
compute-bound on A100 (90% of FP32 peak), but L40S has 4.7× the
compute and only 0.42× the bandwidth (864/2039). The ridge moves
from 9.6 (A100) to ~106 (L40S — 91,600 GFLOPS / 864 GB/s). Many
workloads that were just barely compute-bound on A100 will be
*memory*-bound on L40S. You might see 1.5–2.5× speedup, not 4.7×.
The L2 cache on L40S (96 MB vs A100's 40 MB) helps somewhat — if
the working set fits in L2.

---

## Module 1 wrap-up

You've now covered:

- **Lecture 1.1** — GPU architecture (SMs, sub-partitions, tensor
  cores, compute capability).
- **Lecture 1.2** — CUDA programming model (threads, blocks,
  grids, kernel launches).
- **Lecture 1.3** — Memory hierarchy (registers, shared, L1, L2,
  HBM, coalescing, bank conflicts).
- **Lecture 1.4** — Warps and occupancy (latency hiding, the
  three resource caps, divergence, warp-level primitives).
- **Lecture 1.5** — Roofline model (arithmetic intensity, ridge
  point, classifying kernels memory-bound vs compute-bound).
- **Lecture 1.6** — GPU generations and SKUs (this lecture).

Everything in the rest of the course reduces to these primitives.
When you read about FlashAttention in Module 4, ask: what's the AI?
What's the memory pattern? Where does it live in the hierarchy?
When you tune a kernel in Module 2: what's the binding occupancy
resource? Is the access coalesced? Is the warp diverging?

You don't need to remember every number in this lecture. You need
to know **how to find them in 90 seconds when you need them.**
Datasheet, ridge point, AI of your workload, predict the bound,
profile to confirm, optimize. That loop is the rest of your
career as a performance engineer.

If the six exercises pass, you're ready for Module 2.

---

*End of Lecture 1.6. End of Module 1 lecture-notes.*
