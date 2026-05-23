# Lecture 1.1 — GPU Architecture Deep Dive

> Lesson 1 of 6 in Module 1. Plan for ~90 minutes of reading and
> ~60 minutes on the practice problems at the end.

## Table of Contents

1. [Why a GPU is shaped the way it is](#why-a-gpu-is-shaped-the-way-it-is)
2. [The SM is the unit that matters](#the-sm-is-the-unit-that-matters)
3. [Walking the on-chip hierarchy top-down](#walking-the-on-chip-hierarchy-top-down)
4. [Tensor cores: the second arithmetic engine](#tensor-cores-the-second-arithmetic-engine)
5. [What changes (and doesn't) across generations](#what-changes-and-doesnt-across-generations)
6. [Common misconceptions](#common-misconceptions)
7. [Worked example: counting FP32 lanes on an H100](#worked-example-counting-fp32-lanes-on-an-h100)
8. [Practice problems](#practice-problems)

---

## Why a GPU is shaped the way it is

A CPU and a GPU answer the same question — *how do we execute
instructions on data?* — but they answer it in opposite directions.

A modern x86 CPU core spends most of its silicon on **single-thread
latency**: out-of-order issue, branch prediction, large per-core
caches, memory-dependence prediction, deep speculation pipelines.
The point of all that machinery is to make one stream of dependent
instructions retire as fast as possible. Server CPUs have 8–128 such
cores per socket; each of them is, on its own, a very capable
general-purpose computer.

A modern NVIDIA GPU spends almost all of its silicon on **arithmetic
throughput**: many simple units running in lock-step on many independent
data elements. A single H100 SXM5 has 132 Streaming Multiprocessors
(SMs); each SM has 128 FP32 CUDA cores plus dedicated tensor cores.
That works out to roughly 16,896 FP32 arithmetic units on one chip,
versus 32–128 fully-featured x86 cores on a contemporary server CPU.

The trade: each unit on the GPU is much *simpler* than a CPU core. No
branch prediction. No speculation. No per-core L2. Almost no out-of-order
machinery. The GPU compensates by exposing massive parallelism to the
programmer and relying on the workload itself to keep all those units
busy. If your workload doesn't have enough independent work, the GPU
sits idle while a CPU would have squeezed more out of its single-thread
speed.

This is *why* a 32×32 matrix multiply is faster on a CPU and an
8192×8192 matrix multiply is faster on a GPU. The GPU only wins when
there is enough arithmetic, with low enough data-dependency, to
saturate its arithmetic units. The whole rest of this module gives
you the vocabulary to predict, before writing CUDA, whether the GPU
will win.

### The CPU/GPU silicon split, in one picture

```
+---------------------+      +-------------------------------+
| CPU core            |      | GPU SM (one of ~80–144)       |
|                     |      |                               |
| huge cache          |      | small cache (~256 KB shared)  |
| branch predictor    |      |   (combined with shared mem)  |
| OoO scheduler       |      | very small register file/lane |
| 8–16 wide pipeline  |      | 128 simple FP32 ALUs          |
| 1 thread retires    |      | 1 instruction across 32 lanes |
|                     |      |     (one warp) per cycle      |
+---------------------+      +-------------------------------+
   ~95% control logic           ~80% arithmetic logic
   ~5% arithmetic units         ~20% scheduler + cache
```

(Numbers are rough silhouettes; specific transistor budgets vary by
generation. But the *shape* of the trade is invariant.)

### SIMT — Single Instruction, Multiple Thread

NVIDIA's name for this execution model is **SIMT**: Single Instruction,
Multiple Thread. The unit of execution on a GPU is not a thread but a
**warp** of 32 threads, all of which execute the same instruction at
the same time. You write code as if each thread were independent —
that's the "Multiple Thread" part — and the hardware groups them into
warps for you.

When you read CUDA code, every thread really is writing per-lane code
for a SIMD machine that *looks like* 32 independent threads to the
programmer:

```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

Three things to notice:

1. There is no `for` loop over the array. Each thread handles one
   element. The hardware launches thousands of threads at once.
2. The `if (idx < n)` is a *guard* — when the array length isn't a
   multiple of the block size, some threads in the last block have
   nothing to do. Those threads execute the `if` and immediately exit.
3. The `idx` computation is the same expression for all 32 threads in
   a warp; only `threadIdx.x` differs. The warp issues one ADD, one
   MUL, one compare — across 32 lanes simultaneously.

The 32-threads-per-warp constant has held across every NVIDIA
architecture since G80 in 2006. It's the most important number on a
GPU. You will see it again in:

- **Occupancy calculations** (warps are the unit of scheduling, not
  threads).
- **Shuffle instructions** (`__shfl_sync` family — communicate within a
  warp without going through shared memory).
- **Tensor-core operand layouts** (typically defined in terms of
  16x16 or 8x8 fragments, sized to match warp lane counts).
- **Why divergent branches inside a warp hurt** (covered in Lecture 4).

### When a GPU does *not* help

Three workload shapes don't benefit from GPU acceleration:

- **Low arithmetic, high control flow.** Parsing a JSON file. Walking
  a linked list. Compiler code generation. The GPU has plenty of
  arithmetic units; it has very little branch-prediction logic per
  unit. If your hot path is `if/else/while`, the CPU wins.
- **Small problem size.** Kernel launch overhead is on the order of
  5–20 microseconds, and you need enough work to amortize it. A
  1024-element vector add is below the floor — by the time the
  launch returns, the CPU is done.
- **Lots of host-device data movement relative to compute.** PCIe
  Gen4 x16 tops out around 32 GB/s. If your kernel reads a tensor
  over PCIe, computes one FLOP per byte, and writes back, you are
  running at PCIe speed, not at GPU speed.

You will see all three failure modes in the field. They are the GPU
equivalent of "this `for`-loop is your bottleneck" on a CPU — the
first thing to check, and the most embarrassing thing to miss.

---

## The SM is the unit that matters

A GPU chip is hierarchical:

```
GPU
+-- GPC (Graphics Processing Cluster) x N
|   +-- TPC (Texture Processing Cluster) x M
|       +-- SM (Streaming Multiprocessor) x 2 (typical)
|           +-- 4 processing blocks (sub-partitions), each with:
|           |   +-- 32 FP32 CUDA cores
|           |   +-- Tensor Core(s) (architecture-specific)
|           |   +-- Warp scheduler + dispatch unit
|           |   +-- Register file slice (16K x 32-bit on Ampere)
|           +-- L1 cache / shared memory (combined, 128-256 KB)
|           +-- L0 instruction cache
+-- L2 cache (40-60 MB on H100; 4-6 MB on consumer cards)
+-- Memory controllers --> HBM2e / HBM3 / GDDR6X
```

You will see GPC and TPC mentioned in NVIDIA's whitepapers, but for
performance analysis they're mostly chip-floorplan concerns. The
units you reason about are SMs and the things inside an SM.

### Why "the SM is the unit"

When you launch a CUDA kernel, the GPU's hardware scheduler maps thread
blocks onto SMs. A block is the unit of *scheduling* — once placed on
an SM, a block stays there until it finishes. The block's threads share
that SM's shared memory, its register file, and its warp schedulers.

A single SM runs *multiple* thread blocks concurrently if resources
allow. Each block contributes some warps. The SM's warp schedulers
(one per sub-partition, four per SM on Ampere/Hopper) each pick one
ready warp per cycle and issue its next instruction. The arithmetic
units (FP32 CUDA cores, INT32 units, tensor cores, load/store units,
SFUs) consume the instruction.

This is why occupancy matters. The schedulers' ability to hide memory
latency depends on having multiple ready warps to pick from. If you
have one warp per SM, every memory load stalls everything; the FP32
units idle through the 400-cycle load.

### Sub-partition arithmetic for the curious

On Ampere/Hopper SMs, each of the four sub-partitions has:

- 32 FP32 CUDA cores (so 128 per SM)
- 16 INT32 cores (so 64 per SM)
- 1 tensor core (so 4 per SM)
- 1 warp scheduler + 1 instruction dispatch unit
- 16 KB of register file (so 64 KB = 65,536 32-bit registers per SM)

When you read "an Ampere SM has 64 KB of registers" what's really
happening is each sub-partition has 16 KB and warps are pinned to a
sub-partition at issue time. This matters for one edge case:
register-heavy kernels with very few warps can starve a single
sub-partition while the others sit idle. In practice that's a
tuning problem you'd discover with Nsight Compute, not something to
design around upfront.

---

## Walking the on-chip hierarchy top-down

Here's the path data takes from off-chip memory to an arithmetic unit
on a modern NVIDIA GPU, with the actual latencies and capacities
involved. Numbers are approximate Ampere-class values; specific
numbers shift across generations but the order-of-magnitude shape
holds.

```
HBM (off-chip)
  capacity: 40-141 GB             |  bandwidth: 1-3 TB/s
  latency: ~400-800 cycles        |  scope:    all SMs + host (via PCIe)
        |
        v
L2 cache (on-chip, GPU-global)
  capacity: 4-60 MB               |  bandwidth: hundreds of GB/s
  latency: ~200 cycles            |  scope:    all SMs
        |
        v
L1 / shared memory (on-chip, per-SM)
  capacity: 128-256 KB combined   |  bandwidth: ~TB/s (per SM)
  latency: ~20-30 cycles          |  scope:    per-block (shared) or hw-managed (L1)
        |
        v
Register file (per-SM, lane-private)
  capacity: 64K x 32-bit (256 KB) |  bandwidth: effectively infinite
  latency: ~1 cycle               |  scope:    per-thread
        |
        v
Arithmetic unit (FP32 CUDA core / tensor core / SFU)
```

A few invariants:

1. **Registers are free; everything else costs.** A read from a
   register is one cycle. A read from HBM is 400-800 cycles. The GPU
   hides that gap by switching warps — but only if there are other
   ready warps to switch to.

2. **Shared memory exists because L1 is not enough.** Shared memory
   and L1 share the same SRAM on modern NVIDIA architectures (Volta
   and later); you can configure the split. Use shared memory when
   you have data that is (a) used by multiple threads in the same
   block, and (b) reused enough to pay the cost of staging it from
   global memory. Matrix-multiply tiling is the canonical example.

3. **L2 helps on irregular access.** If you can't pre-stage data into
   shared memory because the access pattern is data-dependent, L2
   still gives you a 5–10× bandwidth improvement over HBM. The L2 is
   shared across all SMs, so it's also the level where one
   bandwidth-hungry kernel starts hurting its neighbors.

4. **HBM bandwidth is finite and shared.** All SMs pull from the same
   HBM stack(s). A kernel that achieves 80% of peak HBM bandwidth
   leaves only 20% for everyone else. In multi-tenant deployments
   that's a contention issue, not just an efficiency one.

You'll spend most of Lecture 3 inside this hierarchy. For now: when
you read about "memory pressure" or "bandwidth-bound" kernels, this
is the picture.

---

## Tensor cores: the second arithmetic engine

Starting with Volta (V100), every NVIDIA SM has a second class of
arithmetic unit — the **tensor core** — that performs a
*matrix-multiply-accumulate* (MMA) instruction in a single issue
slot. Tensor cores are what makes modern AI workloads feasible on
GPUs at all.

The mental model: a tensor core takes two small matrix operands and
one accumulator, and computes `D = A * B + C` in hardware, where the
matrices are (depending on instruction) 16×8×8, 16×8×16, or larger
fragments.

Concrete numbers for the GPUs we'll reference throughout the course
(all values from NVIDIA's datasheets — see `resources.md`):

| GPU | Peak FP32 (TFLOPS, dense) | Peak FP16 tensor (TFLOPS, dense) | Ratio |
|---|---|---|---|
| A100 SXM4 80GB | 19.5 | 312 | 16x |
| H100 SXM5 80GB | 67 | 989 | 14.8x |
| RTX 4090 | 82.6 | 330 | 4.0x |

Two things to internalize:

1. **The "tensor" path is 4–16× faster than the "FP32 CUDA core" path.**
   Any AI workload that *can* live in mixed precision (FP16, BF16,
   TF32, FP8 on Hopper) wants to be on tensor cores. Pure FP32 GEMM,
   in 2026, is leaving most of the chip idle.

2. **The ratio is highest on data-center parts (A100, H100).**
   Consumer GPUs like the RTX 4090 have many fewer tensor cores
   relative to FP32 cores because graphics workloads don't need them.
   This is a hardware market-segmentation decision, not a technical
   one.

Tensor cores are a Lecture 6 + Module 4 topic in depth. For now, when
you see "TF32" or "BF16" in a kernel listing, picture the tensor core
doing the work, not the FP32 cores.

### Why TFLOPS numbers can lie

NVIDIA's datasheets sometimes quote tensor-core throughput **with
sparsity**, meaning they assume 50% of weights are zero and the
hardware exploits that. The sparsity-on number is exactly 2× the
dense number. Whenever you compare TFLOPS across SKUs, verify which
mode is being quoted. In the table above all values are dense (no
sparsity).

You will catch one of these in Exercise 6.

---

## What changes (and doesn't) across generations

Every NVIDIA GPU has a **compute capability** (CC), a major.minor
version that tells you which architectural features are available.

| CC | Architecture | Generation marker | Notable additions |
|---|---|---|---|
| 6.0 / 6.1 | Pascal | P100, GTX 10-series | FP16 throughput, unified memory v2 |
| 7.0 / 7.5 | Volta / Turing | V100, T4, RTX 20-series | First tensor cores, independent thread scheduling |
| 8.0 / 8.6 | Ampere | A100, RTX 30-series, A10G | TF32, BF16, async copy, third-gen TC |
| 8.9 | Ada Lovelace | RTX 40-series, L4, L40 | Fourth-gen TC, FP8 (consumer) |
| 9.0 | Hopper | H100, H200 | FP8, transformer engine, thread block clusters, TMA |
| 10.0 | Blackwell | B100, B200 | FP4, fifth-gen TC, larger HBM |

Compute capability determines:

- Which tensor-core data types are usable (TF32 from Ampere; FP8 from
  Hopper; FP4 from Blackwell).
- Which shuffle / cooperative-groups instructions exist.
- Maximum threads per block, registers per thread, shared memory per
  block.
- Whether features like *async copy* (`cuda::memcpy_async`) or
  *Tensor Memory Accelerator* (TMA, H100+) are available.

The authoritative reference is **Appendix H ("Compute Capabilities")
of the CUDA C Programming Guide**. Memorize the rough shape of the
table above — when you read "this kernel needs CC 9.0," you should
know without looking that it requires a Hopper-class GPU and won't
run on an A100.

### Things that have NOT changed since G80

For all the architectural churn, several invariants have held since
2006:

- **Warp size is 32 threads.** Every NVIDIA GPU ever shipped.
- **Threads in a warp execute the same instruction at the same time.**
  This is the SIMT contract.
- **Memory hierarchy levels are register / shared / L2 / global**
  in that order. Capacities and bandwidths shift; the layering doesn't.
- **The SM is the scheduling unit.** Blocks → SMs, warps → schedulers,
  ever since the very first CUDA chip.

These are the assumptions you can build a mental model on. The CC
table tells you what the *current* hardware does on top of those
invariants.

---

## Common misconceptions

A short field guide to mistakes new GPU programmers make. Each one is
either painful debugging or "why isn't my kernel faster?" — both of
which you can save yourself by getting the model right up front.

### "More threads = faster"

**Wrong.** More threads is a *capacity* metric, not a speed metric.
Beyond enough warps to hide memory latency (covered in Lecture 4),
adding threads adds register pressure and can *reduce* occupancy
because each thread now gets fewer registers. The "256 threads per
block × 4 blocks per SM" sweet spot exists for a reason.

### "GPUs are just fast CPUs"

**Wrong.** A GPU is a *throughput* engine. A CPU is a *latency*
engine. They are not the same architecture with different knob
settings; they are different design philosophies. A workload that
benefits from one is usually mediocre on the other.

### "My kernel will scale linearly with more SMs"

**Mostly wrong.** Kernels scale linearly with SMs only if (a) there's
enough work to keep every SM busy and (b) the kernel isn't HBM-bandwidth-bound.
For memory-bound kernels (which most kernels are), doubling the SM
count doesn't help once HBM is already saturated.

### "If I'm at 100% occupancy I'm done optimizing"

**Wrong.** Occupancy is a *tool* for hiding latency; it is not the
goal. Vasily Volkov's 2010 paper "Better Performance at Lower
Occupancy" shows kernels that *intentionally* run at low occupancy
to free up registers for more instruction-level parallelism.
Occupancy is a diagnostic. The goal is throughput.

### "Tensor cores can speed up anything"

**Wrong.** Tensor cores compute matrix-multiply-accumulate. If your
workload doesn't decompose into MMA — for example, sorting,
reduction without partial sums, or pointer-chasing graphs — tensor
cores cannot help. You're still bottlenecked by FP32 / FP64 CUDA
cores and memory bandwidth.

---

## Worked example: counting FP32 lanes on an H100

You have an H100 SXM5 in front of you (or, more likely, an Nsight
Compute trace from one) and someone asks "how many simultaneous FP32
multiply-adds can this chip issue per cycle?"

Step 1: SM count. H100 SXM5: 132 SMs.

Step 2: FP32 CUDA cores per SM. Hopper SM has 128 FP32 CUDA cores
(4 sub-partitions × 32 cores each).

Step 3: Multiply.

```
FP32 multiply-adds per cycle = 132 * 128 = 16,896
```

Step 4 (the gotcha): each FP32 CUDA core issues one *FMA* per cycle,
which counts as **2 FLOPS** (one multiply + one add). So the FP32
FLOPS-per-cycle is:

```
peak FLOPS per cycle = 16,896 * 2 = 33,792
```

Step 5: multiply by clock. H100 SXM5 boost clock ≈ 1.98 GHz, so:

```
peak FP32 TFLOPS = 33,792 * 1.98e9 / 1e12
                ~= 66.9 TFLOPS
```

Match against the datasheet: 67 TFLOPS dense FP32. Done.

You'll do this for three GPUs in Exercise 2. The arithmetic is
trivial; the value is internalizing the *shape* of the calculation
so you can rederive it during a 20-minute design review.

---

## Practice problems

Do these before moving on. Answers at the bottom of the file.

**P1.** A100 SXM4: 108 SMs, 64 FP32 cores per SM, boost clock 1.41
GHz. Compute peak FP32 TFLOPS. Compare to the datasheet value of
19.5 TFLOPS.

**P2.** Sketch (in ASCII or on paper) the on-chip memory hierarchy
from register file to HBM, with the latency order of magnitude at
each level.

**P3.** Without scrolling back, name the architecture associated
with each compute capability: 7.0, 8.0, 8.9, 9.0.

**P4.** A kernel runs at 5% of peak FP32 throughput on an H100.
Without any further information, list three plausible reasons it's
not hitting more.

**P5.** A vendor pitches you a GPU with "1500 TFLOPS FP16 with
sparsity." You're modeling a workload that doesn't have structured
sparsity. What's the realistic peak you should plan around?

---

### Answers

**A1.** `108 * 64 * 1.41 * 2 / 1000 = 19.49 TFLOPS`. Matches the
datasheet (19.5) within rounding.

**A2.**

```
Register file (1 cycle)
  v
L1 / shared memory (~20-30 cycles)
  v
L2 cache (~200 cycles)
  v
HBM (~400-800 cycles)
```

**A3.** 7.0 → Volta (V100). 8.0 → Ampere (A100). 8.9 → Ada Lovelace
(RTX 40-series, L40). 9.0 → Hopper (H100).

**A4.** Plausible reasons (any three): (1) the kernel is memory-bound,
not compute-bound, so peak FP32 is the wrong ceiling; (2) the kernel
runs on tensor cores instead of FP32 cores; (3) occupancy is too
low to hide memory latency; (4) the launch is too small to fill
the chip; (5) branch divergence wastes half the warp; (6) the data
is being pulled over PCIe rather than from HBM.

**A5.** Halve it — 750 TFLOPS dense. The "with sparsity" number
assumes 2:4 structured sparsity, which is a model-side optimization
your workload may or may not support.

---

*End of Lecture 1.1. Next: Lecture 1.2 — CUDA Programming Model.*
