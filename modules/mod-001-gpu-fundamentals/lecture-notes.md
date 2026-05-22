# Module 1 — GPU Fundamentals: Lecture Notes

> Read this first. Then do the four exercises in `exercises/`. The quiz
> tests the same learning objectives in this document; nothing more.

---

## Lesson 1.1 — Why GPUs are fast on some workloads

A CPU and a GPU solve the same fundamental problem — execute instructions
on data — with opposite priorities.

A modern x86 CPU core spends most of its transistor budget on **latency**
of a single thread: out-of-order execution, branch prediction, large
caches, deep speculation. Its goal is to make one stream of dependent
instructions finish as fast as possible. A typical server CPU has
8–64 such cores.

A modern NVIDIA GPU spends most of its transistor budget on **throughput**:
many simple arithmetic units running in lock-step on many independent
data elements. A single H100 SXM5 has 132 Streaming Multiprocessors (SMs);
each SM has 128 CUDA cores for FP32, plus dedicated tensor cores for
matrix-multiply-accumulate. That is on the order of 16,000 FP32 units in
one chip versus 32–128 fully-featured x86 cores. The trade is: each unit
on the GPU is much simpler (no branch prediction, no speculation, much
smaller cache per unit), and the chip relies on the workload exposing
enough parallelism to keep them all busy.

This is why a 32×32 matrix multiply is faster on a CPU and a 8192×8192
matrix multiply is faster on a GPU. The GPU only wins when there is enough
arithmetic, with low enough data-dependency, to saturate its arithmetic
units.

### SIMT — Single Instruction, Multiple Thread

NVIDIA's name for this execution model is **SIMT**. The unit of execution
on a GPU is not a thread but a **warp** of 32 threads, all of which
execute the same instruction at the same time. (You write code as if
each thread were independent — that is the "Multiple Thread" part — and
the hardware groups them into warps for you. When you read about CUDA
threads, you are really writing per-lane code for a SIMD machine that
*looks* like 32 threads to you.)

The 32-threads-per-warp constant has held across every NVIDIA architecture
since G80 (2006). It is the most important number on a GPU. You will see
it again in occupancy calculations, in shuffle instructions, and in why
divergent branches inside a warp hurt performance.

### When a GPU does *not* help

Three workload shapes that the GPU's strengths do not save:

- **Low arithmetic, high control flow.** Parsing a JSON file. Walking a
  linked list. The GPU has plenty of arithmetic units; it has very little
  branch-prediction logic per unit.
- **Small problem size.** Kernel launch overhead is on the order of
  5–20 microseconds, and you need enough work to amortize it. A
  1024-element vector add is below the floor.
- **Lots of host-device data movement relative to compute.** PCIe Gen4 x16
  tops out around 32 GB/s. If your kernel reads a tensor over PCIe,
  computes one FLOP per byte, and writes back, you are running at PCIe
  speed, not at GPU speed.

You will see all three of these as failure modes in the field. They are
the GPU equivalent of "this for-loop is your bottleneck" on a CPU — the
first thing to check, and the most embarrassing thing to miss.

---

## Lesson 1.2 — NVIDIA GPU architecture

A GPU chip is hierarchical:

```
GPU
├── GPC (Graphics Processing Cluster) × N
│   └── TPC (Texture Processing Cluster) × M
│       └── SM (Streaming Multiprocessor) × 2 (typical)
│           ├── 4 processing blocks (sub-partitions), each with:
│           │   ├── ~32 FP32 CUDA cores
│           │   ├── Tensor Core(s) (architecture-specific)
│           │   ├── Warp scheduler + dispatch unit
│           │   └── Register file slice
│           ├── L1 cache / shared memory (combined, ~128–256 KB)
│           └── L0 instruction cache
├── L2 cache (~40–60 MB on H100; smaller on consumer cards)
└── Memory controllers → HBM2e / HBM3 / GDDR6X
```

The **SM** is the unit that matters for performance. When you launch a
kernel, the GPU's hardware scheduler maps thread blocks onto SMs.
A single SM runs multiple thread blocks concurrently if resources allow,
and inside each block multiple warps are scheduled. The warp scheduler in
each SM picks one ready warp per cycle (per sub-partition) and issues
its next instruction.

### Compute capability

Every NVIDIA GPU has a **compute capability** (CC), a major.minor version
that tells you which architectural features are available. Compute
capabilities you will run into:

| CC | Architecture | Generation marker |
|---|---|---|
| 6.0 / 6.1 | Pascal | P100, GTX 10-series |
| 7.0 / 7.5 | Volta / Turing | V100, T4, RTX 20-series |
| 8.0 / 8.6 | Ampere | A100, RTX 30-series, A10G |
| 8.9 | Ada Lovelace | RTX 40-series, L4, L40 |
| 9.0 | Hopper | H100, H200 |
| 10.0 | Blackwell | B100, B200 |

The CC determines which tensor-core data types you can use (TF32 from
Ampere; FP8 from Hopper), which shuffle instructions exist, the maximum
threads per block, the size of shared memory, and other limits. The
authoritative source is Appendix H of the *CUDA C Programming Guide*.

### Three GPUs we will refer to throughout the course

These come up enough in the exercises that it is worth pinning them now:

| Spec | A100 SXM4 80GB | H100 SXM5 80GB | RTX 4090 |
|---|---|---|---|
| Compute capability | 8.0 | 9.0 | 8.9 |
| SMs | 108 | 132 | 128 |
| FP32 CUDA cores | 6,912 | 16,896 | 16,384 |
| Boost clock (approx) | 1.41 GHz | 1.83 GHz | 2.52 GHz |
| Peak FP32 (TFLOPS, dense) | 19.5 | ~67 | ~82.6 |
| Peak FP16 Tensor (TFLOPS, dense) | 312 | 989 | 330 |
| Memory | 80 GB HBM2e | 80 GB HBM3 | 24 GB GDDR6X |
| Memory bandwidth | ~2.0 TB/s | ~3.35 TB/s | ~1.0 TB/s |

These numbers are from NVIDIA's published spec sheets (linked in
[`resources.md`](resources.md)). Exercise 2 has you derive them from the
underlying clock and core counts so you can do it for any GPU.

---

## Lesson 1.3 — GPU memory hierarchy

Performance work on a GPU is, more often than not, a memory problem.
The hierarchy from fastest / smallest / closest to slowest / largest /
furthest:

| Level | Capacity (per SM unless noted) | Latency | Bandwidth (per SM) | Scope |
|---|---|---|---|---|
| Registers | 64 K × 32-bit (256 KB) | ~1 cycle | TB/s | Per-thread |
| Shared memory / L1 | 128–256 KB combined | ~20–30 cycles | TB/s | Per-block |
| L2 cache | 40–60 MB (whole GPU) | ~200 cycles | hundreds of GB/s | All SMs |
| Global / HBM | 24–141 GB (whole GPU) | ~400–800 cycles | 1–3 TB/s | All SMs + host |

A few rules to internalize:

1. **Registers are free; everything else costs.** A read from a register
   is essentially the latency of an arithmetic instruction. A read from
   HBM, even when coalesced, is ~500 cycles of latency. The GPU hides
   that latency by switching to another ready warp, but only if there is
   another ready warp.

2. **Shared memory exists because L1 is not enough.** On most NVIDIA
   architectures, shared memory and L1 share the same SRAM and you can
   split it by configuration (e.g., on Ampere, 192 KB combined,
   configurable allocations between L1 and shared). Use shared memory
   when you have data that is (a) used by multiple threads in the same
   block, and (b) reused enough to pay the cost of copying it from
   global memory. Matrix multiply tiling is the canonical example.

3. **L2 is your friend on irregular access patterns.** It is shared
   across all SMs and is much larger than L1 / shared. If you cannot
   pre-stage data into shared memory because the access pattern is
   data-dependent, the L2 still gives you a 5–10× bandwidth improvement
   over going to HBM.

4. **HBM bandwidth is finite and shared.** All SMs pull from the same
   HBM stack(s). A kernel that achieves 80% of peak HBM bandwidth
   leaves only 20% for everyone else; in a multi-tenant deployment
   that is a contention issue, not just an efficiency one.

### Coalescing — the single most important memory-pattern rule

A warp of 32 threads issues a memory instruction together. If those
threads access **contiguous, aligned** 4-byte words, the hardware
collapses 32 separate accesses into one 128-byte memory transaction.
That is "coalesced." If they access scattered locations, the hardware
issues up to 32 separate transactions.

The performance gap between coalesced and uncoalesced access on the
same kernel is typically 8–32× — large enough that, in practice,
"is the access coalesced?" is the first question to ask of any
memory-bound kernel.

Exercise 4 has you place real kernels on a roofline and one of them
fails coalescing on purpose; you should be able to predict that without
running the profiler.

---

## Lesson 1.4 — Threads, blocks, grids

When you launch a CUDA kernel, you specify a **grid** of **blocks**,
each of which contains some **threads**. Inside the kernel, a thread
finds itself in the grid using built-in variables:

```python
# Conceptual Python equivalent of the CUDA built-ins.
# In real CUDA C++ these are blockIdx.x, blockDim.x, threadIdx.x, etc.
def linear_thread_id_1d(block_idx, block_dim, thread_idx):
    return block_idx * block_dim + thread_idx
```

Concrete launch in CUDA C++:

```cpp
// 1024-element vector add; one thread per element.
constexpr int N = 1024;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

vector_add<<<BLOCKS, THREADS_PER_BLOCK>>>(a, b, c, N);
```

The hardware constraints (worth memorizing for Ampere and later):

| Constraint | Limit |
|---|---|
| Max threads per block | 1024 |
| Max threads per warp | 32 (always) |
| Max warps per SM | 64 |
| Max thread blocks per SM | 32 |
| Max grid dim X | 2^31 − 1 |
| Max grid dim Y / Z | 65,535 |
| Max shared memory per block (configurable) | 100–227 KB depending on arch |

These are *upper bounds*. The achievable values depend on per-thread
register pressure and per-block shared memory, which is the topic of
occupancy (Lesson 1.6).

### Choosing a block size

A practical default is 128 or 256 threads per block. The reasoning:

- Must be a multiple of 32 (warp size) or you waste lanes.
- Must be large enough to give the warp scheduler something to switch to
  while warps are stalled on memory. 4 warps = 128 threads is a common
  floor.
- Must be small enough that several blocks fit on one SM (so the
  scheduler has multiple blocks to pick from). On Ampere, max 32 blocks
  per SM × max 64 warps per SM means a block of 64 threads (2 warps)
  hits the block-count limit before it hits the warp-count limit.

The "256 threads, ~8 warps, ~4 blocks per SM" combination is a strong
starting point and is what cuBLAS and cuDNN use for most of their
kernels. Tune from there based on profiling.

---

## Lesson 1.5 — Warp execution

A warp of 32 threads executes one instruction per cycle (per sub-partition
of the SM). Three behaviors of the warp model are responsible for most of
the surprises new GPU programmers encounter.

### 1. Latency hiding

When a warp issues a load from global memory, that load takes hundreds
of cycles. The warp scheduler does not stall waiting; it parks the
issuing warp and picks another warp that has an instruction ready. As
long as there are **enough warps in flight on the SM** (high occupancy)
and **enough independent work between dependent memory ops** (ILP), the
arithmetic units stay busy.

This is why occupancy matters: not because a low-occupancy kernel is
inherently bad, but because a kernel whose only way to hide memory
latency is to switch to other warps will run badly at low occupancy.

### 2. Branch divergence

If 32 threads in a warp take different sides of an `if`, the warp
executes both sides serially with predication: threads on the false
branch sit idle while the true branch runs, then vice versa.

```cpp
// BAD: half the warp is idle for each branch.
if (threadIdx.x < 16) {
    // 16 threads work; 16 idle.
    a[idx] = expensive_thing(...);
} else {
    // 16 threads work; 16 idle.
    a[idx] = other_expensive_thing(...);
}
```

The performance cost is up to 2× for a single binary divergence and
keeps multiplying for nested divergence. The fix is either to make the
condition *warp-aligned* (so all 32 threads take the same branch — no
divergence) or to refactor so both branches are cheap.

### 3. Warp-level primitives

Threads in a warp can communicate **without going through shared memory**
using shuffle instructions: `__shfl_sync`, `__shfl_down_sync`,
`__shfl_up_sync`, `__shfl_xor_sync`. These take a few cycles versus
~25 cycles for a shared-memory round-trip. Warp-level reductions and
prefix-sums are dramatically faster with shuffles than with shared
memory.

You will write a shuffle-based reduction in Module 2.

---

## Lesson 1.6 — Performance metrics that matter

There are exactly three metrics worth caring about on a first pass at
any GPU kernel. Everything else is a refinement of one of these.

### 1. Arithmetic intensity (AI)

```
AI = FLOPS performed / bytes moved from DRAM
```

Units: FLOPS per byte. This is the property of the **algorithm**, not
the implementation. Examples:

| Kernel | Approximate AI |
|---|---|
| `c[i] = a[i] + b[i]` (vector add, FP32) | 1/12 = 0.083 FLOP/byte |
| `c[i] = a[i] * b[i] + c[i]` (FMA vector) | 2/16 = 0.125 FLOP/byte |
| Dense matrix multiply, M=N=K=4096, FP32 | ~341 FLOP/byte |
| Self-attention (sequence 2048, d=128) | ~30–60 FLOP/byte |

The reason this is the most important number: it tells you which side of
the roofline the workload lives on, *before* you implement it.

### 2. Roofline model

The roofline model (Williams, Waterman, Patterson, 2009) gives the
ceiling on achievable performance for a kernel as a function of its
arithmetic intensity.

```
attainable_throughput(AI) = min(peak_FLOPS, peak_BW * AI)
```

The "ridge point" is `AI_ridge = peak_FLOPS / peak_BW`. Below the ridge
point, the kernel is **memory-bound** — increasing FLOP throughput on
that kernel cannot help; you need to either reduce bytes moved or move
them faster. Above the ridge point, the kernel is **compute-bound** —
optimizing memory access cannot help; you need to reduce FLOPS or use
faster math units (e.g., tensor cores).

For an A100:

- Peak FP32 ≈ 19.5 TFLOPS
- Peak HBM bandwidth ≈ 2000 GB/s
- Ridge point ≈ 19,500 / 2000 ≈ 9.75 FLOP/byte

So FP32 vector add (AI = 0.083) is *deeply* memory-bound on an A100 —
about two orders of magnitude below the ridge. A dense FP32 GEMM at
size 4096 (AI ≈ 341) is compute-bound. Self-attention sits in between
and behaves differently depending on sequence length, which is the
intuition behind Flash Attention (Module 4).

Exercise 4 has you draw this for A100 and place six real kernels.

### 3. Occupancy

Occupancy is the ratio of *active warps per SM* to the *maximum warps
per SM*. On Ampere, max is 64 warps, so 32 active warps = 50% occupancy.

Occupancy is bounded from above by three resources, and the smallest
upper bound wins:

- **Threads per block × blocks per SM** — capped at 1024 × 32 = 2048
  threads = 64 warps. Easy to hit.
- **Registers per thread × threads per SM** — capped at the SM's register
  file size. An Ampere SM has 65,536 32-bit registers. If your kernel
  uses 64 registers per thread, you can run 65,536 / 64 = 1024 threads
  = 32 warps = 50% occupancy.
- **Shared memory per block × blocks per SM** — capped at the SM's
  shared memory. An Ampere SM has up to 164 KB configurable. If your
  block uses 48 KB shared memory, you can run at most 164 / 48 = 3
  blocks per SM.

A kernel is "register-limited" if registers are the binding constraint,
"shared-memory-limited" if shared memory is, and so on.

**Important:** high occupancy is not the goal. It is *one* tool for
hiding latency. A kernel with low occupancy but high instruction-level
parallelism can outperform a high-occupancy kernel — Volkov's 2010
paper "Better Performance at Lower Occupancy" is the canonical
counterexample and is in `resources.md`. Use occupancy as a diagnostic,
not a target.

Exercise 3 has you reimplement the occupancy calculation from scratch.

---

## Wrap-up

You now have the mental model for the rest of the curriculum. The next
module starts writing CUDA, but every optimization decision there will
reduce to one of:

- "Is this memory-bound or compute-bound?" → roofline / arithmetic
  intensity
- "Are these threads going to access memory together?" → coalescing
- "Can the warp scheduler hide this load?" → occupancy
- "Does this warp diverge?" → branch divergence

If the four exercises feel solid, you are ready for Module 2.

---

*Last revised: 2026-05-22 — initial build.*
