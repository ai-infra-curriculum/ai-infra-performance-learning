# Lecture 1.4 — Warps and Occupancy

> Lesson 4 of 6. Plan for ~75 minutes. This is the lecture that
> ties the architecture (Lecture 1) and memory hierarchy (Lecture 3)
> together into a single answer to "why isn't my kernel faster?"

## Table of Contents

1. [Latency hiding — the central mechanism](#latency-hiding--the-central-mechanism)
2. [Occupancy: definition](#occupancy-definition)
3. [The three resource caps](#the-three-resource-caps)
4. [Worked occupancy calculation](#worked-occupancy-calculation)
5. [Branch divergence and the warp tax](#branch-divergence-and-the-warp-tax)
6. [Warp-level primitives — shuffles, votes, ballots](#warp-level-primitives--shuffles-votes-ballots)
7. [When low occupancy is fine](#when-low-occupancy-is-fine)
8. [Worked example: tuning a kernel for occupancy](#worked-example-tuning-a-kernel-for-occupancy)
9. [Practice problems](#practice-problems)

---

## Latency hiding — the central mechanism

The single most important behavior of a GPU is **latency hiding**.

When a warp issues a load from global memory, that load takes
400–800 cycles to return. On a CPU, you'd stall the pipeline and
wait. On a GPU, the warp scheduler does something different:

1. Mark the issuing warp as "stalled, waiting on memory."
2. Pick a *different* warp on the same sub-partition that has an
   instruction ready to issue.
3. Issue that warp's next instruction.
4. Continue until the original warp's data returns; then it becomes
   ready again.

If the SM has enough ready warps, the arithmetic units stay busy
through the load latency. If it doesn't, the arithmetic units
idle — and that's what "low occupancy hurts memory-bound kernels"
actually means.

```
Time --->
warp A: LOAD....stall (400 cy)......result here, do ADD --
warp B:        do MUL, do ADD, LOAD....stall.................result
warp C:                     do FMA, FMA, FMA, LOAD....stall......
warp D:                                    LOAD....stall.........

SM arithmetic units busy almost every cycle as long as
SOME warp has work to do.
```

This is why a kernel with 32 warps per SM (50% occupancy on Ampere)
often outruns one with 16 warps per SM (25% occupancy): more warps
in flight, more chances to hide a load.

But — and this is the part that surprises new GPU programmers —
*more warps is not always better*. Adding warps costs registers.
At some point you tax register pressure so much that each thread
has fewer registers and the kernel spills to local memory, which
is a 500-cycle HBM trip *every time*. The optimal occupancy
depends on the kernel.

### The metric you should think about: latency × throughput

There's a deeper way to look at this, from Little's Law:

```
Concurrency = Latency × Throughput
```

For an SM to keep its arithmetic units busy:

```
(active warps) × (instructions per warp issue) >= (latency to fill)
```

On Ampere/Hopper an SM can issue ~4 warp instructions per cycle and
the memory latency is ~400 cycles. So you need on the order of
`400 / 4 ≈ 100` warp-instruction "slots" worth of independent work
in flight. That's either:

- Lots of warps (high occupancy), each with low instruction-level
  parallelism (ILP).
- Fewer warps (lower occupancy), each with more ILP.

Both can saturate the chip. The Volkov "Better Performance at Lower
Occupancy" paper exists because for FP-heavy kernels with high ILP,
you can hit peak with as little as 12.5% occupancy. For
memory-latency-dominated kernels, you typically need 50%+.

---

## Occupancy: definition

Occupancy is the ratio of **active warps per SM** to the **maximum
warps per SM**.

```
occupancy = active_warps_per_sm / max_warps_per_sm
```

On Ampere/Hopper, `max_warps_per_sm = 64`. So 32 active warps =
50% occupancy.

A kernel's *theoretical* occupancy is the maximum it can achieve
given its resource usage, independent of how many blocks the chip
actually launches. The CUDA Occupancy API and the spreadsheet
calculator compute this from:

- Block size (threads per block)
- Registers per thread
- Shared memory per block

The runtime *achieved* occupancy can be lower (if the grid doesn't
have enough blocks to fill the chip, or some SMs finish earlier
than others). For tuning, theoretical occupancy is what you target;
achieved occupancy is what Nsight Compute reports.

---

## The three resource caps

Three resources each give you a *cap* on blocks per SM. The smallest
cap wins.

### 1. Warp / thread caps

```
warps_per_block = ceil(threads_per_block / 32)
max_blocks_per_sm_warp_capped = max_warps_per_sm / warps_per_block
```

On Ampere/Hopper with `max_warps = 64`:

- 256 threads/block = 8 warps. Cap = 64 / 8 = 8 blocks.
- 128 threads/block = 4 warps. Cap = 64 / 4 = 16 blocks.
- 1024 threads/block = 32 warps. Cap = 64 / 32 = 2 blocks.

Plus a hard cap on blocks per SM regardless of warps:

```
max_blocks_per_sm = 32   (on Ampere/Hopper; was 16 on older arches)
```

For very small blocks (e.g., 32 threads = 1 warp), `64 / 1 = 64`
exceeds `max_blocks_per_sm = 32`, so blocks become the binding
constraint.

### 2. Register cap

```
registers_per_block = threads_per_block * registers_per_thread
max_blocks_per_sm_reg_capped = register_file_size / registers_per_block
```

On Ampere with `register_file_size = 65,536`:

- 256 threads × 32 regs = 8,192. Cap = 65,536 / 8,192 = 8 blocks.
- 256 threads × 64 regs = 16,384. Cap = 65,536 / 16,384 = 4 blocks.
- 256 threads × 96 regs = 24,576. Cap = 65,536 / 24,576 = 2 blocks.

Add: the per-thread register cap itself is 255 on most modern
archs. Spilling above 64 per thread becomes common.

### 3. Shared memory cap

```
max_blocks_per_sm_shared_capped = shared_mem_per_sm / shared_mem_per_block
```

If `shared_mem_per_sm = 164 KB` (A100 with carveout):

- 4 KB per block: cap = 41 blocks (but max_blocks = 32, so 32).
- 12 KB per block: cap = 13 blocks.
- 48 KB per block: cap = 3 blocks.
- 96 KB per block: cap = 1 block.

If a block uses 0 shared memory, the shared-memory cap is
effectively infinity (no constraint).

### Putting it together

```
active_blocks_per_sm = min(
    warps_cap,
    blocks_cap,
    register_cap,
    shared_mem_cap,
)
active_warps_per_sm = active_blocks_per_sm * warps_per_block
occupancy = active_warps_per_sm / max_warps_per_sm
```

The **binding resource** — the one whose cap is the smallest — is the
one to optimize. If registers are binding, reducing registers per
thread (e.g., by splitting the kernel) raises occupancy. If shared
memory is binding, reducing shared per block raises occupancy.

You'll implement this exact calculation in Exercise 3.

---

## Worked occupancy calculation

A real kernel from a Module 4 problem set. Specs:

- 256 threads per block (= 8 warps)
- 40 registers per thread
- 8 KB of shared memory per block

Target: Ampere A100. SM resources:
- `max_warps_per_sm = 64`
- `max_blocks_per_sm = 32`
- `register_file_size = 65,536` (32-bit)
- `shared_mem_per_sm = 164 KB`

**Step 1: Warps per block.**

```
warps_per_block = ceil(256 / 32) = 8
```

**Step 2: Each resource cap.**

```
warps_cap     = 64 / 8 = 8 blocks
blocks_cap    = 32 blocks
register_cap  = 65536 / (256 * 40) = 65536 / 10240 = 6 blocks (truncated)
shared_cap    = 164 KB / 8 KB = 20 blocks (truncated)
```

**Step 3: Minimum cap.**

```
min(8, 32, 6, 20) = 6 blocks
```

Register cap is binding. The kernel can fit at most 6 blocks per SM.

**Step 4: Active warps and occupancy.**

```
active_warps = 6 * 8 = 48
occupancy = 48 / 64 = 0.75 = 75%
```

**Step 5: What to optimize.**

The kernel is **register-limited**. To raise occupancy, reduce
registers per thread. Options:

- Refactor: break a complex inner loop into two smaller kernels.
- Use `--maxrregcount=N` at compile time to force a lower register
  count (the compiler then spills the rest — careful, can be
  counterproductive).
- Restructure to use less per-thread state (e.g., recompute instead
  of cache).

Going from 40 → 32 registers gives:

```
new register_cap = 65536 / (256 * 32) = 8 blocks
min(8, 32, 8, 20) = 8 -> 64 active warps -> 100% occupancy
```

100% theoretical occupancy. Whether that actually speeds up the
kernel depends on whether memory latency was the real bottleneck
to begin with.

You're going to do this calculation many times in this course.
The pattern *never changes*; only the SM specs and the kernel
inputs do.

---

## Branch divergence and the warp tax

A warp executes one instruction per cycle, across all 32 lanes.
What happens when the 32 threads take different sides of an `if`?

```cpp
if (threadIdx.x < 16) {
    // Lanes 0-15 want to execute A.
    a_path();
} else {
    // Lanes 16-31 want to execute B.
    b_path();
}
```

The hardware **predicates** the execution. It runs the `if` branch
first, with lanes 0–15 enabled and lanes 16–31 idle (their writes
discarded). Then it runs the `else` branch, with lanes 16–31 enabled
and lanes 0–15 idle. Total cost: time(A) + time(B), vs the
non-divergent case of time(A) or time(B) alone.

```
Without divergence (all 32 lanes take the same branch):
  cycle: 1  2  3  4
  L0-31: A  A  A  A   <- A executes in 4 cycles, all lanes active
                          (Or only B; either way 4 cycles)

With divergence (16-16 split):
  cycle: 1  2  3  4  5  6  7  8
  L0-15: A  A  A  A  .  .  .  .   <- A runs, B-lanes predicated off
  L16-31:.  .  .  .  B  B  B  B   <- B runs, A-lanes predicated off
                                      Total cost: 8 cycles (2x slowdown).
```

For nested branches, the cost compounds. A 4-way nested `if/else`
that splits the warp 4 ways can be a 4× slowdown.

### Important nuance: across-warp branching is *free*

If the branch condition only varies *across* warps, not *within* a
warp, there's no divergence. Each warp follows its own branch:

```cpp
// FINE: all 32 threads in a warp share the same blockIdx.x value.
if (blockIdx.x % 2 == 0) {
    even_path();
} else {
    odd_path();
}

// FINE: all 32 threads share the same warp-aligned bit of threadIdx.x.
if (threadIdx.x / 32 == 0) {   // first warp in the block
    first_warp_path();
} else {
    other_warps_path();
}

// DIVERGES: threadIdx.x % 2 differs lane to lane.
if (threadIdx.x % 2 == 0) {
    even_thread_path();
} else {
    odd_thread_path();
}
```

The litmus test: *does this condition produce different values for
different lanes in the same warp?* If no, you're fine. If yes,
you have divergence.

### Common divergence patterns

- **Conditional masking** (`if (idx < n)` boundary check): only one
  warp diverges (the last). Tiny cost. Always do this.
- **Inner-loop conditional**: bad. `if (input[i] > threshold) ...`
  inside a hot loop typically halves throughput.
- **Per-row conditionals on row-aligned data**: usually fine if the
  rows align with warp boundaries (which depends on layout).
- **Hash table / lookup table chasing**: typically diverges hard.
  Each lane chases a different chain length.

### The fix

Three options, in order of how often you reach for them:

1. **Refactor to make the condition warp-aligned.** E.g., sort the
   data so that threads in the same warp take the same branch.
2. **Compute both sides; use predication explicitly.** If both
   branches are cheap, just compute both and select with a mask.
   The hardware was going to predicate anyway.
3. **Use warp-vote intrinsics (`__any_sync`, `__all_sync`,
   `__ballot_sync`)** to coordinate the warp around the
   conditional.

You'll see all three in Module 3.

You'll compute the cost of divergence yourself in **Exercise 5**.

---

## Warp-level primitives — shuffles, votes, ballots

Threads in a warp can communicate **without going through shared
memory** using warp-level intrinsics. These are typically just a
few cycles versus ~25 cycles for a shared-memory round trip.

### __shfl_sync — read a value from another lane

```cpp
// Each thread t reads the value of `val` held by lane (t + offset).
float partner = __shfl_down_sync(0xFFFFFFFF, val, offset);
```

Used in warp-level reductions:

```cpp
__device__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;   // lane 0 has the warp sum
}
```

The five lines above compute a 32-element reduction in ~5 cycles
total — log2(32) = 5 shuffles. Shared-memory equivalent: ~50 cycles.

### __ballot_sync — which lanes pass a predicate?

```cpp
// Returns a 32-bit mask where bit i = (predicate true for lane i).
uint32_t mask = __ballot_sync(0xFFFFFFFF, predicate);
int count = __popc(mask);   // number of lanes that passed
```

Used for stream-compaction patterns: "for each lane that survived a
filter, do something coherent across the surviving lanes."

### __any_sync / __all_sync — reductions of a predicate

```cpp
if (__any_sync(0xFFFFFFFF, condition)) {
    // At least one lane sees condition=true. Coordinate.
}
if (__all_sync(0xFFFFFFFF, condition)) {
    // All 32 lanes see condition=true.
}
```

Useful for early termination of loops: "if no thread in this warp
has more work, break."

### The `_sync` suffix and the mask

Pre-Volta (CC < 7.0), warp intrinsics were `__shfl`, `__ballot`,
etc. Post-Volta (Independent Thread Scheduling), they're
`__shfl_sync`, `__ballot_sync`, etc., and they take a *mask* of
participating lanes as the first argument. `0xFFFFFFFF` means
"all 32 lanes." Use this mask unless you're inside a divergent
section where you explicitly want a subset.

If your CUDA code compiles for CC 7.0+ (anything from Volta
onward, which is everything you care about), use the `_sync`
variants. The unsuffixed versions are deprecated.

You'll write a shuffle-based reduction in Module 2.

---

## When low occupancy is fine

There's a common pathology: a CUDA programmer profiles, sees
"38% occupancy," and decides the goal is to push it higher. This
is usually wrong.

**High occupancy is one tool for hiding latency. It is not the
goal.** The goal is throughput.

A kernel can hit peak with low occupancy if:

1. **It has high instruction-level parallelism.** Multiple
   independent FMAs per loop iteration give the chip something to
   issue without needing other warps.
2. **Each warp does a lot of independent arithmetic between memory
   ops.** If a warp can issue 50 FMAs while waiting for one load to
   return, occupancy stops mattering.
3. **It's compute-bound on tensor cores.** Tensor-core throughput is
   high enough that even one or two warps per SM can saturate the
   pipe.

Volkov 2010 ("Better Performance at Lower Occupancy") shows GEMM
kernels achieving >70% of peak at 12.5% occupancy. The kernels
look weird (deeply unrolled, very high registers per thread) but
they work.

The general rule:

- For **memory-bound** kernels (low arithmetic intensity, lots of
  loads), **higher occupancy helps**. You need other warps to
  switch to during loads.
- For **compute-bound** kernels with high ILP, occupancy past
  ~25% is usually irrelevant.
- For **tensor-core-bound** kernels, occupancy can be quite low.

If you're profiling and you see low occupancy + low achieved
throughput, occupancy might be the problem. If you see low
occupancy + high achieved throughput, you're already done —
move on.

---

## Worked example: tuning a kernel for occupancy

You write a stencil kernel for a 3D laplacian update. First version:

```
threads_per_block: 512 (= 16 warps)
registers per thread: 72
shared memory per block: 32 KB (a tile + halo)
```

On A100:

```
warps_per_block = 16
warps_cap     = 64 / 16 = 4 blocks
blocks_cap    = 32 blocks
register_cap  = 65536 / (512 * 72) = 1 block  <- BINDING
shared_cap    = 164 / 32 = 5 blocks
```

You're at **1 block × 16 warps = 16 warps = 25% occupancy**.
You profile: kernel achieves 380 GB/s on a 2000 GB/s chip. That's
19% of peak HBM bandwidth — memory-bound and far from saturated.
Occupancy probably matters here.

**Tuning round 1:** drop to 256 threads/block, same per-thread regs.

```
warps_per_block = 8
warps_cap     = 64 / 8 = 8
register_cap  = 65536 / (256 * 72) = 3 blocks
shared_cap    = 164 / 32 = 5 (but the halo is sized for 512 threads;
                              you also drop shared to 16 KB)
        new shared_cap = 164 / 16 = 10 blocks
min(8, 32, 3, 10) = 3 blocks
active_warps = 3 * 8 = 24 -> 37.5% occupancy
```

Better. Bandwidth jumps to 720 GB/s (35% of peak). But you're still
register-binding.

**Tuning round 2:** restructure to use 48 registers per thread
(more recomputation, less per-thread caching).

```
register_cap  = 65536 / (256 * 48) = 5 blocks
min(8, 32, 5, 10) = 5 blocks
active_warps = 5 * 8 = 40 -> 62.5% occupancy
```

Bandwidth: 1.4 TB/s (70% of peak). Diminishing returns from here.

**Tuning round 3:** drop registers further — to 32.

```
register_cap  = 65536 / (256 * 32) = 8 blocks
min(8, 32, 8, 10) = 8 -> 64 warps -> 100% occupancy
```

Bandwidth: 1.5 TB/s (75% of peak). Tiny gain. Diminishing returns.
Tuning round 2 was the sweet spot; round 3 didn't help much because
the chip was already memory-bound and the extra warps had less and
less marginal benefit.

The lesson: **measure first, occupancy second.** If occupancy
matters for your kernel, you'll see it in the profile. If you
push occupancy to 100% and bandwidth doesn't move, you've found
the wrong knob.

---

## Practice problems

**P1.** A kernel uses 800 bytes of shared memory per block and 25
registers per thread, with 128 threads per block. On an A100 with
`max_warps=64`, `max_blocks=32`, `regs=65536`, `shared=164KB` — what
is the limiting resource and what is the occupancy?

**P2.** Your kernel has a divergent branch: 60% of warps take the
`if`, 40% take the `else`. Both branches take 100 cycles. What is
the average cost per warp?

**P3.** Why does `if (idx < n)` for a boundary check have negligible
divergence cost?

**P4.** A kernel achieves 90% of peak throughput at 25% occupancy.
You force it to 100% occupancy by lowering register count. The
throughput drops to 70% of peak. What happened?

**P5.** You want to compute the sum of 32 floats held one per lane
in a warp, getting the result into lane 0. Sketch the algorithm
using `__shfl_down_sync`.

---

### Answers

**A1.**
```
warps/block = 4
warps_cap   = 64 / 4 = 16
blocks_cap  = 32
reg_cap     = 65536 / (128 * 25) = 65536 / 3200 = 20
shared_cap  = 164 KB / 800 B = ~205 (but capped at 32 by blocks_cap)
min(16, 32, 20, 205) = 16 -> warps is binding
active_warps = 16 * 4 = 64 -> 100% occupancy
```

**A2.** The cost depends on whether divergence is *within* a warp or
*across* warps. If 60% of *warps* take the `if` and the rest take
the `else`, that's no within-warp divergence — each warp takes one
branch only, cost = 100 cycles per warp. If the wording meant that
within each warp 60% of lanes go one way, then *every* warp pays
both branches: 200 cycles per warp. (Trick question: read the
wording.)

**A3.** Only the *last* block in the grid has a partial warp where
some threads have `idx >= n` and others don't. The cost is borne
once per launch, not once per iteration. For any non-trivial grid
size, that's a rounding error.

**A4.** Reducing register count likely caused register spills to
local memory (HBM). Even though occupancy went up, the spills
added 500-cycle round trips on every spilled variable access,
overwhelming the latency-hiding benefit of more warps. This is
the canonical "high occupancy hurts" case.

**A5.**
```cpp
__device__ float warp_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;  // lane 0 has the sum; other lanes hold partial sums
}
```

5 instructions, ~5 cycles. Same algorithm as a binary-tree reduction.

---

*End of Lecture 1.4. Next: Lecture 1.5 — The Roofline Model.*
