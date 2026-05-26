# Lecture 1.3 — CUDA Memory Hierarchy

> Lesson 3 of 6. Plan for ~90 minutes. This is the lecture you'll
> reference most often for the rest of the curriculum. Performance
> work on a GPU, in practice, is mostly a memory problem; the
> intuitions in this file are the ones you'll lean on.

## Table of Contents

1. [The big picture, with numbers](#the-big-picture-with-numbers)
2. [Registers — where the actual work happens](#registers--where-the-actual-work-happens)
3. [Shared memory — software-managed cache](#shared-memory--software-managed-cache)
4. [L1 vs shared: same SRAM, two modes](#l1-vs-shared-same-sram-two-modes)
5. [L2 — the GPU-wide cache](#l2--the-gpu-wide-cache)
6. [Global memory (HBM/GDDR) and coalescing](#global-memory-hbmgddr-and-coalescing)
7. [Bank conflicts](#bank-conflicts)
8. [Constant and texture memory](#constant-and-texture-memory)
9. [Worked example: tiled matmul, byte by byte](#worked-example-tiled-matmul-byte-by-byte)
10. [Practice problems](#practice-problems)

---

## The big picture, with numbers

The on-chip memory hierarchy on a modern NVIDIA GPU, fastest to
slowest:

| Level | Per-thread / per-block / chip-wide? | Capacity | Latency | Bandwidth | Scope |
|---|---|---|---|---|---|
| Registers | per-thread | 64 K × 32-bit = 256 KB per SM | ~1 cycle | "infinite" (one per cycle) | One thread |
| Shared memory | per-block | 128–228 KB per SM (config) | ~20–30 cycles | TB/s per SM | All threads in block |
| L1 cache | per-SM | shares SRAM with shared mem | ~20–30 cycles | TB/s per SM | All threads in block |
| L2 cache | GPU-wide | 4–60 MB (chip) | ~200 cycles | hundreds of GB/s | All SMs |
| Global (HBM/GDDR) | GPU-wide | 24–141 GB (chip) | ~400–800 cycles | 1–3 TB/s | All SMs + host (via PCIe) |
| Host memory (over PCIe) | host-wide | 100s GB | ~thousands of cycles | ~32 GB/s (Gen4 x16) | CPU + GPU via DMA |

Approximate numbers; specific values shift by generation. Memorize
the *order of magnitude* (latency in single cycles, 10s, 100s,
100s+, 1000s) — that's enough to reason about kernels.

Three invariants:

1. **Each level is roughly 10× slower than the previous one.** Register
   → shared is ~30×. Shared → L2 is ~10×. L2 → HBM is ~3×. HBM →
   host is another order of magnitude. The hierarchy isn't there for
   decoration; the steps are real and large.

2. **Higher levels are *smaller*.** You can't keep everything in
   shared memory. Tiling is the technique that says "let's keep a
   *window* in shared, slide it across the data."

3. **The compiler picks registers; you pick shared memory.** You
   declare `__shared__` and `__device__` explicitly. The compiler
   decides which locals live in registers. Hand-coded shared memory
   is the most common optimization technique in GPU programming.

---

## Registers — where the actual work happens

Every variable that lives long enough to be reused, but isn't
declared `__shared__` or `__device__`, lives in **registers**.

```cpp
__global__ void k(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // register
    if (idx < n) {
        float val = x[idx];      // load into register
        val = val * 2.0f + 1.0f;  // register-register arithmetic
        y[idx] = val;             // store from register
    }
}
```

The register file is *per-SM*, not per-thread. On Ampere/Hopper that's
65,536 32-bit registers per SM (so 64 KB total, split across 4
sub-partitions of 16 KB each).

When you launch a block of 256 threads, those 256 threads share the
SM's register file with other blocks running on the same SM. If your
kernel uses 32 registers per thread:

```
threads per block * registers per thread = 256 * 32 = 8,192
blocks per SM (register-limited) = 65,536 / 8,192 = 8
warps per SM = 8 * 8 = 64 (full occupancy)
```

If your kernel uses 96 registers per thread:

```
256 * 96 = 24,576
65,536 / 24,576 = 2 blocks per SM (truncated)
2 * 8 warps = 16 warps = 25% occupancy
```

This is the classic "register pressure" tradeoff. More registers
let each thread do more work without re-fetching from memory, but
they reduce the number of concurrent warps and hurt latency
hiding. Most production kernels target 32–64 registers per thread.

You can see register usage with `--ptxas-options=-v` at compile
time:

```
ptxas info    : Used 32 registers, 8192 bytes smem, 388 bytes cmem[0]
```

Lecture 4 covers occupancy in detail.

### Register spilling

If a kernel needs more registers than the per-thread limit (255 on
modern arches), the compiler *spills* the overflow into "local
memory." Despite the name, local memory is **physically in HBM** —
just per-thread instead of per-block. A spill is a 500-cycle round
trip to HBM. One spilled variable in a hot loop can cut performance
by 10×.

The `--ptxas-options=-v` flag will print `X bytes stack frame` if
there are spills. Zero is the goal. If you see spills, refactor —
break the kernel into smaller pieces, reduce per-thread state, or
recompute instead of caching in registers.

---

## Shared memory — software-managed cache

Shared memory is a per-block scratchpad that lives on the SM,
backed by SRAM. It's an order of magnitude faster than global
memory and is the single most powerful optimization tool you have.

```cpp
__global__ void reverse_kernel(const float* in, float* out, int n) {
    __shared__ float buf[256];           // per-block scratchpad
    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tx;

    if (idx < n) {
        buf[tx] = in[idx];   // load from global to shared
    }
    __syncthreads();         // wait for the whole block

    int reversed = blockDim.x - 1 - tx;
    if (idx < n) {
        out[idx] = buf[reversed];  // read from shared, in reverse order
    }
}
```

Key properties:

- **Declared with `__shared__`**, either statically sized (as above)
  or dynamically sized (passed as a third launch parameter:
  `kernel<<<blocks, threads, sm_bytes>>>(...)`).
- **Visible to all threads in the block.** Not across blocks.
- **Lives in the SM's L1/shared SRAM.** Capacity depends on
  architecture and the configured split (see next section). Ampere
  A100: up to 164 KB per SM, configurable.
- **Latency ~20–30 cycles** vs ~400–800 for global memory. 20× faster
  on a good day.

### When to use shared memory

The litmus test: *is data read more than once by more than one thread
in the same block?* If yes, stage it in shared. If no, you're better
off going directly to global (which goes through L1/L2 anyway, so
you get caching for free).

Canonical use cases:

- **Tiled matrix multiply.** Each tile of A is reused by `tile_size`
  output threads; each tile of B is reused by `tile_size` output
  threads. Load once into shared, reuse `tile_size` times.
- **Stencil computations.** Each output element reads a neighborhood
  of the input; neighborhoods overlap. Load a tile-with-halo into
  shared, then everyone reads from there.
- **Histograms with bin reuse.** Per-block partial histograms in
  shared, merged via atomics to a global histogram.

### How much shared memory can a block use?

The hardware limits per architecture (assuming the right
`cudaFuncSetAttribute` to opt into "carveout"):

| Architecture | Max shared per block |
|---|---|
| Pascal (CC 6.0) | 48 KB |
| Volta (CC 7.0) | 96 KB |
| Turing (CC 7.5) | 64 KB |
| Ampere A100 (CC 8.0) | 163 KB |
| Ampere consumer (CC 8.6) | 99 KB |
| Ada (CC 8.9) | 99 KB |
| Hopper (CC 9.0) | 228 KB |

But: using a lot of shared memory per block limits how many blocks
fit on an SM. A block using 48 KB on Ampere (164 KB per SM) gives
you `164 / 48 = 3` blocks per SM (truncated), which is often the
binding constraint. The right amount of shared memory depends on
the algorithm — Lecture 4 walks through the tradeoff.

---

## L1 vs shared: same SRAM, two modes

Since Volta, the L1 cache and shared memory on each SM share the
same physical SRAM bank. You configure the split:

```cpp
// Default: 50/50. Or:
cudaFuncSetAttribute(
    my_kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared  // give more to shared
);
```

The carveout options are roughly:
- `cudaSharedmemCarveoutMaxL1` — favor L1 (more hardware caching).
- `cudaSharedmemCarveoutDefault` — balanced (the default).
- `cudaSharedmemCarveoutMaxShared` — favor shared (more
  software-managed scratchpad).

When to favor each:

- **L1**: kernels with irregular access patterns where you can't
  pre-stage data. Random graph traversal. Sparse linear algebra
  with high reuse but unpredictable patterns. Let the hardware
  cache for you.
- **Shared**: kernels with predictable access patterns where you
  know what to pre-stage. Dense linear algebra. Convolutions.
  Tile in shared, run inner loop on shared.

For most workloads the default split is fine. Tune the carveout
only when profiling tells you the cache is undersized for the
working set.

---

## L2 — the GPU-wide cache

The L2 sits between L1 and HBM. Capacity (Ampere/Hopper-era):

| GPU | L2 size |
|---|---|
| RTX 4090 (Ada AD102) | 72 MB |
| A100 SXM4 80GB (GA100) | 40 MB |
| H100 SXM5 80GB (GH100) | 50 MB |
| L40 (Ada AD102) | 96 MB |

L2 is shared across all SMs, so:

- It's where **inter-block reuse** lives. Block 0 reads location X;
  Block 1 reads location X later; Block 1's read hits L2.
- A "memory-bound" kernel with an L2 cache hit rate of ~80% can
  effectively run at L2 bandwidth (hundreds of GB/s) rather than HBM
  bandwidth (TB/s for the chip, but bottlenecked by HBM at low hit
  rates).
- Hopper introduced **L2 cache residency** controls: you can pin
  certain memory regions (frequently re-read tensors, e.g. attention
  KV cache) into L2 so they don't get evicted. This is one of the
  reasons H100 sometimes blows past expectations on transformer
  workloads — clever L2 use.

When you read "this kernel is bandwidth-bound but I'm hitting 4 TB/s,"
that's L2 doing the work — the chip-level HBM bandwidth is only 3.35
TB/s, but L2 hits don't pay the HBM cost.

---

## Global memory (HBM/GDDR) and coalescing

**Global memory** is the bulk storage on the GPU board. Two physical
implementations dominate:

- **HBM (HBM2e, HBM3, HBM3e)** — stacked DRAM connected via a wide
  (1024–5120-bit) bus. Used on data-center GPUs (A100, H100).
  Per-pin Gbps is modest (3–10 Gbps); total bandwidth comes from
  the wide bus.
- **GDDR (GDDR6, GDDR6X)** — high-clock-rate DRAM on a narrower
  (192–384-bit) bus. Used on consumer GPUs (RTX 4090). Per-pin
  Gbps is high (16–24 Gbps); total bandwidth from the high clock.

The bandwidth formula is the same regardless:

```
peak_bandwidth_GB_per_s = (per_pin_Gbps * bus_width_bits) / 8
```

You'll do this for A100/H100/RTX 4090 in Exercise 2.

### Coalescing — the single most important pattern rule

A warp of 32 threads issues a memory instruction together. If those
threads access **contiguous, aligned** 4-byte words, the hardware
collapses 32 separate accesses into one or two 128-byte memory
transactions. That's *coalesced* — the ideal.

```cpp
// COALESCED: thread i reads word i.
float v = x[idx];   // idx = blockIdx.x * blockDim.x + threadIdx.x

// UNCOALESCED: thread i reads word (i * stride).
float v = x[idx * 8];   // each thread strides 8 floats
```

The first pattern: a 32-thread warp touches 32 consecutive 4-byte
words = 128 bytes, served in a single transaction. The second:
each thread is 32 bytes away from the next; the hardware issues
up to 32 separate transactions, an 8× slowdown (or worse).

```
Coalesced (1 transaction, 128 bytes total):
  thread 0 1 2 3 ... 31
  byte    0 4 8 12 ... 124
  +-----+-----+-----+ ... +-----+
  |  4B |  4B |  4B |     |  4B |
  +-----+-----+-----+ ... +-----+
  <----- one 128-byte transaction ----->

Uncoalesced (32 transactions, sectors discarded):
  thread 0      1      2      ...
  byte   0      32     64     ...
  +-----+ +-----+ +-----+ ...
  | sector | sector | sector | (each one is a 32-byte sector;
  +-----+ +-----+ +-----+      28 bytes per sector wasted)
```

### The 8–32× performance gap

The cost of uncoalesced access on the same algorithm is typically
8–32×. This makes "is the access coalesced?" the **first question**
to ask of any memory-bound kernel.

Common patterns and their coalescing properties:

| Pattern | Coalesced? |
|---|---|
| `x[idx]` (thread `idx`) | Yes |
| `x[idx + offset]` (constant offset) | Yes |
| `x[idx * stride]` (stride > 1) | No (the famous one) |
| `x[perm[idx]]` (permuted index) | No |
| 2D row-major `m[row * N + col]` with `col = threadIdx.x` | Yes if row constant in warp |
| 2D column-major `m[col * M + row]` with `col = threadIdx.x` | NO — strided across rows |

The 2D case is the most common bug. Always think about which
dimension `threadIdx.x` varies along, and make sure that's the
*contiguous* dimension in memory.

### A pathological example

```cpp
// Suppose x is shape [N, 64], row-major.
__global__ void bad_access(const float* x, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // BAD: thread idx reads x[idx, 0]. Strided by 64 between threads.
        out[idx] = x[idx * 64];
    }
}
```

Thread 0 reads byte 0. Thread 1 reads byte 256. Thread 2 reads byte
512. Each thread fetches a *different* 32-byte sector. The warp
spends 32 transactions to read 128 bytes of useful data — a 32× tax
relative to coalesced access.

Fix: rearrange the data layout (transpose, or store as `[64, N]`
instead) or rewrite the kernel to assign threads along the
contiguous dimension.

Exercise 4 has you classify six real kernels by their access
patterns; one of them fails coalescing on purpose.

---

## Bank conflicts

Shared memory is divided into **banks** — 32 on every modern NVIDIA
architecture, matching the warp size. Each 4-byte word maps to one
bank by `(address / 4) % 32`. The 32 lanes of a warp can each access
a *different* bank in parallel — one cycle for the whole warp.

If multiple lanes hit the *same* bank, the accesses serialize: a
2-way conflict takes 2 cycles, a 32-way conflict takes 32 cycles.

The classic conflict: a 2D array `tile[32][32]`. Reading `tile[tx][0]`
across the 32 threads of a warp:

```cpp
__shared__ float tile[32][32];   // row-major; row stride = 32

// Reading down a column:
float v = tile[tx][0];  // thread tx reads tile[tx*32 + 0]
//             addresses: 0, 128, 256, ..., 3968 (in floats)
//             bank = (addr / 4) % 32 = 0 for all 32 lanes -> 32-way conflict
```

All 32 threads hit bank 0. Cost: 32 cycles to access shared memory
that *should* take 1 cycle.

Fix: pad the array by one column:

```cpp
__shared__ float tile[32][33];   // note the +1
//                ^^

// Now reading tile[tx][0]:
//   addresses: 0, 132, 264, ...
//   banks:     0, 1,   2,   ... -> no conflict
```

That `+1` padding is one of the most well-known idioms in CUDA. It
costs 4 extra bytes per row (128 bytes for a 32×32 tile of FP32) to
buy a 32× speedup on the column-read.

Bank conflicts only matter when you're already in shared memory and
they're a big enough fraction of the kernel's runtime to measure.
Nsight Compute reports them as `l1tex__data_bank_conflicts_*`.

---

## Constant and texture memory

Two specialized paths into global memory you should know exist:

### Constant memory

Read-only memory, declared `__constant__`, intended for small
broadcast data (configuration values, lookup tables) that every
thread reads.

```cpp
__constant__ float coeffs[8];

__global__ void k(float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    y[idx] *= coeffs[idx % 8];   // every thread accesses small static data
}
```

There's a tiny per-SM constant cache (8 KB). When all threads in a
warp read the same address, it's a single broadcast — as fast as
a register read. When they read different addresses it serializes.
Use constant memory for "every thread reads the same coefficient"
patterns.

### Texture / surface memory

Read-only memory routed through the texture cache, with hardware
support for bilinear / trilinear interpolation, normalized
addressing, and cache locality on 2D access patterns.

In ML/HPC code you almost never see texture memory anymore (PyTorch,
cuBLAS, cuDNN all use regular global memory + tiling). It exists
because of graphics ancestry. Mention it because you'll see
`__ldg` and "texture cache" in CUDA Programming Guide.

`__ldg(ptr)` is a read-through-the-texture-cache load — useful when
you have read-only data with irregular access and you want the
texture cache's locality benefits without the rest of the
texture-memory machinery.

---

## Worked example: tiled matmul, byte by byte

You're computing `C = A @ B` where all three are `M x N` FP32
matrices with `M = N = 4096`.

**Naive approach:** one output element per thread. Each output
element reads one row of A (M floats) and one column of B (N
floats) — `2*4096 = 8,192` global reads per output, and there are
`M*N = 16,777,216` outputs. Total global reads: ~1.4e11 floats =
~5.4e11 bytes = 540 GB. At A100 peak bandwidth of 2 TB/s, that's
~270 ms just for the reads, before any arithmetic.

But: each row of A is read by 4096 different output threads (all of
output column j). Each column of B is read by 4096 different output
threads (all of output row i). The *real* data is just A + B =
2 * 4096² * 4 bytes = 134 MB — we're re-reading it 4096× from HBM.

**Tiled approach:** load a `TILE x TILE` block of A and a `TILE x TILE`
block of B into shared memory; compute a `TILE x TILE` block of C
from them; advance the tile.

```
For TILE = 32:
- Each output element does TILE multiply-adds per tile step,
  for K/TILE tile steps.
- Each tile of A is loaded into shared once, reused TILE^2 / TILE
  = TILE times by the output threads in that block.
- Same for B.

Global bytes read per block:
  (TILE x TILE per A tile + TILE x TILE per B tile) * (K / TILE)
  = 2 * TILE * K * 4 bytes
  = 2 * 32 * 4096 * 4 = 1,048,576 bytes per block

Number of blocks:
  (M / TILE) * (N / TILE) = 128 * 128 = 16,384 blocks

Total global bytes read:
  16,384 * 1,048,576 = 17.2 GB
```

Compared to the naive 540 GB — a 31× reduction in HBM traffic. At
peak bandwidth that's ~9 ms instead of 270 ms.

The output is a kernel that's now *compute-bound* (arithmetic
intensity went from ~0.06 FLOP/byte to ~85 FLOP/byte for TILE=32)
and runs at >50% of peak FP32 on an A100.

You'll write this kernel in Module 2. The point now is:
**tiling = trading shared memory + code complexity for HBM
bandwidth.** The trade is essentially always worth it for dense
kernels.

---

## Practice problems

**P1.** A kernel reports "8 bytes stack frame, 56 registers" at
compile time. What does the 8 bytes stack frame tell you, and
should you care?

**P2.** Your block uses 12 KB of shared memory. You're on an A100
with 164 KB of shared per SM. What is the shared-memory cap on
blocks per SM? Could a different resource cap kick in first?

**P3.** A warp's 32 threads each read `x[threadIdx.x * 4]`. How
many memory transactions does this generate (assume FP32 elements)?

**P4.** You added `+1` padding to a `__shared__ float tile[32][32]`
to make it `tile[32][33]`. Estimate the kernel speedup. Now you
remove the padding. Estimate the slowdown.

**P5.** You profile a kernel and see 60% L2 hit rate. You change
the input layout and L2 hit rate goes to 95%. Why might wall-clock
time only improve by 1.5×, not 5×?

---

### Answers

**A1.** The 8 bytes is one 8-byte stack frame — that's a *spill*.
Yes, you should care: every spill is a 500-cycle HBM round trip.
Try to refactor to fit in registers (you have 56, so there's
margin) and re-check.

**A2.** Shared-memory cap = `164 / 12 = 13` blocks (truncated).
But `max_blocks_per_sm = 32` is below 13, so blocks isn't binding
either. Warps could be: at 256 threads/block (8 warps), the
warp cap is `64 / 8 = 8` blocks. Warps is the binding resource.

**A3.** Each thread reads a 4-byte float at a stride of 16 bytes.
Thread 0 byte 0; thread 1 byte 16; ... thread 31 byte 496. That
covers 16 different 32-byte sectors (one per 2 threads). So
~16 sector transactions versus the ideal 1. Memory throughput is
about 1/16 of peak.

**A4.** The `+1` padding gives roughly a 32× speedup on the affected
shared-memory accesses (eliminating a 32-way bank conflict).
Removing it returns to the 32× slow path. If those accesses are
50% of the kernel's runtime, the overall kernel slows down by
~2× (half the kernel went 32× slower).

**A5.** L2 hit rate change only helps the *misses* that previously
went to HBM. If the kernel is also compute-bound or has high
arithmetic intensity, you've already been amortizing some HBM
cost; reducing it further has diminishing returns. The kernel is
likely now bottlenecked on something else (compute units, atomic
serialization, occupancy).

---

*End of Lecture 1.3. Next: Lecture 1.4 — Warps and Occupancy.*
