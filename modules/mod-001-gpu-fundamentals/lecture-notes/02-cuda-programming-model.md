# Lecture 1.2 — CUDA Programming Model

> Lesson 2 of 6. Plan for ~75 minutes of reading. The code in this
> lecture is meant to be *read*, not *run* — this module is
> CPU-only, and the CUDA syntax is the focus of Module 2. Your goal
> here is to be able to look at a CUDA launch and predict what it
> does on the hardware.

## Table of Contents

1. [The kernel + launch configuration mental model](#the-kernel--launch-configuration-mental-model)
2. [Threads, blocks, grids, warps](#threads-blocks-grids-warps)
3. [Indexing math (the part everyone gets wrong first)](#indexing-math-the-part-everyone-gets-wrong-first)
4. [The host/device boundary](#the-hostdevice-boundary)
5. [Synchronization primitives](#synchronization-primitives)
6. [Streams and concurrency](#streams-and-concurrency)
7. [Worked example: planning a launch for vector addition](#worked-example-planning-a-launch-for-vector-addition)
8. [Common pitfalls](#common-pitfalls)
9. [Practice problems](#practice-problems)

---

## The kernel + launch configuration mental model

A CUDA program has two parts: code that runs on the **host** (the CPU,
in regular C++ or Python) and code that runs on the **device** (the
GPU, in a function called a *kernel*). The host launches the kernel;
the device executes it.

A kernel launch in CUDA C++ looks like this:

```cpp
vector_add<<<num_blocks, threads_per_block>>>(a_d, b_d, c_d, n);
```

The `<<< ... >>>` syntax — three angle brackets on each side — is
not legal C++. It's CUDA's way of saying "this is a kernel launch
with these execution parameters." It compiles into a runtime call
that schedules the kernel on the GPU.

The two parameters are the **launch configuration**:

- **`num_blocks`** — how many thread blocks to launch. Can be 1D, 2D,
  or 3D. Hardware caps the X dimension at 2^31 − 1 and Y/Z at 65,535.
- **`threads_per_block`** — how many threads in each block. Also up to
  3D. Total threads per block is capped at 1024.

The total number of threads is `num_blocks * threads_per_block`. For
an 8M-element problem with 256 threads per block, you'd launch
`num_blocks = 32,768` and have 8,388,608 threads in flight.

```
Launch a kernel:
    vector_add<<<32768, 256>>>(...);
                |       |
                |       +-- 256 threads per block (= 8 warps)
                +-- 32,768 blocks in the grid

Hardware sees:
    Total threads          : 32768 * 256 = 8,388,608
    Total warps in grid    : 8,388,608 / 32 = 262,144
    Warps the chip will run at any one moment:
        depends on occupancy + SM count.
        On an A100 with 50% occupancy: 108 * 32 = 3,456 warps.
        So the chip cycles through 262,144 / 3,456 = 76 "waves".
```

That last line — *waves* — is a key concept. A wave is one batch of
blocks the chip can hold at once. The whole grid runs as a series of
waves; the GPU keeps issuing blocks onto SMs as soon as old blocks
finish. You'll do this calculation in Exercise 3.

---

## Threads, blocks, grids, warps

Four nested concepts, but only three of them are exposed to the
programmer:

```
+--------------------------------------------+
| GRID            (you choose)               |
|                                            |
|   +------------------+ +------------------+|
|   | BLOCK            | | BLOCK            ||
|   |   (you choose)   | |                  ||
|   |                  | |                  ||
|   |  +----+ +----+   | |  +----+ +----+   ||
|   |  |WARP| |WARP|   | |  |WARP| |WARP|   ||
|   |  |(hw)| |(hw)|   | |  |    | |    |   ||
|   |  +----+ +----+   | |  +----+ +----+   ||
|   |  +----+ +----+   | |  +----+ +----+   ||
|   |  |WARP| |WARP|   | |  |WARP| |WARP|   ||
|   |  +----+ +----+   | |  +----+ +----+   ||
|   |                  | |                  ||
|   |  (each warp =    | |                  ||
|   |   32 threads,    | |                  ||
|   |   hw-grouped)    | |                  ||
|   +------------------+ +------------------+|
+--------------------------------------------+
```

- **Thread** — the smallest unit of work, what you write code for in
  the kernel.
- **Warp** — 32 threads, grouped *by the hardware*. You don't
  configure this; the hardware groups consecutive thread IDs into
  warps. Always 32. You can't make it 16 or 64.
- **Block** — a group of threads (max 1024) that lives entirely on
  one SM, sharing its shared memory and synchronizing via
  `__syncthreads()`.
- **Grid** — the collection of all blocks for one kernel launch.

Threads in different blocks **cannot synchronize within a kernel**
(except via global atomic operations or kernel-launch boundaries).
This is a fundamental constraint of the model. If two blocks need
to exchange data, you split the kernel.

(Hopper introduced "thread block clusters" which relax this for
neighboring blocks, but that's a Module 2+ topic.)

### Why this hierarchy?

The hierarchy maps to hardware:

- **Threads** → individual lanes in an SM's arithmetic units.
- **Warps** → the SIMD execution unit; one instruction issued across
  32 lanes per cycle (per sub-partition).
- **Blocks** → resident on one SM. Share shared memory and L1.
  Cooperate via `__syncthreads()`.
- **Grid** → distributed across SMs. SMs are independent.

When you choose a block size, you're choosing how much state
(registers, shared memory) lives on one SM and how granularly the
hardware can move work around. Big blocks = less scheduling
flexibility, more shared resources within a block. Small blocks =
more flexibility, less per-block sharing.

---

## Indexing math (the part everyone gets wrong first)

Inside a kernel, a thread finds itself in the grid using built-in
variables:

```cpp
__global__ void k(...) {
    int tx = threadIdx.x;   // 0 .. blockDim.x - 1
    int bx = blockIdx.x;    // 0 .. gridDim.x - 1
    int bw = blockDim.x;    // threads per block (x dim)
    int gw = gridDim.x;     // blocks in grid     (x dim)

    int linear_id = bx * bw + tx;   // 0 .. (gw*bw - 1)
    ...
}
```

The Python equivalent (just to make the math clear) is:

```python
def linear_thread_id_1d(block_idx: int, block_dim: int, thread_idx: int) -> int:
    return block_idx * block_dim + thread_idx
```

For 2D (the natural fit for image / matrix problems):

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

// For an MxN matrix, the linear element index is:
int idx = row * N + col;   // row-major order
```

For 3D the same pattern repeats with `z`.

### The boundary check

When the array length isn't a clean multiple of the block size, the
last block has threads that don't have work to do. You handle this
with a guard:

```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

That `if (idx < n)` is *not optional*. Without it, the threads with
`idx >= n` index past the end of the arrays and corrupt memory or
crash. Forgetting this guard is one of the most common bugs in
first-week CUDA code.

### Why the guard is cheap

You might worry: doesn't the `if` cause divergence? In the *last*
block, yes — some threads take it, some don't. But this happens at
most once per launch (only the last block), so the cost is
negligible. Don't try to "optimize" by removing the guard.

### Choosing grid dimensions for a given N

The canonical formula:

```cpp
int threads_per_block = 256;
int num_blocks = (n + threads_per_block - 1) / threads_per_block;
```

That ceiling-divide pattern launches *just enough* blocks to cover
N elements with one thread each. The last block may have some idle
threads (handled by the guard).

For 2D over an MxN matrix:

```cpp
dim3 threads(16, 16);             // 256 threads = 8 warps
dim3 blocks(
    (N + threads.x - 1) / threads.x,
    (M + threads.y - 1) / threads.y
);
kernel<<<blocks, threads>>>(...);
```

The `16x16` block has two purposes: (a) it's a multiple of 32
(8 warps), and (b) it's a natural shape for cache locality in 2D
problems.

---

## The host/device boundary

CPU memory and GPU memory are separate physical pools. Until very
recent unified-memory features (and even then, with caveats), data
moves between them explicitly:

```cpp
// Host (CPU) memory.
float* a_h = (float*) malloc(N * sizeof(float));

// Device (GPU) memory.
float* a_d;
cudaMalloc(&a_d, N * sizeof(float));

// Copy host -> device.
cudaMemcpy(a_d, a_h, N * sizeof(float), cudaMemcpyHostToDevice);

// Launch a kernel that operates on device data.
my_kernel<<<blocks, threads>>>(a_d, ...);

// Copy device -> host to see the result.
cudaMemcpy(a_h, a_d, N * sizeof(float), cudaMemcpyDeviceToHost);

// Always free.
cudaFree(a_d);
free(a_h);
```

The Python equivalent in PyTorch (which hides cudaMalloc/cudaMemcpy):

```python
import torch

a_h = torch.zeros(N)             # CPU tensor
a_d = a_h.cuda()                 # implicit cudaMemcpy host -> device
result_d = my_function(a_d)      # runs on GPU
result_h = result_d.cpu()        # implicit cudaMemcpy device -> host
```

The transfer cost is real. PCIe Gen4 x16 caps at about 32 GB/s.
A 4 GB tensor crossing PCIe is ~125 ms — comparable to or longer
than the actual compute time on many kernels. **Plan kernel
sequences to keep data on the device** between operations; this is
the single most important optimization for end-to-end training
loops.

### Pinned (page-locked) host memory

Standard `malloc`'d host memory can be paged out by the OS, which
forces the GPU to bounce-buffer through pinned memory anyway. You
can pre-pin host memory yourself with `cudaMallocHost` (or
`pin_memory=True` in PyTorch's DataLoader); this is a 2× transfer
speedup for free, at the cost of using non-pageable RAM.

You'll see this in Module 6 (data loading pipelines). For now, just
know that "pinned memory" exists and is roughly 2× faster than
unpinned for host↔device transfer.

---

## Synchronization primitives

The CUDA programming model gives you three levels of synchronization:

### 1. Within a warp: implicit

Threads in a warp always execute the same instruction at the same
time (modulo divergence — covered in Lecture 4). You don't need to
synchronize within a warp.

For *explicit* communication within a warp you use **shuffles**:

```cpp
// Each thread in the warp reads `value` from lane (laneId + offset).
float partner_value = __shfl_down_sync(0xFFFFFFFF, value, offset);
```

Shuffles are essentially free (a few cycles) and the right way to do
warp-level reductions, prefix sums, etc. (Module 3.)

### 2. Within a block: __syncthreads()

```cpp
__shared__ float tile[256];

__global__ void k(...) {
    int tx = threadIdx.x;
    tile[tx] = ...;        // each thread writes one slot
    __syncthreads();        // wait for ALL threads in the block
    // now any thread can safely read any tile[i]
    ...
}
```

`__syncthreads()` is a hardware barrier: every thread in the block
must reach it before any thread can pass it. This is how you
coordinate the shared-memory writes/reads in tiled kernels.

Two failure modes to remember:

- **Calling `__syncthreads()` inside a divergent branch.** If some
  threads take the `if` and some don't, the ones in the `if`
  block hit `__syncthreads()` and wait — for threads that will
  never arrive. The kernel hangs. (On modern GPUs with independent
  thread scheduling this is *technically* allowed in narrow
  cases, but treat it as undefined behavior.)
- **Forgetting `__syncthreads()` between phases.** "Why is my
  output wrong?" If two threads write to shared memory and a
  third reads, you need a sync in between. Race conditions in
  GPU code are silent and reproducible.

### 3. Across blocks: kernel boundary

There is no `__syncblocks()`. To synchronize all blocks in a grid
you end the kernel — the kernel boundary is itself a global sync
point — and launch another one. This is why algorithms like prefix
sum on large arrays are typically *multiple* kernel launches:
one per phase.

(Cooperative groups and Hopper's thread block clusters provide
limited cross-block sync, but treat them as advanced features
for now.)

### atomicAdd and friends

For irregular cross-block coordination there are atomic operations:

```cpp
__device__ int counter;

__global__ void k() {
    int my_id = atomicAdd(&counter, 1);
    ...
}
```

These work, but they serialize: every concurrent `atomicAdd` to the
same address has to take a turn. Use them for low-frequency
coordination (work queues, histograms with many bins), not in the
inner loop of a high-throughput kernel.

---

## Streams and concurrency

A **stream** is a queue of operations that execute in order on the
device. Operations in different streams *can* overlap. By default
all work goes to the default stream, which forces serialization;
explicitly creating streams lets you overlap kernel execution with
host↔device copies.

```cpp
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

// These two can overlap on the device.
cudaMemcpyAsync(a_d, a_h, sz, cudaMemcpyHostToDevice, s1);
kernel<<<blocks, threads, 0, s2>>>(b_d, ...);

cudaStreamSynchronize(s1);
cudaStreamSynchronize(s2);
```

In PyTorch:

```python
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

with torch.cuda.stream(s1):
    a_d = a_h.cuda(non_blocking=True)

with torch.cuda.stream(s2):
    result_d = my_function(b_d)
```

Streams matter for end-to-end performance: you can prefetch the next
batch's data while the current batch is being processed. This is
Module 6 in depth. For Module 1, just know:

- One stream = sequential execution. Easy to reason about.
- Multiple streams = potential concurrency. Use when you have
  independent work (especially HtoD copies) you want to overlap
  with the main compute.

---

## Worked example: planning a launch for vector addition

You want to compute `c = a + b` on FP32 arrays of length N = 10,000,000.
Walk through every decision:

**1. What's the per-thread work?** One element. Read `a[idx]`, read
`b[idx]`, compute the sum, write `c[idx]`.

**2. How many threads total?** 10,000,000 — one per element.

**3. Threads per block?** Default 256 (= 8 warps). This is a
reasonable starting point. Multiples of 32 only; 256 is a sweet
spot that leaves room for several blocks per SM (occupancy in
Lecture 4).

**4. Number of blocks?**

```
num_blocks = ceil(N / 256) = ceil(10000000 / 256) = 39063
```

**5. Total launched threads?** 39,063 × 256 = 10,000,128. So 128
threads in the last block will hit the boundary guard and exit.

**6. Will this saturate an A100?** A100 has 108 SMs × 64 max
warps/SM = 6,912 max concurrent warps. Our grid has 10,000,000 / 32
= 312,500 warps. We're 45× oversubscribed — plenty of work to keep
all SMs busy (assuming the kernel achieves any reasonable
occupancy).

**7. Is this compute-bound or memory-bound?** Vector add has
AI = 1 / 12 ≈ 0.083 FLOP/byte. A100 ridge point ≈ 9.5 FLOP/byte.
This is deeply memory-bound; the upper bound is HBM bandwidth ÷ 12
bytes/element = 2039e9 / 12 = 170 G-elements/sec ≈ 0.7 ms for
10M elements. Compute is irrelevant; we're moving bytes.

**8. CUDA launch line:**

```cpp
const int N = 10'000'000;
const int THREADS_PER_BLOCK = 256;
const int NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

vector_add<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(a_d, b_d, c_d, N);
```

You'll do this analysis dozens of times in the rest of the course.
The pattern — *how much work, how many threads, how many blocks,
which resource is the bottleneck* — generalizes to every kernel you
will ever write.

---

## Common pitfalls

### 1. Off-by-one in boundary check

```cpp
if (idx <= n) { c[idx] = a[idx] + b[idx]; }   // BUG: writes c[n]
if (idx <  n) { c[idx] = a[idx] + b[idx]; }   // correct
```

### 2. Block dim chosen without a warp multiple

```cpp
kernel<<<blocks, 100>>>(...);   // 100 = 3.125 warps; the 4th warp wastes 28 lanes
kernel<<<blocks, 128>>>(...);   // 4 full warps, no waste
```

### 3. Forgetting `__syncthreads()` between shared-memory phases

```cpp
tile[tx] = a[idx];
// MISSING SYNC HERE
float other = tile[(tx + 1) % BLOCK_SIZE];  // race
```

### 4. Treating thread IDs as if they were CPU thread IDs

GPU threads aren't OS threads. There's no separate stack, no `pthread`
semantics, no preemption. A thread that finishes its work just
returns; it doesn't pick up new work.

### 5. Allocating in the kernel

`malloc`/`new` work inside CUDA kernels but they're slow and use a
small per-device heap. Don't allocate in kernels. Allocate up-front
on the host and pass pointers in.

### 6. Calling cudaMalloc inside a hot loop

Allocation/deallocation is expensive (milliseconds). Reuse buffers.
PyTorch's caching allocator does this automatically; raw CUDA C++
does not — you have to be disciplined.

### 7. Ignoring return codes

```cpp
cudaError_t err = cudaMalloc(&a_d, sz);
if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    abort();
}
```

CUDA errors are *not exceptions*. They have to be checked. The
`#define CUDA_CHECK(x) do { ... } while(0)` macro is in every
production CUDA codebase for a reason.

---

## Practice problems

**P1.** A grid is launched with `<<<dim3(32, 16), dim3(8, 8)>>>`.
How many total threads? How many warps? Is the block size
warp-friendly?

**P2.** You have N = 1,000,003 elements and you choose 256 threads
per block. How many blocks do you launch? How many threads in the
last block do nothing?

**P3.** Why can't two threads in different blocks safely write to
the same shared-memory address?

**P4.** A kernel uses `__syncthreads()` inside an `if (tx < 64)`
branch on a 128-thread block. What happens?

**P5.** You move a kernel from the default stream onto two named
streams (one for HtoD copy, one for compute). The total wall-clock
time goes *up* slightly. Why?

---

### Answers

**A1.** 32 * 16 * 8 * 8 = 32,768 threads. 32,768 / 32 = 1,024 warps.
Block size = 64; 64 / 32 = 2 warps per block, fully warp-aligned.

**A2.** `ceil(1000003 / 256) = 3907` blocks. Last block has
`3907 * 256 - 1000003 = 253` threads that hit the boundary guard.

**A3.** Shared memory is per-block. Block A's shared memory lives on
one SM; Block B's might be on another SM entirely, or in a different
slot on the same SM. "Shared" means shared *within a block*, not
across blocks.

**A4.** Undefined behavior / hang on most architectures. The
`__syncthreads()` is reached only by the 64 threads in the `if`;
the other 64 never arrive. Pre-Volta this hangs the kernel; on
Volta+ with independent thread scheduling it's still technically
illegal.

**A5.** Stream creation, destruction, and synchronization have
their own overhead. For a single kernel with no actual overlap
opportunity, the extra plumbing costs more than the (zero)
benefit. Streams help when you have *parallelizable* work to
overlap.

---

*End of Lecture 1.2. Next: Lecture 1.3 — CUDA Memory Hierarchy.*
