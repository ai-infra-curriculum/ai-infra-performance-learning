# Module 1 — GPU Fundamentals

> **Track:** AI/ML Performance Engineer
> **Time budget:** ~20 hours (2 weeks part-time)
> **Module status:** first build, 2026-05-22

This is the first module in the Performance Engineering track. It sets the
mental model you will reuse for the rest of the curriculum: what a GPU
actually is, why it is fast on some workloads and slow on others, and how to
reason about that quantitatively *before* you write a single line of CUDA.

The lecture is intentionally light on CUDA code. You cannot meaningfully
optimize what you do not understand at the architecture level, and almost
all real-world performance work is decided in the math (arithmetic
intensity, memory bandwidth, occupancy) — not in the kernel syntax. The
CUDA programming syntax is the focus of [Module 2](../mod-002-cuda-programming/README.md).
For this module, your job is to think like a hardware engineer.

## Learning objectives

By the end of this module, you will be able to:

1. **Explain** the SIMT execution model and how a warp differs from a thread
   in CPU-style multi-threading.
2. **Calculate** the theoretical peak FP32, TF32, and FP16 throughput of a
   given NVIDIA GPU from its published spec sheet.
3. **Calculate** theoretical peak memory bandwidth from a memory clock,
   memory bus width, and memory type (GDDR / HBM).
4. **Identify** whether a workload is compute-bound or memory-bound using
   arithmetic intensity and the roofline model.
5. **Estimate** kernel occupancy from a launch configuration (block size,
   registers per thread, shared memory per block) and explain which of those
   three resources is the limiting one.
6. **Distinguish** the levels of the GPU memory hierarchy (registers,
   shared / L1, L2, global / HBM) and predict which level a memory access
   will hit given a kernel's access pattern.

These are measurable; every exercise targets one or more of them.

## Prerequisites

You should be comfortable with:

- Python (numpy in particular)
- Basic linear algebra (matrix multiply, what FLOPS means)
- Reading a hardware spec sheet without panicking

You do **not** need to have written CUDA before. You also do not need to
have an NVIDIA GPU available *to do this module* — all four exercises run
on CPU. (A GPU is required from Module 2 onward.)

## Module structure

The lecture is split across six chapter files in the
[`lecture-notes/`](lecture-notes/) directory, one per lesson. Read in
order:

| Chapter | What's in it |
|---|---|
| [`lecture-notes/01-gpu-architecture.md`](lecture-notes/01-gpu-architecture.md) | SM hierarchy, CUDA cores, tensor cores, compute capability, the CPU/GPU silicon split, common misconceptions. |
| [`lecture-notes/02-cuda-programming-model.md`](lecture-notes/02-cuda-programming-model.md) | Kernels, threads, blocks, grids, warps, indexing math, host/device boundary, synchronization, streams. |
| [`lecture-notes/03-cuda-memory-hierarchy.md`](lecture-notes/03-cuda-memory-hierarchy.md) | Registers, shared memory, L1 vs shared carveout, L2, HBM/GDDR, coalescing, bank conflicts, tiled matmul walkthrough. |
| [`lecture-notes/04-warps-and-occupancy.md`](lecture-notes/04-warps-and-occupancy.md) | Latency hiding, the three occupancy caps, branch divergence, warp-level primitives, when low occupancy is fine. |
| [`lecture-notes/05-roofline-model.md`](lecture-notes/05-roofline-model.md) | Arithmetic intensity, the roofline, ridge point, hierarchical rooflines, six worked examples on A100. |
| [`lecture-notes/06-gpu-generations.md`](lecture-notes/06-gpu-generations.md) | A100 / H100 / L40S / RTX 4090 specs, how to read an NVIDIA datasheet, procurement framework. |

| Exercise | What it covers |
|---|---|
| [`exercises/exercise-01-cpu-gpu-comparison`](exercises/exercise-01-cpu-gpu-comparison/) | Use the SIMT model to predict which of three workloads accelerates on a GPU and by roughly how much. CPU-only. |
| [`exercises/exercise-02-peak-throughput`](exercises/exercise-02-peak-throughput/) | Given spec-sheet values, compute peak FP32 / HBM bandwidth for A100, H100, RTX 4090. CPU-only. |
| [`exercises/exercise-03-occupancy-calculator`](exercises/exercise-03-occupancy-calculator/) | Reimplement NVIDIA's occupancy calculation from scratch. CPU-only. |
| [`exercises/exercise-04-roofline-analysis`](exercises/exercise-04-roofline-analysis/) | Plot a roofline for an A100 and place six kernels on it. CPU-only. |
| [`exercises/exercise-05-warp-divergence-analyzer`](exercises/exercise-05-warp-divergence-analyzer/) | Model branch divergence cost as a function of branch probability and per-branch cost. CPU-only. |
| [`exercises/exercise-06-tensor-core-throughput`](exercises/exercise-06-tensor-core-throughput/) | Predict GEMM TFLOPS on A100 vs H100 with tensor cores, using the roofline. CPU-only. |

Plus:

| File | What's in it |
|---|---|
| [`quiz.md`](quiz.md) / [`quiz-answers.md`](quiz-answers.md) | 12-question check on the learning objectives. |
| [`resources.md`](resources.md) | Primary sources — NVIDIA architecture whitepapers, the CUDA C Programming Guide, Volkov's thesis, the roofline paper. |

## How to work through this module

1. Read the six chapter files in `lecture-notes/` in order. Plan for
   roughly 6–8 hours total. Each chapter ends with practice problems;
   do them.
2. Do the six exercises in order. Each one has a `starter.py` you edit
   and a `check.py` that grades your answer. `python check.py` should
   print `PASS` when you are done.
3. Take the quiz **without** scrolling to the answer key. If you miss
   more than two questions, reread the relevant chapter before moving
   on to Module 2.

## Assessment criteria for this module

You are ready to move on when you can, on a blank page:

- Write the SIMT execution model in three sentences without using the word
  "thread" the way a CPU programmer would use it.
- Compute peak FP32 throughput of any NVIDIA GPU from CUDA-core count and
  boost clock.
- Sketch a roofline and place a kernel on it given its arithmetic
  intensity and achieved GFLOP/s.

Module 2 picks up from these primitives and starts writing kernels.
