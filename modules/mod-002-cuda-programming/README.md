# Module 2 — CUDA Programming

> **Track:** AI/ML Performance Engineer
> **Time budget:** ~30 hours (3 weeks part-time)
> **Module status:** first build, 2026-05-23

Module 01 built the mental model: SIMT, warps, memory hierarchy, roofline,
occupancy. You can now reason about *whether* a workload belongs on a GPU
and *what* would make a given kernel fast. Module 02 takes you from that
analytical posture to actually writing CUDA: a `__global__` function, a
`<<<grid, block>>>` launch, the CUDA runtime and build toolchain, and the
PyTorch C++ extension path that lets you call your kernel from Python.

The goal is not to turn you into a CUDA library author. The goal is for
you to be able to drop down to a custom kernel when your workload does not
map cleanly onto `torch.*` ops, and to know when that is the right call
versus when it is premature. By the end of the module you will have
written an element-wise kernel that matches `torch.add`, a tiled matmul
that reaches a meaningful fraction of cuBLAS, and a PyTorch extension
that packages a kernel with autograd support — the three building blocks
behind nearly every custom op you will see in production inference code.

Module 03 picks up the profiling side: given the kernels you have now
written, how do you measure them with Nsight Systems / Nsight Compute and
reason from traces rather than guesses. CUDA authoring and CUDA profiling
are deliberately split across the two modules so each can be done
carefully.

## Learning objectives

By the end of this module, you will be able to:

1. **Write** a correct CUDA kernel from scratch — `__global__` signature,
   thread-index math, bounds check, host-side launch — for an
   element-wise op, and get it to within 10% of the equivalent
   `torch.*` call on a ~1M-element input.
2. **Choose** a launch configuration (block size, grid size) for a given
   kernel class (memory-bound, compute-bound, reduction), justify the
   choice from warp size and occupancy, and verify it empirically with
   `cudaOccupancyMaxPotentialBlockSize` plus a sweep.
3. **Apply** shared memory tiling to a matrix multiply, pad to avoid
   bank conflicts, and achieve ≥ 50% of cuBLAS throughput at `N = 4096`
   on a modern NVIDIA GPU.
4. **Implement** three variants of a parallel reduction (naive, shared
   memory tree, warp-shuffle) and explain the performance gap between
   them in terms of global-memory traffic and intra-warp communication
   cost.
5. **Build** a PyTorch C++ extension with `torch.utils.cpp_extension`,
   wire forward + backward so autograd works, and benchmark it against
   the corresponding `torch.*` op with proper synchronization and warmup.
6. **Diagnose** the common correctness and performance pitfalls — block
   size not a multiple of warp size, uncoalesced global accesses,
   shared-memory bank conflicts, warp divergence inside hot loops — from
   source and from profiler output.

Every exercise targets one or more of these; the quiz checks the
underlying concepts.

## Prerequisites

You should be comfortable with:

- **Module 01** — SIMT, warps, occupancy, roofline, the memory hierarchy.
  If you cannot sketch a roofline and place a kernel on it, do Module 01
  first; nothing here will land otherwise.
- **C++ basics** — pointers, structs, `const`-correctness, separate
  compilation. You do not need templates or modern C++; the CUDA C++
  surface we use is small.
- **A working CUDA toolchain** — `nvcc` on `$PATH`, a matched NVIDIA
  driver, and a GPU you can run on. The exercises assume you can run
  `nvcc --version` and `nvidia-smi` and see sensible output. Compute
  capability 7.0 (Volta) or newer is expected; H100 / A100 / L40S /
  RTX 4090 are all fine. CPU-only is **not** an option in this module.

If you do not have GPU access, run on a short-lived cloud instance — the
exercises are measured in minutes of GPU time, not hours.

## Module structure

The lecture is split across three chapter files in the
[`lecture-notes/`](lecture-notes/) directory. Read in order:

| Chapter | What's in it |
|---|---|
| [`lecture-notes/01-kernels-and-launch-config.md`](lecture-notes/01-kernels-and-launch-config.md) | The `__global__` kernel shape, thread/block/grid indexing math, host-side `<<<grid, block>>>` launch, picking a block size for different workload classes, occupancy caps, and the common launch-configuration mistakes. |
| [`lecture-notes/02-memory-coalescing-shared-memory.md`](lecture-notes/02-memory-coalescing-shared-memory.md) | Global-memory coalescing rules, shared memory as scratchpad, the tiled matmul walkthrough, bank conflicts and the `[TS][TS+1]` padding trick, and where a hand-written kernel plateaus versus cuBLAS. |
| [`lecture-notes/03-pytorch-c-extension.md`](lecture-notes/03-pytorch-c-extension.md) | The `torch::Tensor` wrapper surface, `TORCH_CHECK` validation, `setup.py` with `CUDAExtension`, how the build pipeline invokes `nvcc` + `gcc` + setuptools, and how to benchmark against the `torch.*` baseline without fooling yourself. |

| Exercise | What it covers |
|---|---|
| [`exercises/exercise-01-vector-add.md`](exercises/exercise-01-vector-add.md) | Write `add_kernel.cu` plus a minimal PyTorch extension; benchmark against `torch.add`. Target: within 10% at `n = 1M`, within 2× at `n = 10K`. |
| [`exercises/exercise-02-tiled-matmul.md`](exercises/exercise-02-tiled-matmul.md) | Shared-memory tiled matmul. Sweep `TS ∈ {16, 32, 64}`, report TFLOPS, achieved occupancy (from `ncu`), bank-conflict count, and % of cuBLAS at `N = 4096`. |
| [`exercises/exercise-03-reduction.md`](exercises/exercise-03-reduction.md) | Three reduction variants — naive, block-level shared-memory tree, warp-shuffle. Benchmark at `N = 1M`; variant 3 should reach ≥ 80% of `torch.sum`. |
| [`exercises/exercise-04-occupancy-tuning.md`](exercises/exercise-04-occupancy-tuning.md) | Take the tiled matmul and sweep block + tile size. Report achieved occupancy and TFLOPS in an `OCCUPANCY_REPORT.md`. |
| [`exercises/exercise-05-pytorch-extension.md`](exercises/exercise-05-pytorch-extension.md) | Package the matmul as a `torch.utils.cpp_extension` with autograd (forward + backward); confirm gradients match `torch.matmul` in a small linear layer. |

Plus:

| File | What's in it |
|---|---|
| [`quiz.md`](quiz.md) / [`quiz-answers.md`](quiz-answers.md) | 15-question multiple-choice check on the learning objectives. 75% to pass. |
| [`resources.md`](resources.md) | Primary sources — the CUDA C++ Programming Guide, the CUDA Best Practices Guide, the PTX ISA reference, the Volta whitepaper, the PyTorch C++ extension tutorial. |

Companion repo:
[engineer-solutions/mod-107 ex-02 (cuda-kernel)](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-107-gpu-computing/exercise-02-cuda-kernel)
has a working `setup.py`, `add_kernel.cu`, and `bench.py` you can study
after you have taken a first pass yourself. Do not open it before
Exercise 01.

## How to work through this module

1. Read the three chapter files in `lecture-notes/` in order. Plan for
   roughly 4–6 hours of reading plus hands-on code-along; every chapter
   includes a kernel you can type out and run.
2. Do the five exercises in order. Each one lists concrete performance
   targets — hit them before moving on. Exercises 01 and 05 share a
   PyTorch-extension scaffold; reuse it.
3. Expect to spend time in `nvcc` error messages and `cuda-memcheck` /
   `compute-sanitizer` output. CUDA is unforgiving about illegal
   accesses; a silent wrong answer is normal before you learn to trust
   the tooling.
4. Take the quiz **without** scrolling to `quiz-answers.md`. If you miss
   more than four questions, reread the chapter that covers them before
   moving on to Module 03.

## Assessment criteria for this module

You are ready to move on when you can, on a blank page:

- Write a correct `__global__` kernel signature with the thread-index
  math and bounds check, and the matching host-side launch, for an
  arbitrary element-wise op — no reference material.
- Justify a block-size choice for a given kernel in one or two
  sentences, grounded in warp size, register pressure, and the
  occupancy caps from Module 01.
- Explain, from a source snippet, whether a global-memory access is
  coalesced and whether a shared-memory access pattern will cause bank
  conflicts; propose the fix for each.
- Describe the three pieces of a PyTorch C++ extension (the `.cu`
  kernel, the `torch::Tensor` wrapper, `setup.py`) and what each is
  responsible for.

Module 03 picks up here and teaches you to measure what you just wrote
with Nsight Systems and Nsight Compute, so the kernels you build in this
module become the test subjects for the profiler there.
