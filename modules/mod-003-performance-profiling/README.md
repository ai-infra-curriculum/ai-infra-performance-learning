# Module 3 — Performance Profiling

> **Track:** AI/ML Performance Engineer
> **Time budget:** ~25 hours (2–3 weeks part-time)
> **Module status:** first build, 2026-05-24

Modules 01 and 02 gave you the mental model and the authoring skill: you
can reason about whether a workload belongs on a GPU and you can write a
CUDA kernel from scratch. Module 03 closes the loop on that work — given
a running GPU program, how do you find out where the time actually goes,
decide what to optimize, and prove that your change helped. The answer is
not reading code harder; it is measurement.

The core tools are Nsight Systems (`nsys`) for the system-level timeline,
Nsight Compute (`ncu`) for per-kernel deep dives, and the PyTorch profiler
for workloads where the Python side matters as much as the CUDA side.
Alongside them you will learn to read memory snapshots from
`torch.cuda.memory._record_memory_history`, which is the only honest way
to answer "why did this OOM?" for a non-trivial training loop.

The posture this module trains is deliberate: diagnose first, then edit.
Every optimization chapter in this curriculum — the FlashAttention and
fused-kernel work in Module 04, the quantization work in Module 05,
distributed inference in Module 06 — starts from a profile and ends with
another profile proving the change paid off. If profiling is not a
reflex by the end of this module, those later modules will feel like
guessing. The goal here is to make "reason from traces, not vibes" a
default working mode.

## Learning objectives

By the end of this module, you will be able to:

1. **Choose** between `nsys` and `ncu` for a given profiling question —
   system-level time attribution versus per-kernel tuning — and justify
   the choice in one sentence.
2. **Collect** an `nsys` trace of a PyTorch training step, annotate it
   with NVTX ranges for forward / backward / optimizer, and read the
   resulting timeline to identify the top three kernels by total GPU
   time.
3. **Interpret** the Nsight Compute "GPU Speed of Light" section — the
   compute %, memory %, and achieved occupancy numbers — and place the
   kernel on the roofline as memory-bound, compute-bound, or
   under-utilized.
4. **Collect** and read a `torch.profiler` trace in TensorBoard or
   Perfetto, distinguish CPU-side overhead (DataLoader stalls, Python
   dispatch) from GPU-side time, and propose a targeted fix for each.
5. **Capture** a PyTorch memory snapshot, upload it to
   [`pytorch.org/memory_viz`](https://pytorch.org/memory_viz), and
   identify peak memory, the largest single allocation, and whether the
   loop reaches steady state across iterations.
6. **Run** the measure → hypothesize → change → re-measure cycle end to
   end on a slow training script and produce a before/after profile that
   shows a ≥2× improvement in p95 step time.

Every exercise targets one or more of these; the quiz checks the
underlying concepts.

## Prerequisites

You should be comfortable with:

- **Module 01** — roofline, arithmetic intensity, occupancy, the memory
  hierarchy. Every profiler section in this module reports numbers that
  only make sense against those primitives.
- **Module 02** — writing a kernel and launching it from PyTorch. The
  kernels you wrote in Module 02 are the test subjects for the profilers
  here; several exercises point `ncu` directly at them.
- **A working CUDA toolchain** — `nvcc`, a matched NVIDIA driver, and
  a GPU you can run `nsys`, `ncu`, and PyTorch on. Nsight Systems and
  Nsight Compute ship with the CUDA Toolkit; you do not install them
  separately. Compute capability 7.0 (Volta) or newer is expected.
- **PyTorch 2.x** with CUDA support. The memory-snapshot API
  (`torch.cuda.memory._record_memory_history`) is available in recent
  2.x releases; if your version is older, upgrade before Exercise 04.

If you do not have GPU access, run on a short-lived cloud instance — the
exercises are measured in minutes of GPU time, not hours. Profiling
specifically will not work on CPU-only setups.

## Module structure

The lecture is split across three chapter files in the
[`lecture-notes/`](lecture-notes/) directory. Read in order:

| Chapter | What's in it |
|---|---|
| [`lecture-notes/01-nsys-nsight.md`](lecture-notes/01-nsys-nsight.md) | The two profilers and when to use each; the `nsys profile` / `nsys stats` command-line workflow; the key `ncu` sections (Speed of Light, Memory Workload Analysis, Compute Workload Analysis, Source/SASS); the `torch.profiler` API and its TensorBoard trace viewer. |
| [`lecture-notes/02-bottleneck-identification.md`](lecture-notes/02-bottleneck-identification.md) | The roofline as a classification tool, not just a plot; the common bottleneck patterns (long memcpy, launch overhead, single-kernel hotspot, many tiny kernels); reading Speed-of-Light numbers, including the under-utilization case where both compute and memory are low; NVTX ranges as the bridge from your Python code to the profiler timeline. |
| [`lecture-notes/03-memory-profiling.md`](lecture-notes/03-memory-profiling.md) | The PyTorch memory-snapshot API and the `memory_viz` web viewer; the four memory consumers in a training step (activations, gradients, optimizer state, KV cache); the standard memory-reduction levers (gradient checkpointing, mixed precision, 8-bit optimizers, FlashAttention) with their rough memory-vs-speed trade-offs. |

| Exercise | What it covers |
|---|---|
| [`exercises/exercise-01-nsys-trace.md`](exercises/exercise-01-nsys-trace.md) | Capture an `nsys` trace of a ResNet-50 training step, add NVTX ranges around forward / backward / optimizer, and write up the top three hotspot kernels with a fix hypothesis per kernel. |
| [`exercises/exercise-02-ncu-deep-dive.md`](exercises/exercise-02-ncu-deep-dive.md) | Take the top hotspot from Exercise 01 and run `ncu --set full` on it. Report achieved occupancy, memory throughput %, compute throughput %, roofline placement, and whether you would optimize for memory or compute. |
| [`exercises/exercise-03-pytorch-profiler.md`](exercises/exercise-03-pytorch-profiler.md) | Profile a small transformer training step with `torch.profiler`. View in TensorBoard. Distinguish DataLoader idle time, Python dispatch overhead, and GPU kernel time; write a five-bullet diagnosis. |
| [`exercises/exercise-04-memory-snapshot.md`](exercises/exercise-04-memory-snapshot.md) | Capture a memory snapshot of a GPT-2 training step and open it in `memory_viz`. Report peak memory, largest single allocation, whether the loop is in steady state, and one concrete optimization recommendation grounded in the snapshot. |
| [`exercises/exercise-05-end-to-end-optimization.md`](exercises/exercise-05-end-to-end-optimization.md) | Pick a slow training script and run the full cycle — `nsys` to find the phase bottleneck, `ncu` to find the kernel bottleneck, apply one optimization, re-profile, repeat — until p95 step time is ≥2× better. Deliver before/after traces and an `OPTIMIZATION_LOG.md`. |

Plus:

| File | What's in it |
|---|---|
| [`quiz.md`](quiz.md) / [`quiz-answers.md`](quiz-answers.md) | 10-question multiple-choice check on the learning objectives. 70% to pass. |
| [`resources.md`](resources.md) | Primary sources — the Nsight Systems docs, the Nsight Compute docs, the PyTorch profiler recipe, the `memory_viz` viewer, and the GPU MODE community lectures. |

Companion repo:
[`engineer-solutions/mod-107 ex-07 (gpu-memory-profiling)`](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-107-gpu-computing/exercise-07-gpu-memory-profiling)
ships a baseline training script, an optimized version, and the
snapshot-capture scaffolding. Useful after Exercise 04; avoid it before.

## How to work through this module

1. Read the three chapter files in `lecture-notes/` in order. Plan for
   roughly 4–5 hours of reading plus hands-on. Chapter 01 includes the
   `nsys` and `ncu` command invocations you will use again and again —
   type them out against any kernel you have at hand, not just the
   exercise workloads.
2. Do the five exercises in order. The workloads escalate in realism:
   Exercise 01 is a torchvision model, Exercise 03 moves to a
   transformer, Exercise 04 is a GPT-2 training step, and Exercise 05 is
   a full optimization cycle. Exercises 01 and 02 share a trace; keep
   the `.nsys-rep` file around.
3. Expect that the first few profiles will look like noise. Reading a
   `nsys` timeline — knowing which rows matter, which "idle" gaps are
   real and which are artifacts of the sampling resolution, how NVTX
   ranges line up against CUDA kernel rows — is a learned skill. Give
   each profile ten quiet minutes before you form an opinion.
4. Benchmark honestly. Every measurement in this module assumes warm-up
   iterations, `torch.cuda.synchronize()` between timed runs, and the
   median of several runs rather than a single number. The quiz tests
   this explicitly; the exercises will mislead you if you skip it.
5. Take the quiz **without** scrolling to `quiz-answers.md`. If you miss
   more than three questions, reread the chapter that covers them before
   moving on to Module 04.

## Assessment criteria for this module

You are ready to move on when you can, on a blank page:

- Write the `nsys profile` and `ncu --set full --kernel-id` commands
  for a PyTorch training script, including the two or three flags you
  actually need, from memory.
- Read a Speed-of-Light section from `ncu` and classify the kernel as
  memory-bound, compute-bound, or under-utilized — with a sentence of
  evidence for each — in under a minute.
- Add NVTX ranges to an existing PyTorch training loop around the
  forward, backward, and optimizer phases, and explain why that
  annotation is almost always the first change you make before
  profiling anything non-trivial.
- Describe the four main consumers of GPU memory during training
  (activations, gradients, optimizer state, KV cache for inference),
  give the standard lever that reduces each, and name the speed cost of
  that lever.
- Run the full measure → hypothesize → change → re-measure cycle
  without supervision, and know when to stop — when the next profile
  shows a different bottleneck has taken over, or the remaining time is
  in kernels you cannot reasonably rewrite.

Module 04 picks up from the profiler-first posture and applies it to
transformers specifically: FlashAttention, KV-cache optimization, kernel
fusion, paged attention. Every optimization there is justified by the
kind of profile you learn to read in this module.
