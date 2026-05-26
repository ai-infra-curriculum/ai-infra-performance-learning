# Exercise 1 — When does the GPU actually help?

> **Targets learning objectives:** 1, 4
> **Time:** ~45–60 min
> **Requires:** Python 3.10+, numpy. No GPU needed.
> **References:** lecture-notes/01-gpu-architecture.md (SIMT, when GPUs don't help) and lecture-notes/05-roofline-model.md (arithmetic intensity).

## What you'll do

You will be given three concrete workloads. For each, you predict —
*without writing CUDA* — whether a modern data-center GPU will
outperform a modern server CPU, and by approximately how much. You
will defend each answer with a one-paragraph argument grounded in the
SIMT execution model.

The point of this exercise is to make you justify GPU-vs-CPU
decisions the way they get justified in real engineering reviews:
with arithmetic intensity, memory bandwidth, and parallelism
analysis — not with vibes.

## Why this skill matters

Every team running ML infrastructure makes "GPU or CPU?" decisions
constantly. Examples from the field:

- A data preprocessing step takes 4 hours overnight. Engineer A says
  "move it to GPU, it's a tensor library, should be fast." Engineer
  B says "the work is mostly file I/O and string parsing, GPU won't
  help." Without an analytical framework, this becomes an
  authority/intuition argument, not an engineering one.
- A new model architecture has a small attention head running on
  every layer. Some engineer asks "should we keep this on the GPU
  or offload to CPU?" The right answer depends on arithmetic
  intensity, batch size, and host-device transfer cost — not
  on which sounds cooler.
- An inference service has latency tail-spikes. Profiling shows the
  spikes are from a tokenizer running on the GPU. Why is the
  tokenizer on the GPU at all? Because someone assumed "everything
  in the inference path goes on the GPU." That assumption is
  expensive — tokenizers are pure control flow and pure I/O.

The decision *framework* is what you'll practice here. The numbers
are rough; the reasoning is precise.

## The three workloads

You'll evaluate each of these workloads independently:

### 1. `dense_matmul`

Multiply two FP32 matrices of shape (8192, 8192). The output is FP32.
Inputs and outputs all live in device memory already (no host-device
transfer cost in your analysis).

Things to think about:

- How many independent multiply-accumulates are in the output?
- How many bytes of HBM are read?
- What's the arithmetic intensity?
- Where does that AI land relative to the ridge point of an A100
  (~9.6 FLOP/byte for FP32)?

### 2. `json_parse`

Parse a single 1-GB JSON file containing a deeply nested object
hierarchy. Output is a Python dict.

Things to think about:

- How much arithmetic per byte of input? (Hint: ~zero. JSON parsing
  is character classification, state transitions, allocation —
  basically no FLOPS.)
- How many truly independent sub-tasks are there? The "next" token
  depends on the "current" token in the parser state machine.
- Will the workload live on the GPU's arithmetic units or on its
  control-flow logic? (The GPU has very little of the latter.)
- What about kernel launch overhead and host-device transfer for
  the input bytes?

### 3. `elementwise_relu`

Apply `y = max(0, x)` to a single FP32 tensor of shape (1024, 1024).
Input and output already live on the appropriate device (CPU memory
for the CPU case, device memory for the GPU case).

Things to think about:

- Arithmetic intensity is ~0.125 FLOP/byte (one compare/select per
  FP32 element, 8 bytes read+write per element).
- The interesting question is *kernel launch overhead vs total work*.
  How many bytes does the GPU move for a (1024, 1024) FP32 tensor?
  At HBM bandwidth, how long does that take? Compare to a 5–20 µs
  kernel launch.
- What does the CPU do well at this size? (Small tensor, cache-hot,
  no kernel-launch tax.)

## Decision framework

For each workload, ask in order:

1. **What is the arithmetic intensity?** FLOPS per byte. Property of
   the algorithm.
2. **Where is the ridge point of the candidate GPU?** Compute peak ÷
   bandwidth peak.
3. **Is this compute-bound or memory-bound on the GPU?**
4. **How much parallelism is exposed?** Number of independent work
   items. If it's less than a few thousand, GPU launch overhead may
   eat any win.
5. **Are there control-flow / data-dependent paths?** GPUs are bad
   at irregular branches.
6. **Is host↔device transfer in the picture?** PCIe Gen4 x16 caps at
   ~32 GB/s, which is 60× slower than HBM. If the workload includes
   transfer, that's often the dominant cost.

If the first answer is "memory-bound and the AI is very low" you
either need fusion (combining adjacent ops to amortize HBM cost) or
the GPU isn't a meaningful win.

If the second answer is "no parallelism" or "deeply data-dependent"
the GPU isn't going to help, period.

## What to submit

Edit `starter.py`. For each workload, fill in:

- `gpu_helps: bool` — Will the GPU outperform a comparable CPU?
- `expected_speedup_bucket: str` — One of `"<1x"`, `"1-2x"`,
  `"2-10x"`, `"10-100x"`, `">100x"`.
- `reasoning: str` — One paragraph (3–6 sentences). At minimum,
  identify (a) is this compute-bound or memory-bound, (b) how much
  parallelism is available, (c) what kills it on the GPU if anything.

Then run:

```bash
python check.py
```

The check looks at your bucketed answer (it does **not** grade the
prose — that's what the solutions repo's worked answer is for) and
prints `PASS` when all three are right.

## Hints

### For `dense_matmul`

```
total_flops = 2 * M * N * K = 2 * 8192^3 ≈ 1.1 * 10^12 FLOPS
total_bytes_moved = 4 * (M*K + K*N + M*N) ≈ 8 * 10^8 bytes
arithmetic_intensity = 1.1e12 / 8e8 ≈ 1365 FLOP/byte
```

Compare to A100 ridge ≈ 9.6. Way above. **Deeply compute-bound.**

How long would a 19.5 TFLOPS A100 take in theory? `1.1e12 / 19.5e12 ≈
57 ms`. cuBLAS achieves ~85% of peak so in practice ~65 ms.

How long on a CPU? Even at 1 TFLOP/sec (very optimistic for FP32
GEMM on a server CPU with MKL), 1.1e12 / 1e12 = 1.1 seconds. So
GPU is **>15× faster**, plausibly >100× if the CPU is
oversubscribed or single-socket.

Bucket: `>100x`.

### For `json_parse`

The pipeline:
- 1 GB of input bytes.
- For each byte, the parser does a state transition — a small switch
  statement, no arithmetic, no parallelism.
- The output is a tree-shaped data structure with pointers, allocated
  by the parser as it goes.

Arithmetic intensity is essentially 0 (no FLOPS at all). Parallelism
is essentially 0 (next state depends on current state and current
byte).

The GPU has no path to win. Even ignoring transfer costs, the
control-flow nature of parsing means the GPU's lanes mostly idle.
The CPU's branch prediction and out-of-order issue dominate.

Bucket: `<1x` (CPU is faster).

### For `elementwise_relu`

```
total_bytes = 4 (read) + 4 (write) per element
            = 8 bytes * 1024 * 1024 = 8 MB

At A100 HBM bandwidth (2039 GB/s):
  time = 8e6 / 2039e9 ≈ 3.9 microseconds

Kernel launch overhead: 5-20 microseconds.
```

The launch alone is 1.5–5× the actual compute time. GPU sits at the
edge of "barely worth it." A warmed-up CPU with vectorized SIMD
does the same work in <1 ms with no kernel-launch tax.

The realistic outcome: 1–2× faster on GPU, or sometimes slower
depending on launch overhead. Either way not a meaningful
speedup.

Bucket: `1-2x` (GPU helps, slightly).

### General reasoning structure

Each `reasoning` field should follow this shape:

> "The arithmetic intensity here is X, which is [above/below] A100's
> ridge of ~9.6, so this kernel is [compute|memory]-bound. The total
> work is N FLOPS, and the GPU at peak does ~X ms of work versus the
> CPU's Y ms, giving a Z× speedup. The dominant cost is
> [arithmetic | bandwidth | control flow | launch overhead]."

You don't have to literally use that template, but those are the
five facts that should appear.

## Common pitfalls

1. **Forgetting kernel-launch overhead.** A 5–20 µs launch is
   invisible on a 100 ms kernel but kills you on a 5 µs kernel.
   Always ask: is this kernel big enough to amortize the launch?

2. **Assuming "GPU is faster on tensor ops"** without checking AI.
   Vector add is a "tensor op" but it's memory-bound and AI = 0.083.
   The GPU isn't free arithmetic; it's expensive bandwidth.

3. **Not noticing host↔device transfer.** If the data starts on the
   CPU and the answer goes back to the CPU, PCIe is often the
   bottleneck.

4. **Confusing "parallel" with "embarrassingly parallel."** JSON
   parsing has lots of independent sub-objects in principle, but
   the parser sees the file as a serial stream — the
   "parallelism" isn't exposed to the kernel.

5. **Picking the wrong bucket boundary.** `2-10x` and `10-100x` are
   different optimization conversations. 5× is mediocre; 50× is a
   GPU win. Try to be honest about which one.

## Grading rubric

The autograder checks (gpu_helps, expected_speedup_bucket) against
known-correct values:

| Workload | gpu_helps | bucket |
|---|---|---|
| dense_matmul | True | `>100x` |
| json_parse | False | `<1x` |
| elementwise_relu | True | `1-2x` |

The `reasoning` prose is not autograded — the solutions repo has
the worked answer for self-comparison. Two of the three workloads
have a clear "right" answer; `elementwise_relu` is borderline and
the grading bucket reflects the realistic middle.

## What "right" looks like

This is a reasoning exercise; the buckets are coarse on purpose.
There is one defensible answer per workload at this granularity,
and the solutions repo's `SOLUTION.md` walks through the
derivation in full. If you get the bucket wrong but the reasoning
points in the right direction, you've learned the right lesson —
go back, reread the relevant section, and refine.

If the check passes, you've internalized:

- How to compute arithmetic intensity from an algorithm
  description.
- How to compare AI to a GPU's ridge point.
- When kernel launch overhead matters.
- When control flow blocks GPU acceleration.

Those four skills will come up *every single time* you evaluate a
new workload for the GPU.

## Extended case studies

Two more workloads worth thinking through (not graded, but the same
analytical framework). These show how the decision tree scales to
realistic infrastructure questions.

### Case study A: video decoding

You're running a video pipeline that decodes 1080p H.264 frames
at 60 FPS. Each frame is ~6 MB compressed. Should the H.264 decode
happen on the CPU or the GPU?

Applying the framework:

1. **AI:** H.264 decode involves entropy decoding (Huffman, CABAC),
   inverse DCT, deblocking, motion compensation. Total FLOPS per
   byte ≈ ~10-50 depending on the standard. Not zero, but not
   GEMM-tier either.
2. **Parallelism:** Macroblock-level parallelism exists but the
   intra-frame dependencies (motion-compensation pointing to
   neighboring blocks, deblocking depending on quantization
   parameters of neighbors) make it harder than it looks.
3. **Control flow:** Heavy. Entropy decoding has long
   data-dependent branches; CABAC is famously serial.

The answer: **dedicated hardware**, not generic CPU or GPU compute.
Every modern NVIDIA GPU has fixed-function video decode blocks
(NVDEC) that are separate from the CUDA cores. NVIDIA's `nvcuvid`
API uses them. The CUDA-core path would lose to even a desktop
CPU because of the control-flow density; the NVDEC path wins
handily because it's a custom ASIC.

Lesson: "should it go on the GPU?" is sometimes "yes, but on the
*dedicated hardware* of the GPU, not its CUDA cores." Always
check whether a workload has a fixed-function accelerator on the
target chip.

### Case study B: training-step gradient all-reduce

In data-parallel training across 8 GPUs, after each backward pass
you need to all-reduce gradients across the GPUs. Should the
all-reduce run on CPU or GPU?

Applying the framework:

1. **AI:** Gradient all-reduce is just sum across N copies. 1
   FLOP per element. AI = 1 / (4 * N) = vanishingly small for
   FP32. Deeply memory-bound (or bandwidth-bound across the
   interconnect).
2. **Parallelism:** Embarrassingly parallel across tensor
   elements.
3. **Control flow:** Minimal. Just sums.
4. **Data location:** Gradients live on the GPU. Moving them
   over PCIe to CPU and back is **expensive** — PCIe is 30× slower
   than NVLink.

The answer: **GPU**, using NCCL (NVIDIA's collective comms
library) which uses NVLink for the inter-GPU portion. CPU
all-reduce via MPI was the standard before NVLink; now it's a
non-starter for performance at this scale.

But: the *control* of the all-reduce — deciding when to issue
it, coordinating with the optimizer step — typically runs on
the CPU. The *data movement* runs on the GPU/NVLink. This is a
common pattern: control logic on the CPU, bulk data movement on
the GPU.

## Quantitative answer sheet

For your reference (the autograder uses these):

| Workload | gpu_helps | bucket | dominant cost |
|---|---|---|---|
| dense_matmul | True | `>100x` | arithmetic (compute-bound, large parallelism) |
| json_parse | False | `<1x` | control flow + serial dependency |
| elementwise_relu | True | `1-2x` | kernel launch overhead at small size |

If you got all three right, you've internalized:

- A workload's *arithmetic intensity* drives the analysis.
- A workload's *parallelism shape* matters as much as its AI.
- Small problems lose to kernel launch overhead.

These three principles cover ~80% of "should I move this to the
GPU?" decisions in real engineering. The remaining 20% — dedicated
hardware paths, multi-GPU coordination, mixed CPU/GPU pipelines —
are the topic of later modules.

## Related material

- Lecture 1.1 — SIMT model, when GPUs don't help.
- Lecture 1.5 — Arithmetic intensity, the roofline.
- Lecture 1.6 — A100 vs H100 vs RTX 4090 specs.
- NVIDIA NVDEC documentation — for dedicated video decode.
- NCCL programming guide — for multi-GPU collectives.

## Next exercise

Exercise 2 — Compute peak FP32 and HBM bandwidth from spec-sheet
values for three GPUs. Builds the muscle memory for the arithmetic
you did informally in this exercise.
