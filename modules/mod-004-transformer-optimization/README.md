# Module 4 — Transformer Optimization

> **Track:** AI/ML Performance Engineer
> **Time budget:** ~40 hours (3–4 weeks part-time)
> **Module status:** first build, 2026-05-25

Modules 01–03 built the generic GPU-performance toolkit: a mental model
of the hardware, the ability to author a kernel, and the measurement
discipline to tell a real improvement from a vibe. Module 04 points that
toolkit at the workload that actually pays for most GPU clusters —
transformer inference and training — and walks through the optimizations
that matter for it. FlashAttention, KV-cache sizing, PagedAttention,
kernel fusion with `torch.compile` and Triton, speculative decoding,
continuous batching, and prefix caching.

The module's spine is a bias toward **adopting** the tooling that already
works — `torch.nn.functional.scaled_dot_product_attention`, `torch.compile`,
vLLM — before you reach for a hand-written kernel. The published
benchmarks from the vLLM paper and the FlashAttention papers show that
the first ~80% of the available speedup comes from flipping the right
switches, not from writing CUDA. The Triton exercise at the end exists
so you have one experience of authoring a fused kernel yourself, so that
when you later profile a kernel that no library has fused you know what
the fix looks like.

Every optimization here is justified by the kind of profile you learned
to read in Module 03. When Exercise 01 shows FlashAttention cutting peak
memory on long sequences, that is the O(N²) activation that a Module 03
memory snapshot would have flagged. When Exercise 02 shows `torch.compile`
collapsing many tiny kernels into one, that is the "many small kernels"
pattern that the `nsys` chapter named. Module 04 is Module 03's recipe
list.

## Learning objectives

By the end of this module, you will be able to:

1. **Explain**, in one minute, why standard attention is O(N²) in both
   memory and compute, and how FlashAttention's tiled online-softmax
   formulation drops the memory term to O(N) without changing the math.
   Source: FlashAttention v1 / v2 papers ([2205.14135](https://arxiv.org/abs/2205.14135),
   [2307.08691](https://arxiv.org/abs/2307.08691)).
2. **Replace** a hand-written attention block in a PyTorch model with
   `torch.nn.functional.scaled_dot_product_attention`, verify numerical
   equivalence, and report the speedup and memory saving at three
   sequence lengths.
3. **Compute** the KV-cache size for a given model + request shape from
   the formula `batch × seq_len × num_layers × num_heads × head_dim × 2
   × precision_bytes`, and use that number to decide whether the
   bottleneck is HBM capacity or HBM bandwidth.
4. **Describe** PagedAttention's page-table-over-KV-cache design and why
   it lifts the concurrent-request ceiling compared to the contiguous
   allocator in naive serving stacks. Source: vLLM paper
   ([2309.06180](https://arxiv.org/abs/2309.06180)).
5. **Apply** `torch.compile(model, mode="reduce-overhead")` to a model,
   warm up correctly before timing, and measure the speedup over eager
   mode at multiple sequence lengths.
6. **Write** a fused elementwise or reduction kernel in Triton using the
   `@triton.jit` decorator, benchmark it against the equivalent
   `torch.*` call, and reason about when a hand-written kernel is worth
   the maintenance cost.
7. **Serve** a model with vLLM's prefix caching and speculative decoding
   enabled, design a workload that exercises each feature, and measure
   TTFT / throughput against the baseline.

Every exercise targets one or more of these; the quiz checks the
underlying concepts.

## Prerequisites

You should be comfortable with:

- **Module 02** — writing a CUDA (or Triton) kernel and launching it
  from PyTorch. The Triton exercise here assumes you can already reason
  about block size, shared memory, and warp-level reductions in some
  language; Triton just changes the syntax.
- **Module 03** — `nsys`, `ncu`, and the PyTorch profiler. Every
  optimization in this module is scored with before/after profiles, and
  "it felt faster" is not a passing answer. If you cannot read a
  Speed-of-Light section from Module 03, do it first.
- **Transformer basics** — multi-head attention, the KV projections,
  causal masking, autoregressive decoding. You do not need to
  re-derive the backward pass, but you do need to know what Q, K, V, and
  the KV cache are. If you can run a minimal GPT-2 training loop in
  PyTorch, you are ready.
- **A working CUDA toolchain + GPU access** — recent PyTorch 2.x with
  CUDA, Triton installed (it ships with PyTorch 2.x), and at least one
  NVIDIA GPU with compute capability 8.0 (Ampere) or newer for
  FlashAttention and most of the kernel exercises. Older GPUs will run
  SDPA in a slower math-fallback path, which defeats the point of the
  benchmarking exercises.
- **Optional for Exercises 03 and 04**: enough GPU memory to serve a
  mid-size model with vLLM. A single A100-40GB or L40S handles
  7B-parameter models in bf16; the 70B + draft-model configuration in
  Exercise 04 needs 4× such GPUs with tensor parallelism. If you do not
  have that hardware, substitute a 7B target + 1B draft pair on a single
  GPU — the measured speedup shape is the same.

If you do not have the right GPUs locally, a short-lived cloud instance
is cheaper than the hardware churn; the exercises are measured in
minutes to low single-digit hours of GPU time, not days.

## Module structure

The lecture is split across three chapter files in the
[`lecture-notes/`](lecture-notes/) directory. Read in order:

| Chapter | What's in it |
|---|---|
| [`lecture-notes/01-attention-optimization.md`](lecture-notes/01-attention-optimization.md) | Why attention is the bottleneck (O(N²) memory and compute); the FlashAttention tiled-SRAM reformulation and why its intermediate N×N matrix never lives in HBM; the KV-cache size formula and a worked example; PagedAttention's page-table design and the concurrent-request-count win it unlocks. |
| [`lecture-notes/02-kernel-fusion.md`](lecture-notes/02-kernel-fusion.md) | Why one-op-per-kernel costs you launch overhead and HBM round-trips; the four routes to fusion (`torch.compile`, Triton, hand-written CUDA, legacy `torch.jit.script`) and which to try first; the `mode="reduce-overhead"` CUDA-graph path; a minimal Triton kernel with `tl.program_id`, `tl.load`, `tl.store`; the canonical fusion opportunities in a transformer block. |
| [`lecture-notes/03-speculative-decoding.md`](lecture-notes/03-speculative-decoding.md) | Speculative decoding as a draft-and-verify loop (small draft proposes N tokens, large target verifies in one forward pass); the shared-tokenizer and distribution-match requirements; continuous batching as the alternative to length-padded batches; prefix caching for shared system prompts; how these optimizations compose roughly multiplicatively on top of FlashAttention. |

| Exercise | What it covers |
|---|---|
| [`exercises/exercise-01-flash-attention.md`](exercises/exercise-01-flash-attention.md) | Swap a hand-rolled attention block for `F.scaled_dot_product_attention` in a small transformer training loop. Benchmark at seq_len ∈ {512, 2048, 8192}. Verify numerical equivalence within tolerance. Report speedup and peak-memory delta. |
| [`exercises/exercise-02-torch-compile.md`](exercises/exercise-02-torch-compile.md) | Apply `torch.compile(model, mode="reduce-overhead")` to a HuggingFace model. Benchmark eager vs compiled at three sequence lengths. Warm up correctly — the first call compiles and is not the number you report. |
| [`exercises/exercise-03-vllm-prefix-caching.md`](exercises/exercise-03-vllm-prefix-caching.md) | Launch vLLM with `--enable-prefix-caching`. Build a 1000-request workload that shares a 500-token system prompt + varying 50-token user queries. Measure TTFT and throughput with and without the flag. |
| [`exercises/exercise-04-speculative-decoding.md`](exercises/exercise-04-speculative-decoding.md) | Serve a 70B target + 7B draft via vLLM speculative decoding. Benchmark vs the 70B alone on long and short generations. Verify generation quality is unchanged on a held-out set. |
| [`exercises/exercise-05-write-triton-kernel.md`](exercises/exercise-05-write-triton-kernel.md) | Implement a fused LayerNorm in Triton following the [Triton LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html). Benchmark against `torch.nn.LayerNorm` at hidden_dim = 4096; target within 20% of the PyTorch op. |

Plus:

| File | What's in it |
|---|---|
| [`quiz.md`](quiz.md) / [`quiz-answers.md`](quiz-answers.md) | 10-question multiple-choice check on the learning objectives. 70% to pass. |
| [`resources.md`](resources.md) | Primary sources — the FlashAttention v1 / v2 papers, the vLLM / PagedAttention paper, the speculative-decoding paper, the vLLM docs, the Triton tutorials, and HuggingFace `text-generation-inference`. |

Companion repo:
[`engineer-solutions/mod-110`](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-110-llm-infrastructure)
holds 14 LLM-infrastructure exercises that mirror Exercises 03–05 of
this module in more detail (vLLM deployment, speculative decoding,
Triton kernel scaffolding). Useful after you have taken a first pass at
the exercises here; avoid it before.

## How to work through this module

1. Read the three chapter files in `lecture-notes/` in order. Plan for
   roughly 5–7 hours of reading plus the small code-alongs. Chapter 01
   is the longest — the KV-cache formula and the PagedAttention
   explanation are both worth slowing down on, because every later
   capacity-planning conversation on an inference team rests on them.
2. Do the five exercises in order. Exercises 01 and 02 are
   drop-in-replacement exercises on a training-shaped workload and are
   the cheapest to run. Exercises 03 and 04 serve a model with vLLM and
   want a few GB of free GPU memory each. Exercise 05 is a Triton kernel
   and takes the longest wall-clock time to get right; schedule it for a
   block when you can sit with it.
3. Profile every change. "I turned on FlashAttention and it was faster"
   is not what this module is training — the goal is to produce a
   before/after profile for each exercise using the Module 03 tooling.
   Save the `nsys` and `ncu` outputs with the exercise write-ups; they
   are the artifact that proves the change paid off.
4. Benchmark honestly. Every measurement assumes warmup iterations,
   `torch.cuda.synchronize()` before and after the timed region, and the
   median of several runs. `torch.compile` in particular compiles on
   the first call — a timing that includes compile time is not the
   steady-state number you are claiming. Exercise 02 fails without
   warmup; so do most of the others if you are not careful.
5. Verify correctness before you celebrate speed. FlashAttention
   matches standard attention within floating-point tolerance, not
   exactly; `torch.compile` can silently change numerics if a decorator
   is wrong; speculative decoding should produce the *same* output
   distribution as the target model (that is what makes it lossless).
   Every exercise lists a correctness check — run it.
6. Take the quiz **without** scrolling to `quiz-answers.md`. If you miss
   more than three questions, reread the chapter that covers them before
   moving on to Module 05.

## Assessment criteria for this module

You are ready to move on when you can, on a blank page:

- State the FlashAttention memory claim (O(N²) → O(N)) and the
  mechanism (tiled computation + online softmax in SRAM) in two or
  three sentences, without reaching for the paper.
- Compute the KV-cache size in GB for a given `(batch, seq_len,
  num_layers, num_heads, head_dim, precision)` tuple from memory, and
  state whether the bottleneck at that shape is going to be capacity or
  bandwidth.
- Describe PagedAttention in the vocabulary of operating-system virtual
  memory — pages, page tables, no requirement that a request's KV cache
  be physically contiguous — and name the metric it improves (concurrent
  requests per GPU).
- Write a minimal Triton kernel skeleton — `@triton.jit`, `pid =
  tl.program_id(0)`, offsets, mask, `tl.load` / `tl.store` — for an
  elementwise op from memory, and explain in a sentence why you would
  reach for Triton before hand-written CUDA.
- Sketch, with rough numbers, how FlashAttention, continuous batching,
  prefix caching, and speculative decoding compose on top of a vanilla
  HuggingFace `generate()` baseline (roughly multiplicative, 20–30×
  realistic on published vLLM benchmarks), and explain why one of them
  can regress short-sequence latency even when it wins on long sequences.

Module 05 picks up here and moves from *computing attention faster* to
*making the model itself smaller* — post-training quantization,
quantization-aware training, structured pruning, and TensorRT. The
compounding story continues: a model that is already running
FlashAttention + continuous batching + prefix caching is a reasonable
candidate for int8 or int4 compression, and the measurement discipline
from Module 03 is still the thing that keeps the resulting numbers
honest.
