# Module 5 — Model Compression

> **Track:** AI/ML Performance Engineer
> **Time budget:** ~35 hours (3–4 weeks part-time)
> **Module status:** first build, 2026-11-06

Module 04 made the forward pass faster by changing *how* the model
computes — FlashAttention, kernel fusion, PagedAttention, speculative
decoding. Module 05 attacks the same wall from the other side: making
the model itself smaller. A 7B-parameter model in fp16 is ~14 GB of
weights; the same model in int4 (AWQ or GPTQ) is ~4 GB, which is the
difference between "needs an A100-80GB" and "fits on a single L4" —
and every byte you do not have to pull out of HBM is a byte that does
not cost latency or bandwidth.

Four families of compression matter in production today, and this
module covers each in order of blast radius: **quantization** (change
the number format, keep the weights), **pruning** (keep the format,
zero out weights), **distillation** (change the architecture, train a
smaller model to mimic a larger one), and **parameter-efficient
fine-tuning via LoRA** (keep the base model frozen, train a small
low-rank update on top). The posture is: measure the quality hit,
prove it is tolerable on a task-relevant eval, then take the serving
win. "The MMLU score dropped 0.3 points" is not a conclusion; "the
MMLU score dropped 0.3 points *and* the memory footprint dropped 3.5×"
is.

Every method in this module composes with the Module 04 optimizations
rather than replacing them. A model that is already running
FlashAttention + PagedAttention + continuous batching is a reasonable
candidate for AWQ int4; the resulting stack is roughly multiplicative
on top of a vanilla `HuggingFace.generate()` baseline. The measurement
discipline from Module 03 — warm-up, synchronize, median of several
runs, before/after profiles — is still the thing that keeps the
numbers honest.

## Learning objectives

By the end of this module, you will be able to:

1. **Explain**, in one minute, what AWQ does differently from naive
   per-tensor int4 quantization — activation-aware per-channel scaling
   that protects the ~1% of weight channels that dominate output
   quality — and why that preserves accuracy on large LLMs. Source:
   AWQ paper ([2306.00978](https://arxiv.org/abs/2306.00978)).
2. **Compare** the three canonical int4 weight-only quantizers (AWQ,
   GPTQ, bitsandbytes NF4) along four axes: calibration data required,
   quantization wall-clock time, reported accuracy, and runtime
   support in vLLM / TensorRT-LLM.
3. **Apply** post-training static int8 quantization to a vision model
   with PyTorch's `torch.ao.quantization` flow — fuse, prepare,
   calibrate, convert — and report size, latency, and accuracy before
   and after.
4. **Identify** which pruning variants have real hardware acceleration
   (2:4 structured sparsity on NVIDIA Ampere+ Sparse Tensor Cores) and
   which do not (unstructured magnitude pruning on dense GPUs), and
   explain why the published speedups for the latter are almost
   always theoretical.
5. **Write** the distillation loss as a weighted combination of
   cross-entropy with the ground-truth label and KL divergence with
   the teacher's softened softmax, and state the two conditions that
   make a distillation campaign worth running (a strong teacher; a
   student architecture expressive enough for the target task).
6. **Train** a LoRA adapter on a mid-sized LLM, compute the parameter
   count vs the base model, and serve the result with vLLM's
   `--enable-lora` hot-swap so multiple adapters share one base model
   in GPU memory. Source: LoRA paper
   ([2106.09685](https://arxiv.org/abs/2106.09685)).
7. **Stack** compression techniques — e.g. AWQ int4 weights plus 2:4
   sparsity, or QLoRA (4-bit NF4 base + LoRA) for fine-tuning — and
   reason about where the savings compound cleanly and where the
   kernels stop composing.

Every exercise targets one or more of these; the quiz checks the
underlying concepts.

## Prerequisites

You should be comfortable with:

- **Module 02** — writing and launching a CUDA/Triton kernel. You do
  not write a quantization kernel from scratch in this module, but the
  exercises depend on understanding what an int4 GEMM actually does
  under the hood (dequantize-on-the-fly into shared memory, then a
  regular tensor-core matmul), and that intuition is what Module 02
  builds.
- **Module 03** — `nsys`, `ncu`, memory snapshots, and the
  before/after profiling discipline. Every compression claim in this
  module is scored with a measurement; "the file is smaller" is a
  necessary but not sufficient check.
- **Transformer and vision model basics** — multi-head attention, the
  KV cache, conv-bn-relu blocks, ImageNet-style classification. You
  do not need to re-derive training; you do need to know what you are
  pointing a quantizer at.
- **A working CUDA toolchain + GPU access** — recent PyTorch 2.x with
  CUDA, plus `autoawq`, `bitsandbytes`, and `peft` installed. Compute
  capability 8.0 (Ampere) or newer is required for Exercise 03 (2:4
  sparsity on Sparse Tensor Cores) and strongly recommended for the
  AWQ and LoRA exercises; older GPUs will fall back to slower
  emulation paths that distort the speedup numbers.
- **A task-representative eval set** — for every compression exercise
  you will report an accuracy delta, and the delta is only as
  trustworthy as the eval. An MMLU subset is acceptable for the LLM
  exercises if you have nothing else; an ImageNet validation subset
  is the reference for Exercise 02. Mismatched evals (a chat model
  judged only on perplexity, an instruction-tuned model judged only
  on MMLU) are a common way to ship a quietly-broken quantization.

If you do not have the right GPUs locally, a short-lived cloud
instance is cheaper than the hardware churn; the exercises are
measured in minutes to low single-digit hours of GPU time, not days.

## Module structure

The lecture is split across three chapter files in the
[`lecture-notes/`](lecture-notes/) directory. Read in order:

| Chapter | What's in it |
|---|---|
| [`lecture-notes/01-quantization.md`](lecture-notes/01-quantization.md) | Why smaller weights are faster (HBM capacity *and* bandwidth, not just disk size); the numeric formats you actually see in practice (fp32 / bf16 / fp16 / int8 / int4) and what each costs in accuracy; the four canonical methods — dynamic, static with calibration, QAT, and weight-only int4 (AWQ, GPTQ) — with the tooling to run each and the runtime that consumes the output. |
| [`lecture-notes/02-pruning-distillation.md`](lecture-notes/02-pruning-distillation.md) | Unstructured vs structured pruning and the hardware reality that only 2:4 structured sparsity has real speedups on Ampere+ Sparse Tensor Cores; the distillation loss as CE-plus-KL on softened logits; the common model-family shapes (BERT-large → DistilBERT, Llama-70B → Llama-7B); when distillation pays and when it does not. |
| [`lecture-notes/03-lora-adapters.md`](lecture-notes/03-lora-adapters.md) | LoRA as a frozen-base plus low-rank update (`W + A·B` with `r ≪ d`) and the ~1000× trainable-parameter reduction it buys; the three operational wins (cheap fine-tuning, multi-tenant serving, cheap experimentation); QLoRA as 4-bit NF4 base + LoRA, and the single-24GB-GPU fine-tuning regime it unlocks; vLLM's `--enable-lora` hot-swap for one-base-many-adapters serving. |

| Exercise | What it covers |
|---|---|
| [`exercises/exercise-01-awq-quantize.md`](exercises/exercise-01-awq-quantize.md) | Quantize Mistral-7B to int4 with AutoAWQ. Serve the result with vLLM's `--quantization awq`. Measure file size, inference throughput, and a quality proxy (an MMLU subset or a task-specific eval). Report percentage change on all three. |
| [`exercises/exercise-02-int8-static.md`](exercises/exercise-02-int8-static.md) | Post-training static int8 quantization of ResNet-50 via `torch.ao.quantization` — fuse conv-bn-relu, insert observers, run a 100–500 image calibration pass, convert. Target ~3.5–4× size reduction, ≥1.5× CPU latency win, ≤1.0 pp top-1 accuracy drop. |
| [`exercises/exercise-03-2-4-sparsity.md`](exercises/exercise-03-2-4-sparsity.md) | Apply 2:4 semi-structured sparsity to a transformer (e.g. DeBERTa) and verify end-to-end inference speedup on Ampere+ hardware. Follows the PyTorch 2:4-sparsity prototype tutorial. |
| [`exercises/exercise-04-lora-finetune.md`](exercises/exercise-04-lora-finetune.md) | Fine-tune Mistral-7B with a LoRA adapter (`peft.LoraConfig`) on a small domain-specific dataset. Compare adapter size (MB) to base-model size (GB). Serve with vLLM `--enable-lora` and hot-swap the adapter at request time. |
| [`exercises/exercise-05-distillation.md`](exercises/exercise-05-distillation.md) | Distill a 13B teacher into a 7B (or custom smaller-architecture) student with a CE + KL loss on teacher logits. Report benchmark score vs the teacher and the end-to-end inference speedup. Document dataset, loss, and hyperparams in `DISTILLATION_REPORT.md`. |

Plus:

| File | What's in it |
|---|---|
| [`quiz.md`](quiz.md) / [`quiz-answers.md`](quiz-answers.md) | 10-question multiple-choice check on the learning objectives. 70% to pass. |
| [`resources.md`](resources.md) | Primary sources — the AWQ, GPTQ, LoRA, QLoRA, and DistilBERT papers; the AutoAWQ, PEFT, and bitsandbytes libraries; the NVIDIA 2:4-sparsity write-up; the PyTorch static-quantization tutorial. |

Companion repo:
[`engineer-solutions/mod-110`](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-110-llm-infrastructure)
holds a LoRA / QLoRA / DDP fine-tuning scaffold (ex-05) and a full
inference-optimization chain including AWQ quantization and vLLM
serving (ex-06) that mirror Exercises 01, 04, and 05 in
production-shaped detail. Useful after a first pass at the exercises
here; avoid it before.

## How to work through this module

1. Read the three chapter files in `lecture-notes/` in order. Plan
   for roughly 4–5 hours of reading plus small code-alongs. Chapter
   01 is the longest and the most paper-dense — the AWQ
   "salient-channel" intuition and the int8 calibration flow are
   both worth slowing down on, because every later quantization
   decision rests on them.
2. Do the five exercises in order. Exercise 01 (AWQ) is the cheapest
   to run and the most generalizable; Exercise 02 (int8 static) is
   the only CPU-friendly exercise and is useful for building
   intuition about calibration before you point AWQ at an LLM.
   Exercise 03 (2:4 sparsity) needs Ampere+ hardware; Exercises 04
   and 05 need a fine-tuning-capable GPU with enough memory for the
   base model (24 GB is the usable floor with QLoRA; 40+ GB is more
   comfortable without it).
3. Profile every change. "I turned on AWQ and the model got
   smaller" is not what this module is training — the goal is a
   before/after measurement for each exercise using the Module 03
   tooling. Save the profiler outputs and the eval scores together;
   one number without the other is not a result.
4. Verify quality honestly. Every compression method has a quality
   failure mode, and the catastrophic ones are catastrophic on the
   *task* (code generation, medical Q&A, instruction following),
   not on MMLU. Pick an eval that is representative of what the
   model is actually for; if you only have MMLU, say so in the
   write-up and flag it as a known limitation.
5. Benchmark honestly. Every measurement assumes warm-up iterations,
   `torch.cuda.synchronize()` before and after the timed region, and
   the median of several runs. Quantized kernels in particular can
   trigger a one-time dequantization-cache warm-up on first call —
   that call is not the steady-state number you are claiming.
6. Take the quiz **without** scrolling to `quiz-answers.md`. If you
   miss more than three questions, reread the chapter that covers
   them before moving on to Module 06.

## Assessment criteria for this module

You are ready to move on when you can, on a blank page:

- State the AWQ one-line claim (activation-aware per-channel scaling
  protects the ~1% of weight channels that dominate output quality,
  preserving accuracy under int4) and name the one thing it does
  differently from GPTQ (AWQ picks scales from activation
  statistics; GPTQ minimizes layer-wise output error via an
  approximate Hessian).
- Walk through the PyTorch static-quantization flow —
  `fuse_modules` → `qconfig` → `prepare` → calibration loop →
  `convert` — and name the artifact each step produces.
- Name the one pruning variant with real hardware acceleration (2:4
  structured sparsity on NVIDIA Ampere+ Sparse Tensor Cores) and
  explain in a sentence why unstructured magnitude pruning rarely
  helps in production on dense GPUs (sparse-matrix kernels on dense
  GEMM hardware still read the zeros).
- Write the distillation loss as
  `α · CE(student, label) + β · KL(student_softmax_T, teacher_softmax_T)`
  with temperature `T` on the softened logits, and state the two
  prerequisites for the campaign to be worth running (a teacher that
  already does the task well; a student architecture expressive
  enough to learn it).
- Explain LoRA's parameter-count math (`A ∈ R^{d×r}`, `B ∈ R^{r×d}`,
  trainable params per adapted matrix ≈ `2 · d · r`) for a 7B model
  at rank 16, and explain why the adapter is a few tens of megabytes
  on disk while the base model is ~14 GB in fp16.
- Sketch, with rough numbers, how AWQ int4, 2:4 sparsity, and LoRA
  compose on top of a FlashAttention + continuous-batching serving
  stack, and name one combination that compounds cleanly (AWQ base
  + LoRA adapters: the kernels already dequantize weights into the
  compute path) and one where the composition is more delicate
  (weight-only int4 + 2:4 sparsity requires kernel support for
  *both* at once, which not every runtime ships).

Module 06 picks up here and moves from *making one model smaller* to
*making one model span many GPUs*: tensor parallelism, pipeline
parallelism, expert parallelism, and multi-replica autoscaling. The
compression wins from this module carry over directly — a model that
is already AWQ-quantized needs fewer GPUs per replica, which changes
the parallelism arithmetic — and the measurement discipline from
Module 03 is still the thing that tells you whether the resulting
stack is worth the operational complexity.
