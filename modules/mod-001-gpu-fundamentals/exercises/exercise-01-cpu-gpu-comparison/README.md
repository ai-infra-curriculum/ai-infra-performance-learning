# Exercise 1 — When does the GPU actually help?

> **Targets learning objectives:** 1, 4
> **Time:** ~45 min
> **Requires:** Python 3.10+, numpy. No GPU needed.

## What you'll do

You will be given three concrete workloads. For each, you predict —
*without writing CUDA* — whether a modern data-center GPU will outperform
a modern server CPU, and by approximately how much. You will defend each
answer with a one-paragraph argument grounded in the SIMT execution
model.

The point of this exercise is to make you justify GPU-vs-CPU decisions
the way they get justified in real engineering reviews: with arithmetic
intensity, memory bandwidth, and parallelism analysis — not with vibes.

## The workloads

1. **`dense_matmul`** — Multiply two FP32 matrices of shape (8192, 8192).
   The output is FP32. Inputs and outputs all live in device memory
   already (no host-device transfer cost in your analysis).

2. **`json_parse`** — Parse a single 1-GB JSON file containing a deeply
   nested object hierarchy. Output is a Python dict.

3. **`elementwise_relu`** — Apply `y = max(0, x)` to a single FP32
   tensor of shape (1024, 1024). Input and output already live on the
   appropriate device (CPU memory for the CPU case, device memory for
   the GPU case).

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
prose — that is what the solutions repo's worked answer is for) and
prints `PASS` when all three are right.

## Hints

- For `dense_matmul`: how many independent multiply-accumulates are in
  the output? How many bytes of HBM are read? Compute the arithmetic
  intensity and place it relative to an A100's ridge point (~9.75
  FLOP/byte for FP32).
- For `json_parse`: how much arithmetic per byte of input? How many
  truly independent sub-tasks are there? Is the workload going to live
  on the GPU's arithmetic units or on its control-flow logic?
- For `elementwise_relu`: arithmetic intensity is ~0.125 FLOP/byte
  (one compare, one select per FP32). The interesting question is
  *kernel launch overhead vs total work*. How many bytes does the
  GPU move for a (1024, 1024) FP32 tensor? At HBM bandwidth, how long
  does that take? Compare to a 5–20 µs kernel launch.

## What "right" looks like

This is a reasoning exercise; the buckets are coarse on purpose. There
is one defensible answer per workload at this granularity, and the
solutions repo's `SOLUTION.md` walks through the derivation in full.
