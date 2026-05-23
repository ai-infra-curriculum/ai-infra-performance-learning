# Exercise 2 — Peak throughput from a spec sheet

> **Targets learning objectives:** 2, 3
> **Time:** ~30–45 min
> **Requires:** Python 3.10+. No GPU needed.
> **References:** lecture-notes/01-gpu-architecture.md (worked example: counting FP32 lanes) and lecture-notes/06-gpu-generations.md (the reference table).

## What you'll do

You will compute two numbers — peak FP32 throughput and peak memory
bandwidth — for three GPUs (A100 SXM4 80GB, H100 SXM5 80GB, RTX 4090)
from their published spec-sheet values.

This is one of those skills that looks trivial until the first time
you have to defend a procurement decision in a meeting and someone
asks "what is the FP32 throughput of an L40 at 90% utilization?"
without a spec sheet handy. After this exercise you don't need to
look it up; you can derive it.

## Why this skill matters

You will rederive these numbers, by hand or in your head, in
roughly these contexts:

- **Sizing a training cluster.** "How many GPU-hours does training
  cost?" → throughput per GPU × number of GPUs × utilization × wall
  clock. Knowing the peak lets you spot when reality is off by 5×
  and dig into why.
- **Reading research papers.** "This kernel achieves 87 TFLOPS on
  an A100" — is that compute-bound near peak, or memory-bound and
  irrelevant? You need the peak in your head.
- **Choosing precision.** "FP32 vs BF16 vs FP8" — the ratio of peak
  throughputs on the candidate GPU directly drives the decision.
- **Debugging slow kernels.** A kernel at 4% of peak is dramatically
  different from one at 80% of peak. The diagnosis path branches
  on that ratio.

The arithmetic is trivial; the *fluency* is the value. After this
exercise, you should be able to do the calculation from memory for
any new GPU NVIDIA ships.

> **Why not tensor cores too?** Tensor-core throughput is
> architecture-specific (third-gen on Ampere, fourth-gen on Hopper)
> and the published TFLOPS often quote *with sparsity*, which is a
> 2× cooked number. Deriving tensor TFLOPS from clock + structure
> is a Module 4 topic, after we have unpacked what tensor cores
> actually do. For now, stay with FP32 and bandwidth — both are
> clean derivations.

## The formulas

**Peak FP32 (dense, no tensor cores):**

```
peak_fp32_tflops = (sm_count * cuda_cores_per_sm * boost_clock_ghz * 2) / 1000
```

Walking through the units:

- `sm_count * cuda_cores_per_sm` = total FP32 CUDA cores on the
  chip.
- Each CUDA core can issue one FMA per clock cycle.
- An FMA (fused multiply-add) counts as **2 FLOPS** — one multiply
  + one add, fused into one instruction. This is where the "×2"
  comes from. Without the FMA, peak FP32 would be half.
- `boost_clock_ghz` is in GHz = 10^9 cycles per second.
- Result is FLOPS / 10^9 = GFLOPS. Divide by 1000 again to get
  TFLOPS.

**Peak memory bandwidth (GB/s):**

```
peak_bandwidth_gbs = (effective_per_pin_gbps * bus_width_bits) / 8
```

Walking through the units:

- `effective_per_pin_gbps` is the data rate per memory pin, in
  gigabits per second.
- `bus_width_bits` is the number of pins (the parallel width of
  the memory bus).
- Multiplied: total Gbps. Divide by 8 bits/byte → GB/s.

The "effective per-pin Gbps" is the data rate per memory pin.
NVIDIA's spec sheets quote it directly for GDDR (e.g. "21 Gbps" for
the RTX 4090's GDDR6X). For HBM2e and HBM3 they quote either the
per-pin Gbps or the total bandwidth — both are equivalent given
the bus width.

| Memory | Approx effective per-pin Gbps | Bus width |
|---|---|---|
| GDDR6X (RTX 4090) | 21.0 | 384-bit |
| HBM2e (A100 SXM4 80GB) | ~3.19 | 5120-bit |
| HBM3 (H100 SXM5 80GB) | ~5.23 | 5120-bit |

You will plug these into the formula and confirm you reproduce
NVIDIA's quoted total bandwidth.

## What to submit

Edit `starter.py`. For each of the three GPUs, fill in the spec
values and implement the two formula functions. Then run:

```bash
python check.py
```

The check verifies your computed peak FP32 and bandwidth values
match NVIDIA's published values within 5% tolerance (allowing for
clock-rate variation between SKUs).

## Hints

### A100 SXM4 80GB

- 108 SMs (Ampere GA100).
- **64 FP32 CUDA cores per SM.** (A100 is the outlier — only
  data-center Ampere has 64; all other modern GPUs have 128.)
- Boost clock ≈ 1.41 GHz.
- HBM2e at ~3.19 Gbps per pin × 5120-bit bus → ~2039 GB/s.

Walking through:
```
peak_fp32 = (108 * 64 * 1.41 * 2) / 1000
          = (108 * 64 * 2.82) / 1000
          = 19,491 / 1000
          ≈ 19.49 TFLOPS  ✓ (matches datasheet 19.5)

peak_bw   = (3.19 * 5120) / 8
          = 16,332.8 / 8
          ≈ 2,042 GB/s    ✓ (matches datasheet 2039 within rounding)
```

### H100 SXM5 80GB

- 132 SMs (Hopper GH100).
- **128 FP32 CUDA cores per SM.**
- Boost clock ≈ 1.98 GHz.
- HBM3 at ~5.23 Gbps per pin × 5120-bit bus → ~3350 GB/s.

Walking through:
```
peak_fp32 = (132 * 128 * 1.98 * 2) / 1000
          ≈ 66.9 TFLOPS  ✓ (matches datasheet 67)

peak_bw   = (5.23 * 5120) / 8
          ≈ 3,347 GB/s   ✓ (matches datasheet 3350)
```

### RTX 4090

- 128 SMs (Ada AD102).
- **128 FP32 CUDA cores per SM.**
- Boost clock ≈ 2.52 GHz.
- GDDR6X at 21 Gbps per pin × 384-bit bus → ~1008 GB/s.

Walking through:
```
peak_fp32 = (128 * 128 * 2.52 * 2) / 1000
          ≈ 82.6 TFLOPS  ✓ (matches datasheet 82.6)

peak_bw   = (21.0 * 384) / 8
          = 8,064 / 8
          = 1,008 GB/s   ✓ (matches datasheet 1008)
```

## Why these specific numbers

You'll be tempted to look up "peak FP32" directly. Don't. The
derivation is the muscle you're building. Three reasons:

1. **NVIDIA's quoted peak TFLOPS sometimes includes weird
   multipliers.** A 2:1 INT/FP combined number. A "with sparsity"
   number on tensor TFLOPS. The dense FP32 number from cores ×
   clock × 2 is unambiguous.
2. **You'll meet new SKUs.** The L4, L40, H200, B100, B200, and
   whatever comes next. Knowing the formula means you can compute
   peak before NVIDIA's marketing team publishes a webpage about
   it.
3. **You'll spot impossible claims.** When someone says "this
   kernel achieved 200 TFLOPS FP32 on an A100" you should know
   immediately that's 10× the chip's peak — and that the speaker
   is either wrong, using tensor cores, or quoting FLOPS for a
   precision they're not stating.

## Common pitfalls

### 1. Forgetting the ×2 for FMA

```
WRONG: peak = sm_count * cores * clock_ghz / 1000   # gives half the answer
RIGHT: peak = sm_count * cores * clock_ghz * 2 / 1000
```

This is the most common mistake. The FMA counts as 2 FLOPS because
the architecture issues both the multiply and the add in one
instruction. Any "peak FP32" number ignoring the FMA undercounts
by 2×.

### 2. Confusing CUDA cores per SM across architectures

| Architecture | FP32 CUDA cores per SM |
|---|---|
| Ampere data-center (GA100, A100) | 64 |
| Ampere consumer (GA102, RTX 30-series) | 128 |
| Hopper (GH100, H100) | 128 |
| Ada (AD102, RTX 40-series, L40S) | 128 |

A100 is the only modern outlier with 64 cores/SM. Get this number
wrong and your peak is 2× off.

### 3. Mixing units on bandwidth

```
WRONG: bw = pin_gbps * bus_bits         # gives Gbits/s, not GB/s
RIGHT: bw = pin_gbps * bus_bits / 8     # 8 bits per byte
```

NVIDIA quotes per-pin in Gbps (gigabits) and total bandwidth in
GB/s (gigabytes). The 8× factor is easy to forget.

### 4. Using base clock instead of boost clock

CUDA cores typically run at boost clock under load. Using the
"base clock" gives you a number that's ~20% too low. Always use
boost.

### 5. Powers-of-2 vs powers-of-10

NVIDIA quotes "GB/s" using powers of 10 (1 GB = 10^9 bytes), not
powers of 2 (1 GiB = 2^30 bytes). Stay in powers of 10 for the
bandwidth calculation to match the datasheet.

## Grading rubric

The autograder checks:

1. All three `GPUSpec` dataclasses are filled in (no Ellipsis
   remaining).
2. `peak_fp32_tflops` returns values within 5% of:
   - A100: 19.5 TFLOPS
   - H100: 67 TFLOPS
   - RTX 4090: 82.6 TFLOPS
3. `peak_memory_bandwidth_gbs` returns values within 5% of:
   - A100: 2039 GB/s
   - H100: 3350 GB/s
   - RTX 4090: 1008 GB/s

The 5% tolerance is generous — it allows for clock-rate variation
between SKUs and rounding in your per-pin values. Exact match is
not the goal; *understanding the derivation* is.

## What "right" looks like

Your numbers should land within 5% of NVIDIA's published spec-sheet
peak values. Exact match is not the goal (clocks vary, SKUs vary);
*understanding the derivation* is.

Bonus exercise (not graded): predict the L40S peak FP32 and
bandwidth. From the lecture-notes reference table: 142 SMs, 128
FP32 cores per SM, ~2.52 GHz boost, GDDR6 at 18 Gbps per pin on a
384-bit bus.

```
peak_fp32 = (142 * 128 * 2.52 * 2) / 1000 ≈ 91.6 TFLOPS
peak_bw   = (18 * 384) / 8 = 864 GB/s
```

These match NVIDIA's L40S spec sheet. Once you can do this in
under 60 seconds for any new GPU, you've internalized the skill.

## Extended worked examples

### Other GPUs (do the math yourself)

After the autograder passes, try these as practice — the formulas
are the same; only the inputs change. Answers at the bottom of
this section.

| GPU | SMs | Cores/SM | Boost GHz | Pin Gbps | Bus width |
|---|---|---|---|---|---|
| **V100 SXM2 32GB** | 80 | 64 | 1.53 | ~1.75 | 4096-bit |
| **T4** | 40 | 64 | 1.59 | 10.0 | 256-bit |
| **A10G** | 80 | 128 | 1.71 | 18.75 | 384-bit |
| **L4** | 60 | 128 | 2.04 | 18.75 | 192-bit |
| **L40S** | 142 | 128 | 2.52 | 18.0 | 384-bit |
| **H200 SXM5 141GB** | 132 | 128 | ~1.98 | ~6.4 | 6144-bit |
| **B100 (preliminary)** | ~120 | 128 | ~1.8 | ~8.0 | 6144-bit |

Compute peak FP32 and bandwidth for each. Check against NVIDIA's
datasheets.

(Answers: V100 ≈ 15.7 TFLOPS / 900 GB/s. T4 ≈ 8.1 TFLOPS / 320
GB/s. A10G ≈ 35 TFLOPS / 900 GB/s. L4 ≈ 31.3 TFLOPS / 300 GB/s.
L40S ≈ 91.6 TFLOPS / 864 GB/s. H200 ≈ 67 TFLOPS / ~4915 GB/s.
B100 numbers are preliminary and subject to change.)

### Sanity-checking marketing claims

You see a marketing slide saying "GPU X achieves 5 PFLOPS for AI
workloads." How do you sanity-check?

```
5 PFLOPS = 5000 TFLOPS = 5 * 10^15 FLOPS
```

That's roughly 5x H100's FP8 dense peak (~2000 TFLOPS) or 8x its
FP16 dense peak. Possibilities:

- With sparsity (multiply dense by 2): 5 PFLOPS = 2.5 PFLOPS dense
  FP8 = 25% above H100 dense FP8 = plausible if it's a Blackwell
  successor or an aggregated multi-die number.
- Multi-chip: B200 is two-die. Quoting both dies as one number is
  common.
- INT8 / FP4: those have 2x and 4x the dense FP8 peak; "5 PFLOPS
  INT8" could be ~625 TFLOPS dense FP16 equivalent — much less
  impressive.

The math lets you decompose the claim. If a vendor doesn't
disclose precision or sparsity assumptions, treat the number as
suspicious until they do.

### Estimating training time from peak throughput

A common forward-pass calculation: GPT-3 175B has ~3.5 * 10^11
FLOPS per token (forward only; ~3x that for backward + optimizer
= ~1.05e12 FLOPS per token training step).

To train on 300B tokens at H100 FP16 dense throughput:

```
total_FLOPS = 300e9 * 1.05e12 = 3.15e23 FLOPS

At H100 FP16 dense peak (989 TFLOPS):
  ideal_time = 3.15e23 / 989e12 = 3.19e8 seconds = 9.7 years (on one GPU)

At 50% sustained efficiency (realistic for training):
  effective_throughput = 989 / 2 = 494 TFLOPS
  time = 6.4e8 seconds = 20 years (one GPU)

So: ~3500 H100s for 21 days of training time (close to GPT-3's
actual training duration on A100s with appropriate scaling).
```

This is the kind of back-of-envelope cluster sizing that the
formulas in this exercise enable. The numbers are coarse but
they're correct *as orders of magnitude*, and that's what
infrastructure decisions need.

### Verifying the AI ridge for each GPU

Now that you have peak compute and peak bandwidth, the ridge point
falls out:

```
A100 FP32 ridge = 19500 GFLOPS / 2039 GB/s ≈ 9.56 FLOP/byte
H100 FP32 ridge = 67000 GFLOPS / 3350 GB/s ≈ 20.0 FLOP/byte
RTX 4090 ridge  = 82600 GFLOPS / 1008 GB/s ≈ 82.0 FLOP/byte
```

A100's ridge is the lowest — meaning A100 is most "balanced" for
mid-AI workloads. RTX 4090 is the most compute-rich relative to
bandwidth; many workloads compute-bound on A100 become
memory-bound on 4090. H100 sits between them.

You'll see this ridge math come up repeatedly in Exercises 4 and
6. The peak numbers from this exercise are the inputs.

## Related material

- Lecture 1.1 — Worked example: counting FP32 lanes on an H100.
- Lecture 1.6 — Full reference table for A100/H100/L40S/RTX 4090.
- `resources.md` — Datasheet links.
- NVIDIA's [compute-capabilities table](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) — official limits per CC.

## Next exercise

Exercise 3 — Reimplement NVIDIA's occupancy calculation from
scratch. The arithmetic is similar in spirit (peak resource ÷
per-block resource → number of blocks). But the conclusion drives
*how to write a kernel*, not just how to evaluate one.
