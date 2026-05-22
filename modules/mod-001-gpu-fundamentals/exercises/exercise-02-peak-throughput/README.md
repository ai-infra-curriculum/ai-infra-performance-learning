# Exercise 2 — Peak throughput from a spec sheet

> **Targets learning objectives:** 2, 3
> **Time:** ~30 min
> **Requires:** Python 3.10+. No GPU needed.

## What you'll do

You will compute two numbers — peak FP32 throughput and peak memory
bandwidth — for three GPUs (A100 SXM4 80GB, H100 SXM5 80GB, RTX 4090)
from their published spec-sheet values.

This is one of those skills that looks trivial until the first time you
have to defend a procurement decision in a meeting and someone asks
"what is the FP32 throughput of an L40 at 90% utilization?" without a
spec sheet handy. After this exercise you do not need to look it up;
you can derive it.

> **Why not tensor cores too?** Tensor-core throughput is
> architecture-specific (third-gen on Ampere, fourth-gen on Hopper) and
> the published TFLOPS often quote *with sparsity*, which is a 2× cooked
> number. Deriving tensor TFLOPS from clock + structure is a Module 4
> topic, after we have unpacked what tensor cores actually do. For now,
> stay with FP32 and bandwidth — both are clean derivations.

## The formulas

**Peak FP32 (dense, no tensor cores):**

```
peak_fp32_tflops = (sm_count * cuda_cores_per_sm * boost_clock_ghz * 2) / 1000
```

The factor of 2 comes from the fused multiply-add (FMA) — each FP32 CUDA
core can issue one FMA per cycle, counting as 2 FLOPS.

**Peak memory bandwidth (GB/s):**

```
peak_bandwidth_gbs = (effective_per_pin_gbps * bus_width_bits) / 8
```

The "effective per-pin Gbps" is the data rate per memory pin. NVIDIA's
spec sheets quote it directly for GDDR (e.g. "21 Gbps" for the RTX 4090's
GDDR6X). For HBM2e and HBM3 they quote either the per-pin Gbps or the
total bandwidth — both are equivalent given the bus width.

| Memory | Approx effective per-pin Gbps | Bus width |
|---|---|---|
| GDDR6X (RTX 4090) | 21.0 | 384-bit |
| HBM2e (A100 SXM4 80GB) | ~3.19 | 5120-bit |
| HBM3 (H100 SXM5 80GB) | ~5.23 | 5120-bit |

You will plug these into the formula and confirm you reproduce NVIDIA's
quoted total bandwidth.

## What to submit

Edit `starter.py`. For each of the three GPUs, fill in the spec values
and implement the two formula functions. Then run:

```bash
python check.py
```

The check verifies your computed peak FP32 and bandwidth values match
NVIDIA's published values within 5% tolerance (allowing for clock-rate
variation between SKUs).

## Hints

- A100 SXM4 80GB: 108 SMs, 64 FP32 CUDA cores per SM (Ampere GA100),
  boost ≈ 1.41 GHz. HBM2e at ~3.19 Gbps per pin × 5120-bit bus →
  ~2039 GB/s.
- H100 SXM5 80GB: 132 SMs, 128 FP32 CUDA cores per SM (Hopper GH100),
  boost ≈ 1.98 GHz. HBM3 at ~5.23 Gbps per pin × 5120-bit bus →
  ~3350 GB/s.
- RTX 4090: 128 SMs, 128 FP32 CUDA cores per SM (Ada AD102),
  boost ≈ 2.52 GHz. GDDR6X at 21 Gbps per pin × 384-bit bus →
  ~1008 GB/s.

## What "right" looks like

Your numbers should land within 5% of NVIDIA's published spec-sheet
peak values. Exact match is not the goal (clocks vary, SKUs vary);
*understanding the derivation* is.
