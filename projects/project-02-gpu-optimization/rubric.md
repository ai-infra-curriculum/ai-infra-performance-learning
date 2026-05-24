# Rubric — Custom CUDA Kernels for Transformer Optimization

Reviewers score each dimension 1-5. **Pass** requires all hard
performance gates from `requirements.md` Section 3 AND >= 3/5 in
every dimension. **Distinction** requires >= 4/5 in at least 6 of 8.

Hard gates fail the project regardless of rubric score:

- PR-1..PR-12 missed.
- `make all` fails on a clean checkout inside the provided Docker
  image.
- Any kernel returns NaN on the documented input shapes.
- Register spilling on any of the four hot kernels.

## 1. Scoring scale

| Level | Meaning |
|-------|---------|
| 1 | Missing or fundamentally broken. |
| 2 | Partial; works only on one shape; no SOL analysis. |
| 3 | Meets requirements; defensible. (Pass bar.) |
| 4 | Exceeds requirements with measured ablations. |
| 5 | Reference-quality; could go into CUTLASS / FlashAttention upstream. |

## 2. Dimensions

### D1. Speedup (weight: highest)

Speedup vs PyTorch baselines AND vs upstream `flash-attn`.

| Level | Evidence |
|-------|----------|
| 1 | < 1.5x on any kernel; or fails to beat baseline. |
| 2 | 1.5-2.5x on most kernels. |
| 3 | All hard PR-* speedup gates met. |
| 4 | Attention within 10% of upstream `flash-attn` v2; E2E >= 3.0x. |
| 5 | Attention matches or beats upstream `flash-attn` v2; FA3-style path on H100 reported with numbers. |

### D2. Numerical correctness

Bit-tolerance vs FP32 reference; `gradcheck` for backward.

| Level | Evidence |
|-------|----------|
| 1 | Kernels return NaN or large drift. |
| 2 | Passes on one shape; no fuzz; no `gradcheck`. |
| 3 | Max abs err <= 1e-2 BF16; `gradcheck` passes on small case. |
| 4 | Numerical fuzz tests covering random shapes / strides; documented tolerance budget per kernel. |
| 5 | Deterministic mode available; reduction order documented; numerical regression CI test. |

### D3. Memory bandwidth utilization

Nsight Compute DRAM throughput SOL% on memory-bound kernels.

| Level | Evidence |
|-------|----------|
| 1 | No SOL data. |
| 2 | < 50% on LayerNorm and GELU. |
| 3 | LayerNorm >= 80%, GELU >= 90% (PR-2, PR-4). |
| 4 | All memory-bound kernels >= 85%; analysis explains where the remaining bandwidth goes. |
| 5 | L2 hit rate analysis included; cache-bypass `ld.global.nc` ablation; bandwidth-saturation chart per tile size. |

### D4. Compute utilization

Tensor Core SOL% on compute-bound kernels.

| Level | Evidence |
|-------|----------|
| 1 | No data. |
| 2 | < 50% Tensor Core SOL on FA2 fwd. |
| 3 | FA2 fwd Tensor Core SOL >= 70% (PR-9). |
| 4 | FA2 fwd >= 80%; QKV+RoPE Tensor Core active > 60%. |
| 5 | Hopper WGMMA path >= 80% Tensor Core SOL; explained why FP32 epilogue is the remaining bottleneck. |

### D5. Profiling depth

Roofline, Nsight Compute analysis, SASS reading.

| Level | Evidence |
|-------|----------|
| 1 | No Nsight artifacts. |
| 2 | Nsight Systems trace only; no kernel-level analysis. |
| 3 | One Nsight Compute report per kernel; one roofline plot per kernel; hottest kernel classified. |
| 4 | Roofline plots show before / after for each kernel; bandwidth and compute saturation tracked across tile sizes. |
| 5 | SASS dumps annotated; warp state histograms presented; "this is why the kernel sits at point X on the roofline" written for each kernel. |

### D6. Triton parity

Quality of Triton port and cross-comparison.

| Level | Evidence |
|-------|----------|
| 1 | No Triton port. |
| 2 | Triton port exists but slower by > 50%. |
| 3 | Triton matches or beats hand-tuned on at least 2 of 4 kernels (within 10%). |
| 4 | SASS comparison written; differences quantified. |
| 5 | Hand-tuned wins explained by specific SASS instruction count or memory pipeline issue; documented "use Triton when X, use hand-tuned when Y". |

### D7. Engineering judgment

Right tool for the job; clean trade-offs; honest negative results.

| Level | Evidence |
|-------|----------|
| 1 | Cargo-culted tile sizes; no exploration. |
| 2 | One tile config per kernel; no sweep. |
| 3 | Tile config table justified per arch (Ampere vs Hopper). Negative results reported (e.g., "Br=256 spills 8 regs, kept Br=128"). |
| 4 | Ablations: causal vs non-causal, head_dim 64 vs 128, FP16 vs BF16; trade-offs explained. |
| 5 | Comparison against CUTLASS reference for at least one kernel; "when to use CUTLASS vs hand-tuned" guidance. |

### D8. Hopper-specific work

WGMMA, TMA, FP8. Distinction-only path on A100-only submissions.

| Level | Evidence (H100 submissions) |
|-------|------------------------------|
| 1 | No Hopper path. |
| 2 | Hopper path exists but not measurably faster than Ampere path. |
| 3 | WGMMA + TMA path implemented; meets PR-14 (>= 80% Tensor Core SOL). |
| 4 | FP8 attention path with measured PR-15 (>= 1.5x BF16). |
| 5 | FA3-style warp-specialized producer/consumer pipeline; matches or beats upstream FA3; FP8 recipe with accuracy table. |

For A100-only submissions, D8 caps at 3 with H100 ablation
written-but-not-run; this is acceptable for Pass but precludes
Distinction.

## 3. Bonus rubric (distinction only)

- **B1**: FlashDecoding++ split-KV kernel for sl >= 32k with measured
  numbers.
- **B2**: Paged KV cache attention (block table indirection).
- **B3**: Persistent CTAs for decode (one CTA per SM, looping over
  steps).
- **B4**: CUTLASS 3.x epilogue port of QKV+RoPE with comparison to
  hand-tuned.
- **B5**: Deterministic reduction mode toggleable via env var.

## 4. Anti-patterns (auto-deduction)

- **`time.perf_counter` instead of CUDA events in bench**: -2 on D1.
- **No warmup or < 100 iters of warmup**: -1 on D1.
- **Hand-tuned kernel spilled registers (ptxas reports > 0)**: -1 on
  D7 and -1 on D4 (it cost you compute SOL%).
- **`std/p50 > 5%` in published numbers**: -1 on D1.
- **`flash-attn` baseline built against different CUDA than the
  candidate's kernels**: -1 on D1 (unfair comparison).
- **No autograd binding**: -2 on D2.
- **Triton port is a one-liner wrapping `torch.nn.functional`**: -2
  on D6.
- **No SOL%, only "speedup"**: -2 on D3 and D4 (no mechanism story).

## 5. Reviewer checklist

```
[ ] Hard gates PR-1..PR-12 met?     (else fail)
[ ] make all reproduces in Docker?  (else fail)
[ ] D1 Speedup:               _/5
[ ] D2 Numerical correctness: _/5
[ ] D3 Memory bandwidth:      _/5
[ ] D4 Compute utilization:   _/5
[ ] D5 Profiling depth:       _/5
[ ] D6 Triton parity:         _/5
[ ] D7 Engineering judgment:  _/5
[ ] D8 Hopper-specific:       _/5

Total: __/40
Pass:        all >= 3 and hard gates met
Distinction: >= 6 dimensions at 4+
```

## 6. How to defend your score

For each dimension, the reviewer expects ONE artifact + ONE sentence:

| Dimension | Artifact | Sentence |
|-----------|----------|----------|
| D1 | `reports/bench_summary.md` | "Attention won W from FA2 algorithm, V from `cp.async` overlap, U from BF16x8 vectorization." |
| D2 | `tests/gradcheck/*.py` output + `reports/numerical_fuzz.md` | "Max abs err 4.3e-3 BF16 over 1000 random shapes." |
| D3 | `reports/sols_memory.md` + Nsight Compute screenshot | "LayerNorm sits at 84% DRAM SOL; the missing 16% is L2 misses on gamma/beta." |
| D4 | `reports/sols_compute.md` | "FA2 fwd at 73% Tensor Core SOL; epilogue softmax-rescale FP32 ops are the gap." |
| D5 | `reports/roofline_<kernel>.png` + `reports/sass_<kernel>.txt` | "Kernel moved from memory-bound at AI=4 to compute-bound at AI=64 after tiling." |
| D6 | `reports/sass_diff_<kernel>.txt` | "Triton emits one extra `BAR.SYNC` per tile; cost is 2.3%." |
| D7 | `reports/tile_sweep_<kernel>.csv` | "Br=128 wins on H100; Br=64 spills 4 regs on A100." |
| D8 | `reports/hopper_path.md` | "WGMMA + TMA path at 81% Tensor Core SOL; FP8 path at 1.7x BF16." |
