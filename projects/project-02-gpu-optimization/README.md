# Project 2: Custom CUDA Kernels for Transformer Optimization

> **Tier**: 4 (Capstone-grade)
> **Track**: AI/ML Performance Engineering
> **Estimated effort**: 60 hours
> **Complexity**: Advanced
> **Primary modules**: mod-002 (CUDA Fundamentals), mod-004 (Transformer Optimization), mod-006 (Kernel Engineering)
> **Secondary modules**: mod-003 (Profiling), mod-008 (Advanced Topics)

## 1. Overview

Write **production-grade custom CUDA kernels** that beat the standard
PyTorch / cuBLAS / cuDNN implementations by at least **3x** on the
hottest paths in a decoder-only transformer. The deliverable is a
loadable Python extension (`torch.utils.cpp_extension` or `setuptools`)
that exposes four kernels:

1. **Flash Attention v2** — fused, online-softmax, tile-streaming
   attention for both prefill and decode. Targets >= 3x speedup over
   `torch.nn.functional.scaled_dot_product_attention` (math backend)
   and competitive with the upstream `flash-attn` package within 15%.
2. **Fused Rotary Position Embedding (RoPE)** — applies rotation to
   Q and K in-place inside the QKV projection epilogue. Targets full
   elimination of the standalone RoPE kernel from the timeline.
3. **Welford-based LayerNorm** — numerically stable, single-pass mean
   and variance using Welford's online algorithm, vectorized loads,
   warp-shuffle reductions. Targets >= 80% of HBM peak bandwidth.
4. **Vectorized GELU (tanh approximation)** — `float4` vectorized
   memory accesses, fused into the FFN epilogue. Targets >= 90% of
   HBM peak bandwidth (it is purely memory bound).

You will write each kernel in raw CUDA C++ AND a Triton equivalent,
benchmark them against PyTorch on A100 and H100, and prove via Nsight
Compute that the speedup is **mechanism-explained** (occupancy,
arithmetic intensity, memory pipeline utilization) and not just a lucky
number.

This is the project that proves the candidate understands what the
hardware is actually doing — not just what `torch.compile` happens to
produce.

## 2. Performance targets (the gates you must hit)

Measured on **NVIDIA H100 SXM5 (80 GB)** unless otherwise noted; A100
80GB SXM acceptable with reported derate. Sequence length = 4096,
head dim = 128, num heads = 32, dtype = BF16. Batch sizes 1 (decode)
and 16 (prefill).

| Kernel | Baseline (PyTorch) | Target | Gate |
|--------|--------------------|--------|------|
| Attention prefill (bs=16, sl=4096) | `F.scaled_dot_product_attention` math | **>= 3.0x speedup** | Hard |
| Attention decode (bs=1, sl=4096, +1 token) | `F.scaled_dot_product_attention` math | **>= 3.0x speedup** | Hard |
| Attention vs upstream `flash-attn` v2 | `flash_attn.flash_attn_func` | **within 15% (slower OK)** | Hard |
| Fused QKV + RoPE | matmul + RoPE separate kernels | **>= 3.5x speedup** | Hard |
| LayerNorm (Welford, BF16) | `torch.nn.LayerNorm` | **>= 3.0x speedup** | Hard |
| GELU (vectorized BF16) | `F.gelu` | **>= 2.0x speedup** | Hard |
| LayerNorm mem-bandwidth utilization | n/a | **>= 80% of HBM peak** | Hard |
| Flash Attention compute utilization | n/a | **>= 70% of Tensor Core peak** | Hard |
| End-to-end 7B Llama-style block forward | unfused PyTorch baseline | **>= 2.5x speedup** | Hard |
| Bit-equivalent numerical accuracy | n/a | max abs err <= 1e-2 BF16; rel err <= 1e-3 | Hard |
| CUDA Graphs replay overhead (decode) | n/a | <= 5 us per replay | Soft |

H100-specific bonus targets (graded in rubric, not hard gates):

| Kernel | H100 target | Notes |
|--------|--------------|-------|
| Attention with WGMMA + TMA | >= 80% Tensor Core peak | Hopper-only |
| Attention with FP8 (TransformerEngine compat) | >= 1.5x BF16 throughput | Hopper-only |

## 3. Learning outcomes

By the end of this project a learner will be able to:

1. Reason about GPU memory hierarchy (HBM, L2, SMEM, registers) and
   the bandwidth ratios between them, and design tiling around them.
2. Write a tiled, online-softmax Flash Attention kernel from scratch
   without leaning on a reference repo, and explain every line.
3. Use `cp.async` (Ampere) and TMA (Hopper) for asynchronous global -
   shared memory transfers; pipeline producer and consumer stages.
4. Use warp specialization, named barriers, and `mbarrier` for
   producer/consumer pipelines on Hopper.
5. Use Tensor Core MMA intrinsics: `mma.sync.aligned.m16n8k16` (Ampere)
   and `wgmma.mma_async` (Hopper); manage register fragments.
6. Write the same kernel in Triton; compare generated SASS; explain
   when Triton matches hand-tuned CUDA and when it does not.
7. Profile with Nsight Compute: Memory Workload Analysis, Source View
   (SASS, PTX, CUDA-C correspondence), Warp State Statistics, Roofline.
8. Defend a kernel choice with measured numbers: arithmetic intensity,
   theoretical occupancy, achieved occupancy, SOL% for SM and HBM.

## 4. Prerequisites

### 4.1 Hardware

- **Strongly preferred**: 1x NVIDIA H100 SXM5 (80 GB). Hopper-specific
  paths (WGMMA, TMA, FP8) cannot be exercised otherwise.
- **Acceptable**: 1x NVIDIA A100 80GB SXM. WGMMA / TMA paths become
  rubric-only ablations.
- **Acceptable for development**: 1x RTX 4090 (Ada). Numbers reported
  separately.
- 128 GB system RAM, 1 TB NVMe, Linux x86_64.

### 4.2 Software (pinned)

- CUDA Toolkit **12.4**+ (12.5 strongly preferred for WGMMA on H100)
- cuDNN **9.x**
- PyTorch **2.4**+ built against CUDA 12.4
- **CUTLASS 3.5+** (for reference kernels and Hopper recipes)
- **Triton 3.x** (matches PyTorch 2.4)
- **FlashAttention-3** (for reference comparison, H100 only)
- **NVIDIA TransformerEngine 1.7+** (for FP8 reference)
- Nsight Systems **2024.2**+, Nsight Compute **2024.2**+
- `nvcc` host compiler: gcc-12 or clang-15
- `ncu-ui` and `nsys-ui` for desktop inspection
- `cuda-gdb` for kernel debugging

### 4.3 Knowledge

- Completion of mod-002 (CUDA Fundamentals) and mod-004 (Transformer
  Optimization).
- Comfort with C++17, template metaprogramming, and pointer arithmetic.
- Familiarity with the PTX ISA reference; ability to read SASS at the
  level of "this is a load, this is an MMA, this is a sync".
- Comfortable reading the FlashAttention paper (Dao et al. 2022) and
  FlashAttention-2 (Dao 2023).

## 5. Deliverables

Per `deliverables/README.md`, the final submission must include:

- `csrc/` — raw CUDA C++ kernels (`.cu` and `.h`).
  - `attention_fwd.cu`, `attention_bwd.cu` — Flash Attention v2.
  - `attention_decode.cu` — single-token decode specialization.
  - `qkv_rope_fused.cu` — fused QKV projection + RoPE epilogue.
  - `layernorm_welford.cu` — Welford-online LayerNorm.
  - `gelu_vec.cu` — vectorized GELU.
  - `binding.cpp` — pybind11 / TORCH_EXTENSION binding.
- `triton/` — Triton equivalents for cross-comparison.
- `python/fastkernels/` — pure-Python wrapper package (autograd hooks,
  shape checks, dtype handling).
- `bench/` — benchmark harness using CUDA events, with PyTorch /
  `flash-attn` / Triton baselines.
- `profiles/` — Nsight Compute and Nsight Systems reports for each
  kernel.
- `reports/` — roofline plots, SOL tables, SASS dumps, write-ups.
- `tests/` — pytest suite (correctness + autograd `gradcheck` +
  numerical tolerance).
- `setup.py` / `pyproject.toml` — installable extension.
- `Dockerfile` — reproducible build environment.
- `Makefile` — `make all`, `make bench`, `make profile`, `make verify`.

## 6. Week-by-week breakdown

Six weeks at ~10 hours/week, or four weeks at 15. Sequential phases —
do not start a phase before the previous one's gate passes.

### Week 1 — Memory-bound warmup: LayerNorm + GELU (8-10h)

These are the easiest kernels and the right place to build the harness.
**Gate**: GELU achieves >= 90% HBM peak. LayerNorm achieves >= 80% HBM
peak and matches PyTorch within tolerance.

- Set up the build system (`setup.py` with `CUDAExtension`).
- Write `BenchHarness` using CUDA events, 100 warmup + 1000 measured
  iters, std-dev gate.
- Implement vectorized GELU with `float4` / `bfloat16x8` loads.
- Implement LayerNorm using Welford's online algorithm + warp shuffle
  reduction (`__shfl_xor_sync`).
- Nsight Compute: confirm "Memory Throughput" and "DRAM Throughput"
  rows show > 80%.

### Week 2 — Compute-bound warmup: QKV projection + RoPE (10-12h)

**Gate**: Fused QKV + RoPE eliminates the standalone RoPE kernel from
the Nsight Systems timeline and hits >= 3.5x speedup.

- Implement Q, K, V projection using `mma.sync.aligned` MMA on Ampere
  (or `wgmma.mma_async` on Hopper).
- Apply RoPE in the epilogue of the same kernel — compute sin/cos
  on the fly to avoid a second memory pass.
- Verify against `torch.nn.functional.linear` + a Python RoPE.
- Nsight Compute: assert "Tensor Active %" >= 50% and no global memory
  spill.

### Week 3 — Flash Attention v2 forward (12-15h)

**Gate**: Prefill (bs=16, sl=4096) hits >= 3.0x vs PyTorch math and
within 15% of upstream `flash-attn` v2.

- Tile Q along sequence dimension (block size Br = 64 or 128 typical).
- Tile K/V along sequence dimension (Bc = 64).
- Online softmax: keep running `m_i` (max) and `l_i` (sum-of-exp),
  rescale `O` on each tile.
- Use `cp.async` to overlap K/V load with compute on the previous tile.
- Causal masking: skip K/V tiles beyond the diagonal entirely.
- Numerical check: max abs error vs FP32 reference <= 1e-2.

### Week 4 — Flash Attention v2 backward + decode kernel (10-12h)

**Gate**: Backward passes `gradcheck` within tolerance. Decode kernel
(bs=1, sl=4096+1) hits >= 3.0x vs SDPA math backend.

- Forward saves `m_i`, `l_i`, `O`; backward recomputes attention
  weights tile-by-tile (the FA2 trick — no `O(N^2)` storage).
- Decode kernel specialization: single Q row, full K/V history,
  no Q tiling needed. KV cache assumed contiguous in HBM.
- CUDA Graph capture of the decode path — single replay should be
  <= 5 us overhead.

### Week 5 — Triton equivalents + Hopper-specific paths (10-12h)

**Gate**: Triton matches or beats hand-tuned CUDA within 10% on at
least 2 of 4 kernels. On H100: WGMMA + TMA path matches hand-tuned
within 5%.

- Port each kernel to Triton. Use `tl.dot`, `tl.load` with hints,
  `tl.constexpr` for tile sizes.
- Compare generated SASS (`cuobjdump --dump-sass`) for one tile of the
  attention kernel; explain divergences.
- On H100: write a WGMMA + TMA variant of attention forward. Use
  `cute::TMA::COPY` semantics or raw `cp.async.bulk.tensor` PTX.
- Compare against FlashAttention-3 reference if available.

### Week 6 — Integration, profile, report (8-10h)

**Gate**: End-to-end 7B Llama-style block forward shows >= 2.5x
speedup over unfused PyTorch.

- Wire all four kernels into a `LlamaBlock` module that replaces the
  HuggingFace block in-place.
- Run a 7B-parameter forward at sl=4096, bs=1 and bs=16.
- Nsight Systems: confirm kernel timeline has 4 kernels per layer
  (QKV+RoPE, FA2 fwd, LN, GELU+down-proj), not the original ~12.
- Nsight Compute: roofline for each kernel, with at least one annotated
  to show "this is where it sits and why".
- Write `reports/kernel_summary.md` and `reports/roofline_*.png`.

## 7. Architecture pointer

See [`architecture.md`](./architecture.md) for kernel-level designs,
data flow diagrams, tiling decisions, and trade-off discussion
(Ampere vs Hopper, hand-tuned CUDA vs Triton vs CUTLASS, FA2 vs FA3).

## 8. Step-by-step build guide

See [`STEP_BY_STEP.md`](./STEP_BY_STEP.md). Contains the canonical
"do this exactly" path with kernel snippets, profiling commands,
and the gotcha list (bank conflicts, register spilling, async barrier
ordering, etc.).

## 9. Rubric summary

Full rubric in [`rubric.md`](./rubric.md). High-level dimensions:

1. **Speedup** (vs PyTorch baseline; vs upstream flash-attn)
2. **Numerical correctness** (autograd gradcheck, tolerance bounds)
3. **Memory bandwidth utilization** (Nsight Compute SOL%)
4. **Compute utilization** (Tensor Core SOL%)
5. **Profiling depth** (roofline, SASS analysis, defensible reasoning)
6. **Triton parity** (port quality, SASS comparison)
7. **Engineering judgment** (when to hand-tune, when to use Triton/CUTLASS)
8. **Hopper-specific work** (WGMMA, TMA, FP8 — distinction-only on A100)

Pass = all hard performance gates met AND rubric score >= 3/5 in every
dimension. Distinction = rubric score >= 4/5 in at least 6 of 8.

## 10. Success criteria checklist

- [ ] LayerNorm: >= 3.0x vs `torch.nn.LayerNorm`, >= 80% HBM SOL.
- [ ] GELU: >= 2.0x vs `F.gelu`, >= 90% HBM SOL.
- [ ] Fused QKV + RoPE: >= 3.5x vs unfused; standalone RoPE kernel
      gone from timeline.
- [ ] Flash Attention prefill (bs=16, sl=4096): >= 3.0x vs SDPA math,
      within 15% of upstream `flash-attn`.
- [ ] Flash Attention decode (bs=1, sl=4096+1): >= 3.0x vs SDPA math.
- [ ] Backward `gradcheck` passes within tolerance.
- [ ] CUDA Graph replay <= 5 us on decode.
- [ ] End-to-end 7B block forward: >= 2.5x speedup.
- [ ] One Nsight Compute report per kernel with roofline annotation.
- [ ] One Triton port per kernel; SASS comparison for attention tile.
- [ ] `make all` reproducible from clean checkout in Docker.

## 11. Related modules

| Module | Why it matters |
|--------|----------------|
| mod-002 CUDA Fundamentals | Memory hierarchy, MMA, async copy primitives. |
| mod-003 Performance Profiling | Nsight Compute Source View, SOL%, roofline. |
| mod-004 Transformer Optimization | Why attention is the bottleneck; what fusions matter. |
| mod-006 Kernel Engineering | Tiling, register pressure, occupancy, bank conflicts. |
| mod-008 Advanced Topics | FP8, WGMMA, FlashAttention-3, FlashDecoding. |

## 12. Stretch goals (for distinction)

- **FP8 attention** with TransformerEngine: prove FP8 attention beats
  BF16 by >= 1.5x on H100, accuracy within tolerance after recipe.
- **FlashAttention-3** style warp specialization with producer (TMA)
  and consumer (WGMMA) split.
- **FlashDecoding++** for very long contexts (sl >= 32k) with
  split-KV across SMs.
- **Paged KV cache** kernel: attention reads K/V from a non-contiguous
  block table (preview of Project 3's PagedAttention).
- **Persistent CTAs** for the decode kernel: one CTA per SM, looping
  over decode steps to amortize launch overhead.
- **CUTLASS 3.x epilogue** for the QKV projection — compare against
  hand-tuned CUDA, explain when each wins.

## 13. Out of scope

- Multi-GPU collectives (no NCCL).
- Sparse attention patterns (BigBird, Longformer).
- Quantized weights inside the attention kernel (this is Project 1's
  quantization land; we keep weights BF16 here).
- Serving / batching (that's Project 3).

## 14. References

- Dao et al., **"FlashAttention: Fast and Memory-Efficient Exact
  Attention with IO-Awareness"**, NeurIPS 2022.
- Dao, **"FlashAttention-2: Faster Attention with Better Parallelism
  and Work Partitioning"**, 2023.
- Shah et al., **"FlashAttention-3: Fast and Accurate Attention with
  Asynchrony and Low-precision"**, 2024.
- NVIDIA, **CUDA C++ Programming Guide v12.5**, chapters on async
  copy, TMA, WGMMA.
- NVIDIA, **PTX ISA 8.5**, `cp.async`, `mbarrier`, `wgmma`.
- Tillet et al., **"Triton: An Intermediate Language and Compiler for
  Tiled Neural Network Computations"**, MAPL 2019.
- NVIDIA Nsight Compute User Guide, Memory Workload Analysis section.
- Williams, Waterman, Patterson, **"Roofline: An Insightful Visual
  Performance Model for Multicore Architectures"**, CACM 2009.
- CUTLASS 3.x documentation, CuTe layout algebra.
