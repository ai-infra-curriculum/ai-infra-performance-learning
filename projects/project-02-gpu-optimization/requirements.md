# Requirements — Custom CUDA Kernels for Transformer Optimization

Every "MUST" is a gate. Every "SHOULD" is graded in the rubric.

## 1. Scope

### 1.1 In scope

- Four custom kernels with hand-tuned CUDA C++ + Triton equivalents:
  - Flash Attention v2 forward and backward.
  - Single-token attention decode kernel.
  - Fused QKV projection + Rotary Position Embedding.
  - Welford-online LayerNorm.
  - Vectorized GELU (tanh approximation).
- Targets: BF16 on Ampere, BF16 + FP8 path on Hopper.
- Sequence lengths 512 / 2048 / 4096 / 8192.
- Head dim 64 / 128.
- Causal and non-causal attention.
- Loadable PyTorch extension with autograd support.

### 1.2 Out of scope

- Multi-GPU communication.
- Sparse attention (block-sparse, sliding window).
- Quantized weight kernels (INT8/INT4 MM).
- LLM serving / batching (Project 3).
- Backward pass for QKV-RoPE fused kernel (forward-only is acceptable
  if attention bwd is correct; rubric note only).

## 2. Functional requirements

### FR-1: Build system

- MUST be installable via `pip install .` or `python setup.py develop`.
- MUST support both CUDA 12.4 and CUDA 12.5 host toolchains.
- MUST compile cleanly with NVCC `-Werror cross-execution-space-call`,
  `-Wno-deprecated-gpu-targets`.
- MUST support arch sm_80 (A100), sm_89 (Ada), sm_90a (Hopper) via
  `TORCH_CUDA_ARCH_LIST`.
- MUST be importable as `import fastkernels` after install.

### FR-2: Kernel — Vectorized GELU

- MUST be `__global__ void gelu_vec(...)` exposed via pybind11.
- MUST use vectorized loads (`float4` or `bfloat16x8`).
- MUST handle non-multiple-of-vector tail correctly (boundary epilogue).
- MUST match `F.gelu(approximate="tanh")` within max abs err 1e-3 BF16.
- MUST be measured against `F.gelu` in the bench harness.

### FR-3: Kernel — Welford LayerNorm

- MUST use Welford's online algorithm for numerical stability.
- MUST use warp-shuffle reduction (`__shfl_xor_sync`) within a warp
  and shared-memory reduction across warps.
- MUST support arbitrary normalized shape divisible by 32.
- MUST match `torch.nn.LayerNorm` within max abs err 1e-2 BF16.
- MUST support backward (autograd binding).

### FR-4: Kernel — Fused QKV + RoPE

- MUST compute Q, K, V projection in a single launch.
- MUST apply RoPE rotation to Q and K in the kernel epilogue
  (sin/cos computed on the fly OR from a passed-in cache).
- MUST match `nn.Linear` + Python RoPE within max abs err 1e-2 BF16.
- MUST appear in the Nsight Systems timeline as ONE kernel; the
  separate `mul_add`/`stack` ops must not appear in the hot path.

### FR-5: Kernel — Flash Attention v2 forward

- MUST tile Q along sequence dim (configurable Br = 64 or 128).
- MUST tile K/V along sequence dim (configurable Bc = 64 or 128).
- MUST use online softmax with running max + sum-of-exp; output is
  rescaled per tile.
- MUST support causal masking with tile-level skip (no compute past
  diagonal).
- MUST use `cp.async` (sm_80+) for K/V loads overlapping compute.
- MUST support head dim 64 AND head dim 128.
- MUST match a reference FP32 attention within max abs err 1e-2 BF16,
  relative err 1e-3.

### FR-6: Kernel — Flash Attention v2 backward

- MUST recompute attention weights tile-by-tile (no `O(N^2)` storage).
- MUST consume `m_i`, `l_i`, `O` saved during forward.
- MUST pass `torch.autograd.gradcheck` on a small case (sl=64, dim=64)
  with double-precision reference.
- MUST match the upstream `flash_attn` backward within 1e-2 abs.

### FR-7: Kernel — Decode specialization

- MUST handle bs=1, single Q row, full K/V cache (length up to 8192).
- MUST NOT tile Q (one Q row fits in registers).
- MUST be capturable into a CUDA Graph.
- Replay overhead MUST be <= 5 us per step (CUDA Graphs warm).

### FR-8: Triton parity

- MUST provide a Triton implementation for each of the four kernels.
- MUST achieve within 10% of hand-tuned CUDA on at least 2 of 4.
- MUST emit one written comparison per kernel: "Triton matches /
  diverges by X% because Y".

### FR-9: Bench harness

- MUST use `torch.cuda.Event` for timing, not Python time.
- MUST do >= 100 warmup iters and >= 1000 measured iters.
- MUST report P50, P90, P99, mean, std, TFLOPs, GB/s.
- MUST gate on `std/p50 < 5%` and abort with a clear error otherwise.
- MUST capture results to `reports/bench_<kernel>_<dtype>_<shape>.json`.
- MUST compare against PyTorch baseline + upstream `flash-attn` (when
  installed).

### FR-10: Profiling integration

- MUST emit one Nsight Systems trace per kernel.
- MUST emit one Nsight Compute report per kernel using `--launch-skip`
  and `--launch-count 1` (avoid 100GB reports).
- MUST extract: SOL% (memory + compute), achieved occupancy,
  arithmetic intensity, warp state histogram.
- MUST produce a roofline plot per kernel using measured FLOPs and
  bytes.

### FR-11: Correctness tests

- MUST include unit tests per kernel: shape coverage, dtype coverage,
  causal vs non-causal, head-dim variants.
- MUST include `gradcheck` for kernels with backward.
- MUST run a numerical fuzz: random shapes / strides / seeds, compare
  vs PyTorch reference.

### FR-12: End-to-end integration

- MUST provide a `FastLlamaBlock` that uses all four kernels.
- MUST be a drop-in replacement for HuggingFace LlamaDecoderLayer
  (forward only).
- MUST be benchmarked at 7B parameter scale, sl=4096, bs=1 and bs=16.

## 3. Performance requirements (hard gates)

Measured on **H100 SXM5 80GB** (A100 80GB SXM acceptable; report
separately). All numbers BF16 unless noted.

| ID | Metric | Target | Notes |
|----|--------|--------|-------|
| PR-1 | LayerNorm speedup vs `torch.nn.LayerNorm` | >= 3.0x | sl=4096, dim=4096 |
| PR-2 | LayerNorm DRAM throughput SOL% | >= 80% | Nsight Compute Memory Workload |
| PR-3 | GELU speedup vs `F.gelu` | >= 2.0x | sl=4096, dim=14336 (FFN inter) |
| PR-4 | GELU DRAM throughput SOL% | >= 90% | Nsight Compute Memory Workload |
| PR-5 | Fused QKV+RoPE speedup vs unfused | >= 3.5x | sl=4096, dim=4096 |
| PR-6 | FA2 prefill (bs=16, sl=4096) vs SDPA math | >= 3.0x | head dim 128 |
| PR-7 | FA2 prefill vs upstream `flash-attn` v2 | within 15% | OK to be slower |
| PR-8 | FA2 decode (bs=1, sl=4096+1) vs SDPA math | >= 3.0x | |
| PR-9 | FA2 Tensor Core SOL% | >= 70% | Nsight Compute Compute Workload |
| PR-10 | E2E 7B Llama block forward speedup | >= 2.5x | vs unfused PyTorch |
| PR-11 | Numerical: max abs err vs FP32 ref | <= 1e-2 | BF16 |
| PR-12 | Numerical: relative err vs FP32 ref | <= 1e-3 | BF16 |
| PR-13 | Decode CUDA Graph replay overhead | <= 5 us | Soft |
| PR-14 | H100 WGMMA + TMA attention SOL% | >= 80% Tensor Core peak | Soft / H100 only |
| PR-15 | H100 FP8 attention vs BF16 speedup | >= 1.5x | Soft / H100 only |

## 4. Non-functional requirements

### NFR-1: Code structure

- One kernel per `.cu` file.
- Each `.cu` <= 800 LOC.
- Header file exposes signatures + tile config struct.
- Type-safe pybind11 binding (no `void*`).
- Python wrapper has type hints, dtype + shape asserts before launch.

### NFR-2: Testing

- >= 80% line coverage on the Python wrapper.
- Every kernel has a correctness test in `tests/correctness/`.
- Every kernel with backward has a `gradcheck` test.
- A "tiny model" integration test runs in CI in <= 60 seconds.

### NFR-3: Documentation

- Each `.cu` has a header comment with: kernel purpose, tile config,
  occupancy target, register-budget target, shared-memory budget,
  expected SOL%.
- A short markdown writeup per kernel in `docs/kernels/` explaining
  the algorithm and the optimization choices.

### NFR-4: Reproducibility

- `Makefile` targets: `make all`, `make bench`, `make profile`,
  `make verify`, `make clean`.
- `Dockerfile` based on `nvcr.io/nvidia/pytorch:24.07-py3` or later.
- Bench results MUST reproduce within 5% across runs on the same SKU.

### NFR-5: Compilation hygiene

- No NVCC warnings at `-Wall -Wextra` (host) or
  `-Werror cross-execution-space-call` (device).
- No register spilling on the hot kernels — verified via
  `--ptxas-options=-v` output and recorded in
  `reports/build/ptxas_<kernel>.txt`.

### NFR-6: Security / safety

- All device pointers checked for non-null before launch.
- All shape and stride asserts must fail fast with a clear message.
- No kernel may launch with a workgroup that would exceed device
  shared-memory budget — checked at launch time.

## 5. Constraints

- Single-GPU only.
- BF16 (default) and FP16 (optional) for memory-format compatibility.
- INT8 / INT4 / FP4 NOT in scope.
- Head dim restricted to 64 and 128 (other dims may be silently
  rejected with a clear error).

## 6. Assumptions

- The reviewer has at least an A100 80GB. H100 is preferred.
- The reviewer has `flash-attn` v2 (or v3 on H100) installed for the
  baseline comparison; if missing, the bench harness reports "skip".
- The reviewer has Nsight Compute installed with kernel profiling
  permissions (`NVreg_RestrictProfilingToAdminUsers=0` on Linux, or
  running as root).

## 7. Dependencies (external)

| Component | Version | Reason |
|-----------|---------|--------|
| CUDA Toolkit | 12.4+ | Hopper WGMMA / TMA |
| cuDNN | 9.x | Reference baselines |
| PyTorch | 2.4+ | New extension API, BF16 stability |
| Triton | 3.x | Matches PyTorch 2.4 |
| CUTLASS | 3.5+ | CuTe references |
| FlashAttention | 2.5+ | Reference baseline |
| FlashAttention-3 | head | H100-only reference |
| TransformerEngine | 1.7+ | FP8 attention reference |
| Nsight Systems | 2024.2+ | Trace format |
| Nsight Compute | 2024.2+ | Roofline metrics |

## 8. Acceptance test sketch

```
git clone <repo>
cd project-02-gpu-optimization
docker build -t fastkernels:dev .
docker run --gpus all -v $PWD:/work fastkernels:dev make all
docker run --gpus all -v $PWD:/work fastkernels:dev make verify
```

Acceptance is granted if:

1. `make all` exit code 0.
2. `make verify` reproduces every bench number within 5% of
   `reports/bench_summary.json`.
3. Every PR-* hard gate in Section 3 is met.
4. `pytest` exits 0 with coverage >= 80% on the Python wrapper.
5. `reports/roofline_*.png` exists for each kernel.

## 9. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Bank conflicts in SMEM tile | High | Medium | Use 8-byte stride padding; verify via Nsight Compute "Shared Memory Bank Conflicts" |
| Register spilling > 0 | Medium | High | Check ptxas `-v` output; reduce tile size if spilling |
| WGMMA / TMA paths require Hopper | High | Medium | A100 fallback path with `cp.async` only; rubric notes Hopper-only ablation |
| FA2 backward numerical drift | Medium | High | Use FP32 accumulation in recompute; gate with `gradcheck` |
| `flash-attn` package version skew | Medium | Low | Pin version in Dockerfile; bench reports "vN.M" alongside numbers |
| Clock unlocking causes flaky bench | High | High | Hard gate `std/p50 < 5%`; abort + diagnostic |
| CUDA Graph capture fails on first compile | Medium | Medium | Warm the kernel cache with 10 launches before capture |
| H100 WGMMA needs sm_90a (not sm_90) | Medium | High | Explicit `--gpu-architecture=sm_90a` in build flags; documented |

## 10. Glossary

- **SOL%**: "Speed of light" percentage from Nsight Compute, i.e.
  achieved / theoretical peak.
- **HBM**: High Bandwidth Memory (DRAM on the GPU package).
- **WGMMA**: Warpgroup MMA on Hopper. Async, 4-warp-wide.
- **TMA**: Tensor Memory Accelerator on Hopper. Async bulk copy.
- **`cp.async`**: Asynchronous global-to-shared copy on sm_80+.
- **`mbarrier`**: Memory barrier object for producer/consumer sync.
- **Roofline**: Performance model relating arithmetic intensity
  (FLOPs/byte) to achievable throughput (GFLOPs/s).
- **Online softmax**: Streaming softmax that processes K/V tile-by-tile
  with running max + sum-of-exp, used in Flash Attention.
- **Welford's algorithm**: Numerically stable online mean and variance.
- **RoPE**: Rotary Position Embedding (Su et al. 2021).
- **CTA**: Cooperative Thread Array (a CUDA block).
