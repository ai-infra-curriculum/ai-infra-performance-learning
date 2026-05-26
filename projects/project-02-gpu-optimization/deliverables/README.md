# Deliverables — Custom CUDA Kernels for Transformer Optimization

A reviewer should unzip a submission into this directory, run
`make verify` from the project root, and reproduce every published
number.

## 1. Required submission inventory

```
deliverables/
  csrc/                           # Raw CUDA C++ kernels
    attention_fwd.cu
    attention_bwd.cu
    attention_decode.cu
    qkv_rope_fused.cu
    layernorm_welford.cu
    gelu_vec.cu
    binding.cpp
    include/
      tile_config.hpp
      mma_helpers.hpp
      async_copy.hpp
  triton/                         # Triton equivalents
    attention_triton.py
    qkv_rope_triton.py
    layernorm_triton.py
    gelu_triton.py
  python/fastkernels/             # Importable wrapper package
    __init__.py
    api.py
    autograd.py
    layers.py
  bench/                          # Benchmark harness
  tests/                          # pytest tree
    correctness/
    gradcheck/
    integration/
  reports/
    bench_summary.md              # Single human-readable table
    bench_<kernel>_<dtype>_<shape>.json
    e2e_summary.md
    sols_memory.md
    sols_compute.md
    roofline_<kernel>.png
    roofline_<kernel>.csv
    sass_<kernel>_cuda.txt
    sass_<kernel>_triton.txt
    sass_diff_<kernel>.txt
    numerical_fuzz.md
    tile_sweep_<kernel>.csv
    hopper_path.md                # H100 submissions only
    build/
      ptxas_<kernel>.txt
  profiles/
    nsys_<kernel>.nsys-rep        # >= 1 per kernel
    ncu_<kernel>.ncu-rep          # >= 1 per kernel
    nsys_e2e.nsys-rep
    ncu_e2e_hottest.ncu-rep
  docs/kernels/                   # Per-kernel writeups
    attention.md
    qkv_rope.md
    layernorm.md
    gelu.md
  setup.py
  pyproject.toml
  Makefile
  Dockerfile
  README_submission.md
```

## 2. `reports/bench_summary.md` schema

```
| Kernel                       | dtype | shape                  | hw   | PyTorch (ms) | Ours (ms) | flash-attn (ms) | Speedup vs PT | Mem SOL% | Compute SOL% | Gate |
|------------------------------|-------|------------------------|------|--------------|-----------|------------------|---------------|----------|--------------|------|
| GELU                         | bf16  | 8x4096x14336           | h100 | 0.520        | 0.221     | n/a              | 2.35x         | 92.1     | n/a          | PASS |
| LayerNorm                    | bf16  | 8x4096x4096            | h100 | 0.348        | 0.106     | n/a              | 3.28x         | 84.0     | n/a          | PASS |
| QKV+RoPE fused               | bf16  | 16x4096 hidden 4096    | h100 | 1.840        | 0.491     | n/a              | 3.75x         | 38.2     | 64.7         | PASS |
| FA2 fwd                      | bf16  | bs=16 sl=4096 d=128    | h100 | 8.20         | 2.34      | 2.18             | 3.50x         | 41.0     | 73.2         | PASS |
| FA2 decode                   | bf16  | bs=1 sl=4096+1 d=128   | h100 | 0.184        | 0.052     | 0.048            | 3.54x         | 18.0     | n/a          | PASS |
| E2E Llama-7B block           | bf16  | bs=1 sl=4096           | h100 | 19.2         | 7.1       | n/a              | 2.70x         | n/a      | n/a          | PASS |
```

## 3. `reports/sols_memory.md` schema

Per memory-bound kernel:

```
Kernel: LayerNorm (Welford, BF16)
Shape:  rows=8*4096, cols=4096
Hardware: H100 SXM5

DRAM throughput SOL:        84.0%   (target >= 80%)   PASS
L2 cache hit rate:          88.3%
Achieved occupancy:         24%
Theoretical occupancy:      25%
Shared memory bank conflicts: 0
Register spills:            0

Analysis:
  - Single-pass Welford eliminates the second HBM read PyTorch does in BF16.
  - L2 hit rate is high because gamma/beta are small and persistent.
  - The remaining 16% to peak is L2->SM transfer overhead during the
    second normalization pass (gamma/beta read).
```

## 4. `reports/roofline_<kernel>.png`

X-axis: arithmetic intensity (FLOPs / byte).
Y-axis: achieved GFLOPs/s.
Roofline: hardware ridge.
  - H100 BF16: 989 TFLOPs / 3.35 TB/s -> ridge at AI = 295 ops/byte.
  - A100 BF16: 312 TFLOPs / 1.555 TB/s -> ridge at AI = 200 ops/byte.

At least one annotated point per kernel showing where it sits.

## 5. `reports/sass_diff_<kernel>.txt`

For at least the attention kernel: diff of MMA/LDG/STG/BAR opcodes
between hand-tuned CUDA and Triton-compiled SASS. Format:

```
--- cuda  (one full tile of FA2 fwd)
+++ triton

  MMA   x 16
  LDGSTS x 4   (cp.async)
  BAR.SYNC x 2
  STG x 1

vs

  MMA   x 16
  LDGSTS x 4
  BAR.SYNC x 3      <-- Triton emits an extra barrier
  STG x 1

Cost of extra BAR.SYNC: measured at 2.3% on the inner loop.
```

## 6. `reports/numerical_fuzz.md`

```
Test:  random shapes, random strides, BF16 vs FP32 reference.
Trials: 1000 per kernel.

| Kernel       | Max abs err  | Max rel err  | Tolerance | Status |
|--------------|--------------|--------------|-----------|--------|
| GELU         | 7.8e-4       | 4.2e-4       | 1e-3      | PASS   |
| LayerNorm    | 4.3e-3       | 8.0e-4       | 1e-2      | PASS   |
| QKV+RoPE     | 5.2e-3       | 9.1e-4       | 1e-2      | PASS   |
| FA2 fwd      | 8.0e-3       | 9.6e-4       | 1e-2      | PASS   |
| FA2 bwd      | 9.4e-3       | 1.1e-3       | 1e-2      | PASS   |
```

## 7. Profiles — capture command reference

```bash
# Nsight Systems per kernel
nsys profile -o profiles/nsys_${KERNEL} \
  --trace=cuda,cudnn,cublas,nvtx \
  python -m bench.bench_kernel ${KERNEL} --shape ${SHAPE}

# Nsight Compute one launch full metrics per kernel
ncu -o profiles/ncu_${KERNEL} \
  --launch-skip 100 --launch-count 1 --set full \
  python -m bench.bench_kernel ${KERNEL} --shape ${SHAPE}

# End-to-end
nsys profile -o profiles/nsys_e2e --trace=cuda,nvtx \
  python -m bench.bench_e2e --model llama-7b --variant fast
```

## 8. Tests

```
tests/
  correctness/
    test_gelu.py
    test_layernorm.py
    test_qkv_rope.py
    test_attention_fwd.py
    test_attention_decode.py
  gradcheck/
    test_attention_bwd_gradcheck.py
    test_layernorm_gradcheck.py
  integration/
    test_fast_llama_block.py
  fuzz/
    test_numerical_fuzz.py
```

Coverage >= 80% on `python/fastkernels/`.

## 9. Reproducibility

```bash
make all          # Build + bench + profile + report
make bench        # Bench only (reuses built kernels)
make profile      # Profile only
make verify       # Reruns bench, compares to bench_summary.json
make clean        # Wipe build/ and reports/
```

`make verify` MUST reproduce bench numbers within 5% on the same SKU.
On different SKU (e.g. A100 vs H100), use `make verify SKU=a100` and
the tolerance widens to 15%; H100-only metrics are skipped.

## 10. What NOT to submit

- Intermediate compile artifacts (`build/`, `*.o`, `*.cubin`).
- `__pycache__`, `.pytest_cache`, `*.egg-info`.
- Nsight Compute "full" reports from training runs (only the
  hottest-kernel ones; the full report can be > 1 GB).
- Triton cache directory (`~/.triton/cache/`).
- Any single file > 500 MB without justification.

## 11. Submission checklist

- [ ] All files in Section 1 present.
- [ ] `make all` clean run completes on documented hardware.
- [ ] `make verify` numbers within 5% of `bench_summary.md`.
- [ ] All PR-* hard gates met (see `requirements.md` Section 3).
- [ ] `pytest` exit 0, coverage >= 80%.
- [ ] No register spills on any of the four hot kernels.
- [ ] `README_submission.md` declares hardware used and any deviations.
