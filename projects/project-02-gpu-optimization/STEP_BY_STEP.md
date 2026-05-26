# Step-by-Step Build Guide — Custom CUDA Kernels for Transformer Optimization

This is the canonical walkthrough. Do the phases in order; do not skip
a gate check. Every code snippet here is illustrative; the real kernels
live in `csrc/`.

The full build is six weeks at ~10h/week. Use this document as your
weekly checklist.

---

## Phase 0 — Environment, before week 1 (2-3h)

### 0.1 Verify hardware and toolchain

```bash
nvidia-smi
nvidia-smi -q -d COMPUTE | grep -E "(Architecture|Compute Mode)"
nvcc --version                                  # >= 12.4
gcc --version                                   # >= 12.0 (nvcc host)
ncu --version && nsys --version                 # >= 2024.2
python -c "import torch; print(torch.cuda.get_device_capability())"
# Expected on H100: (9, 0). On A100: (8, 0).
```

### 0.2 Lock GPU clocks

```bash
sudo nvidia-smi --persistence-mode=1

# H100 SXM5 base SM clock: 1830 MHz; HBM3: 2619 MHz
sudo nvidia-smi --lock-gpu-clocks=1830,1830
sudo nvidia-smi --lock-memory-clocks=2619

# A100 80GB SXM base SM clock: 1410 MHz; HBM2e: 1593 MHz
sudo nvidia-smi --lock-gpu-clocks=1410,1410
sudo nvidia-smi --lock-memory-clocks=1593
```

**Gate 0**: `nvidia-smi -q -d CLOCK` shows clocks locked. If not,
abort and capture the deviation in the bench output.

### 0.3 Enable profiler access (Linux)

Nsight Compute requires either root or
`NVreg_RestrictProfilingToAdminUsers=0` in your modprobe options.
Without this, `ncu` will fail with "ERR_NVGPUCTRPERM".

### 0.4 Build the Docker image

```Dockerfile
FROM nvcr.io/nvidia/pytorch:24.07-py3
RUN pip install --no-cache-dir \
        triton==3.0.* \
        flash-attn==2.6.* --no-build-isolation \
        transformer-engine \
        pybind11 ninja pytest pytest-cov
WORKDIR /work
ENV TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0a"
```

### 0.5 Repo skeleton

```
project-02-gpu-optimization/
  README.md
  requirements.md
  architecture.md
  STEP_BY_STEP.md
  rubric.md
  Makefile
  Dockerfile
  setup.py
  pyproject.toml
  csrc/
    binding.cpp
    gelu_vec.cu
    layernorm_welford.cu
    qkv_rope_fused.cu
    attention_fwd.cu
    attention_bwd.cu
    attention_decode.cu
    include/
      tile_config.hpp
      mma_helpers.hpp
      async_copy.hpp
  triton/
    gelu_triton.py
    layernorm_triton.py
    qkv_rope_triton.py
    attention_triton.py
  python/fastkernels/
    __init__.py
    api.py
    autograd.py
    layers.py            # FastLlamaBlock
  bench/
    bench_harness.py
    bench_kernel.py
    roofline.py
    nsys_wrapper.py
    ncu_wrapper.py
  tests/
    correctness/
    gradcheck/
    integration/
  reports/
  profiles/
  docs/kernels/
  deliverables/
```

---

## Phase 1 — Bench harness + GELU + LayerNorm (week 1, 8-10h)

These are the easiest kernels. Build the harness here so subsequent
phases just plug in.

### 1.1 The bench harness

```python
# bench/bench_harness.py
import torch
import numpy as np

def bench(fn, *args, warmup=100, iters=1000):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(*args)
        ends[i].record()
    torch.cuda.synchronize()

    t = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    if t.std() / np.median(t) > 0.05:
        raise RuntimeError(
            f"std/p50 = {t.std()/np.median(t):.3f}; clocks not locked?"
        )
    return {
        "p50_ms": float(np.percentile(t, 50)),
        "p90_ms": float(np.percentile(t, 90)),
        "p99_ms": float(np.percentile(t, 99)),
        "mean_ms": float(t.mean()),
        "std_ms":  float(t.std()),
    }
```

### 1.2 GELU kernel

```cuda
// csrc/gelu_vec.cu
#include <cuda_bf16.h>
#include <cstdint>

__global__ void gelu_vec_bf16(const __nv_bfloat16* __restrict__ x,
                              __nv_bfloat16* __restrict__ y,
                              int64_t N) {
    constexpr int VEC = 8;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t idx = tid * VEC;
    if (idx >= N) return;

    int n = (idx + VEC <= N) ? VEC : (int)(N - idx);

    __nv_bfloat16 in[VEC] = {};
    if (n == VEC) {
        *reinterpret_cast<uint4*>(in) =
            *reinterpret_cast<const uint4*>(&x[idx]);
    } else {
        for (int i = 0; i < n; ++i) in[i] = x[idx + i];
    }

    __nv_bfloat16 out[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
        float v = __bfloat162float(in[i]);
        float t = tanhf(0.7978845608f * (v + 0.044715f * v * v * v));
        out[i] = __float2bfloat16(0.5f * v * (1.0f + t));
    }

    if (n == VEC) {
        *reinterpret_cast<uint4*>(&y[idx]) =
            *reinterpret_cast<uint4*>(out);
    } else {
        for (int i = 0; i < n; ++i) y[idx + i] = out[i];
    }
}
```

### 1.3 Build and bench

```bash
python setup.py develop
python -m bench.bench_kernel gelu_bf16 --shape 8,4096,14336
# Expected on H100: ~250 us at 4096 tokens, ~92% DRAM SOL
```

### 1.4 Profile GELU

```bash
ncu --launch-skip 50 --launch-count 1 --set full \
    -o profiles/ncu_gelu_bf16 \
    python -m bench.bench_kernel gelu_bf16 --shape 8,4096,14336
ncu --import profiles/ncu_gelu_bf16.ncu-rep \
    --csv --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed
```

### 1.5 Welford LayerNorm

The key trick: each block handles one row; warp-shuffle reduces; second
warp-shuffle broadcasts mean/rstd.

```cuda
// Sketch — full kernel in csrc/layernorm_welford.cu
__device__ float2 welford_merge(float2 a, float2 b, int na, int nb) {
    int n = na + nb;
    float delta = b.x - a.x;
    float mean = a.x + delta * (float)nb / (float)n;
    float m2 = a.y + b.y + delta * delta * (float)na * (float)nb / (float)n;
    return {mean, m2};
}

__global__ void layernorm_welford_bf16(
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ gamma,
        const __nv_bfloat16* __restrict__ beta,
        __nv_bfloat16* __restrict__ y,
        int rows, int cols, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const __nv_bfloat16* xr = x + row * cols;
    __nv_bfloat16* yr = y + row * cols;

    // Pass 1: Welford
    float local_mean = 0.f, local_m2 = 0.f;
    int local_n = 0;
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = __bfloat162float(xr[i]);
        local_n += 1;
        float delta = v - local_mean;
        local_mean += delta / (float)local_n;
        local_m2   += delta * (v - local_mean);
    }
    // Warp reduce (xor shuffle), then block reduce via SMEM.
    // ... (omitted; uses __shfl_xor_sync)
    __shared__ float s_mean, s_rstd;
    // ... store final mean/rstd in s_mean/s_rstd

    // Pass 2: normalize + affine
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = (__bfloat162float(xr[i]) - s_mean) * s_rstd;
        v = v * __bfloat162float(gamma[i]) + __bfloat162float(beta[i]);
        yr[i] = __float2bfloat16(v);
    }
}
```

### Gate 1

- [ ] GELU: speedup >= 2.0x vs `F.gelu`; DRAM SOL >= 90%.
- [ ] LayerNorm: speedup >= 3.0x vs `torch.nn.LayerNorm`; DRAM SOL >= 80%.
- [ ] Numerical: max abs err <= 1e-2 BF16 for both.
- [ ] `bench_harness` aborts when clocks unlocked (test by unlocking
      briefly).

### Gotchas

- **GELU boundary epilogue silently corrupts on non-multiple-of-8
  shapes**: write a fuzz test that runs shapes (N+1, N+7) for N in
  {1024, 4096, 14336}.
- **LayerNorm with cols not a multiple of 32 may shuffle from invalid
  lanes**: use `__shfl_sync` with a computed mask, or pad the
  reduction.
- **Calling `tanhf` on FP32 is faster than `__tanhf` on H100**: the
  compiler swaps to MUFU for you. Don't second-guess it.

---

## Phase 2 — Fused QKV + RoPE (week 2, 10-12h)

### 2.1 Tile config

```cpp
// csrc/include/tile_config.hpp
struct QkvRopeTile {
    int BM = 128;   // tokens per CTA
    int BN = 384;   // 3 * 128 (head_dim) (output cols per CTA)
    int BK = 64;    // hidden chunked
    int stages = 3; // cp.async double-/triple-buffer depth
};
```

### 2.2 Ampere path (sm_80)

```cuda
// csrc/qkv_rope_fused.cu — Ampere variant
//
// Tile layout: BM=128 tokens, output cols 384 (3 * head_dim).
// One CTA = 128 threads = 4 warps.
// Each warp owns a 16-row x 384-col slab of the output.
//
// Hot loop:
//   for k in 0..K/BK:
//     cp.async load X[BM,BK]  -> smem_x[stage]
//     cp.async load W[BK,BN]  -> smem_w[stage]
//     wait stage-(N-1)
//     mma.sync.aligned.m16n8k16  C += A * B
//   epilogue:
//     for each output element (row=token, col=qkv_lane):
//       if col in Q range or col in K range:
//         apply RoPE rotation using (token_idx, head_dim_idx)
//     store to HBM (uint4 vectorized)

asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n"
    :: "r"(smem_ptr), "l"(gmem_ptr), "n"(16), "r"(predicate));
asm volatile("cp.async.commit_group;\n");
asm volatile("cp.async.wait_group %0;\n" :: "n"(stages - 1));

asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
      "r"(b[0]), "r"(b[1]),
      "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
```

### 2.3 Hopper path (sm_90a)

```cuda
// Hopper-only: WGMMA + TMA. One warpgroup (128 threads) per CTA.
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
    "{%0,...}, %64, %65, %66, %67, %68;\n"
    : /* accumulator regs */
    : "l"(desc_a), "l"(desc_b), "n"(0), "n"(0), "n"(0));
asm volatile("wgmma.commit_group.sync.aligned;\n");
asm volatile("wgmma.wait_group.sync.aligned 0;\n");
```

### 2.4 RoPE in the epilogue

```cuda
// For Q and K only (not V). q[2i], q[2i+1] is a complex pair.
float cos_v = cosf(pos * inv_freq[i]);
float sin_v = sinf(pos * inv_freq[i]);
float q0 = q[2*i],  q1 = q[2*i + 1];
q[2*i]     = q0 * cos_v - q1 * sin_v;
q[2*i + 1] = q0 * sin_v + q1 * cos_v;
```

### 2.5 Verify timeline collapse

```bash
nsys profile --trace=cuda,nvtx -o profiles/nsys_qkv_rope \
    python -m bench.bench_kernel qkv_rope --shape 16,4096,4096
nsys stats --report cudaapisum profiles/nsys_qkv_rope.nsys-rep
# Expected: ONE kernel `qkv_rope_fused_bf16`. The standalone
# `mul_add` or `rotate_half` kernel must be ABSENT.
```

### Gate 2

- [ ] Fused QKV + RoPE >= 3.5x vs unfused PyTorch.
- [ ] Standalone RoPE kernel absent from timeline.
- [ ] `ptxas --verbose` reports 0 spilled bytes.
- [ ] Tensor Core SOL% >= 50% (Nsight Compute).

### Gotchas

- **`cp.async` barrier mismatch causes silent corruption**: every
  `commit_group` MUST be matched by a `wait_group`. Test with an
  assertion barrier inserted between stages.
- **Synchronous PTX MMA shapes vary by arch**: `m16n8k16` on Ampere,
  `m16n8k16` also works on Hopper but WGMMA is the right fit.
- **RoPE sin/cos at FP32 is too slow if recomputed per-element**:
  precompute `inv_freq[head_dim/2]` into a constant.

---

## Phase 3 — Flash Attention v2 forward (week 3, 12-15h)

### 3.1 Algorithm in pseudocode (single Q tile)

```
Inputs:  Q  [Br, d]       in SMEM
         K  [N, d], V  [N, d]   in HBM
Outputs: O  [Br, d]       in HBM
         m  [Br], l [Br]  in HBM (saved for backward)

m_i = -inf  (in registers, per row)
l_i =  0
O_i =  0    (FP32 acc in registers)

for j in 0..ceil(N / Bc):
    if causal and tile j is entirely above this Q tile: continue
    K_j = load K[j*Bc:(j+1)*Bc]    via cp.async to SMEM
    V_j = load V[j*Bc:(j+1)*Bc]    via cp.async to SMEM
    cp.async.wait (stage - 1)

    S_ij = Q_i @ K_j^T            via mma.sync       [Br, Bc]
    if causal: mask diagonal inside this tile

    m_ij = rowmax(S_ij)
    P_ij = exp(S_ij - m_ij)        FP32
    l_ij = rowsum(P_ij)

    m_new = max(m_i, m_ij)
    alpha = exp(m_i  - m_new)
    beta  = exp(m_ij - m_new)
    l_new = alpha * l_i + beta * l_ij

    O_i = alpha * O_i + beta * (P_ij @ V_j)   via mma.sync
    m_i, l_i = m_new, l_new

O_i = O_i / l_i
store O_i, m_i, l_i to HBM
```

### 3.2 Tile config decision table

| sm | head_dim | Br | Bc | stages | SMEM / CTA |
|----|----------|-----|-----|--------|-------------|
| 80 | 64       | 128 | 64  | 2      | 24 KB       |
| 80 | 128      | 64  | 64  | 2      | 32 KB       |
| 90a | 128     | 128 | 128 | 3      | 96 KB       |

### 3.3 Profile sanity check

```bash
ncu --launch-skip 100 --launch-count 1 --set full \
    -o profiles/ncu_fa2_fwd \
    python -m bench.bench_kernel fa2_fwd --bs 16 --sl 4096 --d 128
```

In the resulting report check:

| Metric | Section in Nsight Compute | Target |
|--------|---------------------------|--------|
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | Compute Workload Analysis | >= 60% |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | Memory Workload | reasonable but not the gate |
| Achieved occupancy | Occupancy | >= 25% |
| Shared memory bank conflicts | Memory Workload -> Shared | 0 (or near 0) |
| Register spills (loads + stores) | Source View | 0 |

### Gate 3

- [ ] FA2 prefill (bs=16, sl=4096, d=128): >= 3.0x vs SDPA math.
- [ ] Within 15% of upstream `flash-attn` v2.
- [ ] Tensor Core SOL% >= 70%.
- [ ] Numerical: max abs err <= 1e-2, rel err <= 1e-3.

### Gotchas

- **Online softmax rescale order matters numerically**: rescale O
  using `alpha = exp(m_i - m_new)`, not `exp(m_i) / exp(m_new)` — the
  latter overflows in FP32.
- **Diagonal-tile causal mask off-by-one**: token at sequence position
  `q_pos` may attend to position `k_pos <= q_pos`. Check both ends of
  the tile boundary.
- **`cp.async` reading past the end of K causes nondeterministic
  data**: predicate the load or pad K with zeros and rely on the
  softmax `-inf` mask.
- **MMA fragment layout differs row vs col**: use `mma_helpers.hpp`
  abstractions; don't hand-decode register tiles in every kernel.

---

## Phase 4 — FA2 backward + decode (week 4, 10-12h)

### 4.1 Backward recomputation

```
Inputs:  Q, K, V, O, m, l   (all from forward)
         dO                  (gradient w.r.t. output)
Outputs: dQ, dK, dV

D_i = rowsum(O_i * dO_i)          (precomputed, [N])

for j in 0..N/Bc:
    K_j, V_j = load
    for i in 0..N/Br:
        Q_i, dO_i, m_i, l_i, D_i = load
        S_ij = Q_i @ K_j^T
        P_ij = exp(S_ij - m_i)    (recompute attention prob)
        dV_j += P_ij^T @ dO_i
        dP_ij = dO_i @ V_j^T
        dS_ij = P_ij * (dP_ij - D_i)
        dQ_i += dS_ij @ K_j
        dK_j += dS_ij^T @ Q_i
```

### 4.2 `gradcheck` test

```python
# tests/gradcheck/test_attention_bwd.py
import torch
from fastkernels import flash_attn_fwd

def test_gradcheck_small():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 64, 64, dtype=torch.float64,
                    device="cuda", requires_grad=True)
    k = torch.randn(1, 4, 64, 64, dtype=torch.float64,
                    device="cuda", requires_grad=True)
    v = torch.randn(1, 4, 64, 64, dtype=torch.float64,
                    device="cuda", requires_grad=True)
    # `gradcheck` requires double; we provide a FP64 reference path
    # for testing only. Production kernel is BF16.
    torch.autograd.gradcheck(
        lambda Q, K, V: flash_attn_fwd(Q, K, V, causal=True, ref=True),
        (q, k, v), eps=1e-6, atol=1e-4
    )
```

### 4.3 Decode kernel skeleton

```cuda
// csrc/attention_decode.cu
// One Q row, full K/V history. No Q tiling.
__global__ void fa_decode_bf16(
    const __nv_bfloat16* __restrict__ Q,   // [bs, h, 1, d]
    const __nv_bfloat16* __restrict__ K,   // [bs, h, sl, d]
    const __nv_bfloat16* __restrict__ V,   // [bs, h, sl, d]
    __nv_bfloat16* __restrict__ O,         // [bs, h, 1, d]
    int sl, int d, float scale)
{
    // ... one CTA per (batch, head)
    // ... load Q row to registers (fits: d*2 bytes <= 256 bytes)
    // ... tile K, V along sl in chunks of Bc=256
    // ... online softmax in registers
    // ... store O row
}
```

### 4.4 CUDA Graph capture

```python
# python/fastkernels/api.py
def make_decode_graph(q, k_cache, v_cache):
    # Warm
    for _ in range(10):
        _ = flash_attn_decode(q, k_cache, v_cache)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = flash_attn_decode(q, k_cache, v_cache)
    return g, out
```

### Gate 4

- [ ] Backward `gradcheck` passes.
- [ ] Backward matches `flash_attn` bwd within 1e-2 abs.
- [ ] Decode >= 3.0x vs SDPA math.
- [ ] CUDA Graph replay <= 5 us warm.

### Gotchas

- **`gradcheck` is slow at FP64**: keep the test to sl <= 64, d <= 64.
- **Decode latency hidden by graph capture warmup**: measure ONLY
  steady-state replay time, not the first replay.
- **KV cache stride mismatch**: K/V from HuggingFace are
  `[bs, h, sl, d]`. Some pipelines store as `[bs, sl, h, d]`. Assert
  contiguity at the binding layer.

---

## Phase 5 — Triton ports + Hopper paths (week 5, 10-12h)

### 5.1 Triton attention skeleton

```python
# triton/attention_triton.py
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BR": 64,  "BC": 64},  num_warps=4, num_stages=2),
        triton.Config({"BR": 128, "BC": 64},  num_warps=4, num_stages=2),
        triton.Config({"BR": 128, "BC": 128}, num_warps=8, num_stages=3),
    ],
    key=["sl", "d"],
)
@triton.jit
def _attn_fwd(Q, K, V, O, M, L, sl, d,
              stride_qb, stride_qh, stride_qs, stride_qd,
              BR: tl.constexpr, BC: tl.constexpr,
              CAUSAL: tl.constexpr):
    # ... q_block = tl.load(...); for k_block in range(...): ...
    pass
```

### 5.2 SASS comparison

```bash
# Extract SASS for one tile of the CUDA attention forward
cuobjdump --dump-sass build/lib*/fastkernels*.so \
    | awk '/attention_fwd/,/^Function/' > reports/sass_fa2_cuda.txt

# Extract SASS for the Triton-compiled kernel
TRITON_CACHE_DIR=/tmp/triton_cache python -m bench.bench_kernel triton_fa2 ...
find /tmp/triton_cache -name "*.cubin" \
    | head -1 | xargs cuobjdump --dump-sass > reports/sass_fa2_triton.txt

diff <(grep -E "MMA|LDG|STG|BAR" reports/sass_fa2_cuda.txt) \
     <(grep -E "MMA|LDG|STG|BAR" reports/sass_fa2_triton.txt) \
    > reports/sass_diff_fa2.txt
```

### 5.3 Hopper WGMMA + TMA path

```cuda
// csrc/attention_fwd.cu — Hopper variant (sm_90a)
//
// Producer warps: issue cp.async.bulk.tensor.5d (TMA) for K/V tiles.
// Consumer warpgroup: wgmma.mma_async on Q @ K^T and P @ V.
// Sync via mbarrier objects in SMEM.
//
// Key PTX:
//   cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx
//   wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16
//   mbarrier.try_wait.parity
//
// See PTX ISA 8.5 sections 9.7.13 (cp.async.bulk) and 9.7.14 (wgmma).
```

### Gate 5

- [ ] Triton attention within 10% of hand-tuned CUDA on A100.
- [ ] One written SASS-comparison note per kernel.
- [ ] On H100: WGMMA + TMA path matches hand-tuned within 5%.
- [ ] FlashAttention-3 baseline run if installed (rubric data point).

### Gotchas

- **Triton autotune cache invalidation**: a different stride breaks
  the cache key; pass strides explicitly.
- **WGMMA needs sm_90a, not sm_90**: NVCC silently lowers to sm_90
  otherwise and you get an illegal instruction at runtime.
- **TMA descriptor layout**: 5-d for image/video, 2-d for matrix.
  Use `cuTensorMapEncodeTiled` to build descriptors at host time;
  do not encode by hand.

---

## Phase 6 — Integration + report (week 6, 8-10h)

### 6.1 FastLlamaBlock

```python
# python/fastkernels/layers.py
class FastLlamaBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = FastLayerNorm(cfg.hidden)
        self.qkv = FastQkvRope(cfg.hidden, cfg.heads, cfg.head_dim)
        self.attn = FastAttention(causal=True)
        self.out_proj = torch.nn.Linear(cfg.hidden, cfg.hidden, bias=False)
        self.ln2 = FastLayerNorm(cfg.hidden)
        self.gate_proj = torch.nn.Linear(cfg.hidden, cfg.ffn, bias=False)
        self.up_proj   = torch.nn.Linear(cfg.hidden, cfg.ffn, bias=False)
        self.down_proj = torch.nn.Linear(cfg.ffn, cfg.hidden, bias=False)

    def forward(self, x, kv_cache=None):
        h = self.ln1(x)
        q, k, v = self.qkv(h)
        a = self.attn(q, k, v, kv_cache=kv_cache)
        x = x + self.out_proj(a)
        h = self.ln2(x)
        x = x + self.down_proj(fast_silu(self.gate_proj(h)) * self.up_proj(h))
        return x
```

### 6.2 End-to-end bench

```bash
python -m bench.bench_e2e \
    --model llama-7b \
    --variant fast \
    --bs 1 --sl 4096 \
    --iters 200 > reports/bench_e2e_fast.json

python -m bench.bench_e2e \
    --model llama-7b \
    --variant pytorch_unfused \
    --bs 1 --sl 4096 \
    --iters 200 > reports/bench_e2e_baseline.json

python -m bench.compare \
    reports/bench_e2e_baseline.json reports/bench_e2e_fast.json \
    > reports/e2e_summary.md
```

### 6.3 Roofline plot per kernel

```python
# bench/roofline.py
# For each kernel:
#   1. Run with `ncu --metrics` to get achieved GFLOPs and GB/s.
#   2. Compute arithmetic intensity = GFLOPs / GB.
#   3. Plot against the hardware ridge.
#       H100 BF16:  989 TFLOPs / 3.35 TB/s = 295 ops/byte ridge.
#       A100 BF16:  312 TFLOPs / 1.555 TB/s = 200 ops/byte ridge.
#   4. Annotate which kernel sits where.
```

### 6.4 Reviewer-ready summary

```bash
python -m bench.summarize \
    --include-rooflines \
    --include-sols \
    --include-e2e \
    > reports/kernel_summary.md
```

### Gate 6

- [ ] End-to-end Llama-7B block forward >= 2.5x vs unfused PyTorch.
- [ ] Nsight Systems timeline: 4 kernels per layer instead of ~12.
- [ ] Roofline plot for each kernel with annotation.
- [ ] `reports/kernel_summary.md` is reviewer-readable.
- [ ] `make all` works from clean checkout in Docker.

### Final gotchas

- **`torch.compile` on the wrapper module disables CUDA Graph
  capture**: capture before compile, or capture the compiled callable.
- **`flash-attn` installed via pip uses CUDA 11.8 wheels by default**:
  re-build from source against the project's CUDA 12.4 in the
  Dockerfile or the speedup comparison will be unfair.
- **Bench std/p50 creeps over 5% as the GPU warms**: insert a 30s
  cooldown between variants; or run them in interleaved order and
  average.

---

## Appendix A — Profiling commands quick reference

```bash
# Nsight Systems: full timeline
nsys profile -o profiles/nsys_${RUN} \
    --trace=cuda,cudnn,cublas,nvtx \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    python -m bench.bench_kernel ${KERNEL}

# Nsight Compute: one launch, full metrics
ncu -o profiles/ncu_${RUN} \
    --launch-skip 100 --launch-count 1 \
    --set full --target-processes all \
    python -m bench.bench_kernel ${KERNEL}

# Roofline section only (faster than `--set full`)
ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section LaunchStats \
    --section Occupancy \
    -o profiles/ncu_${RUN}_lite \
    --launch-skip 100 --launch-count 1 \
    python -m bench.bench_kernel ${KERNEL}

# Extract SOL% as CSV
ncu --import profiles/ncu_${RUN}.ncu-rep --csv \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread
```

## Appendix B — Reading ptxas output

```bash
nvcc -c csrc/attention_fwd.cu -arch=sm_90a \
     --ptxas-options=-v -o /tmp/attn.o 2>&1 \
     | tee reports/build/ptxas_attention_fwd.txt

# Expected lines:
#   ptxas info  : Used N registers, M stack, K bytes smem, ...
#   ptxas info  : Stack frame: 0 bytes, spill stores: 0, spill loads: 0
#
# Spill stores or loads > 0 means you blew the register budget.
# Stack frame > 0 means you have local memory in HBM. Both kill perf.
```

## Appendix C — When the build breaks

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `error: identifier "wgmma" is undefined` | Built for sm_90 not sm_90a | `TORCH_CUDA_ARCH_LIST="9.0a"` |
| Kernel returns NaN | FP32 accumulator initialized to denorm or BF16 overflow | Init acc to 0.0f; use FP32 for reductions |
| `CUDA error: invalid argument` on launch | SMEM > device max | Lower BR/BC; check `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)` |
| Decode CUDA Graph capture fails | Module dispatched to a different kernel on first call | Warm 10 launches before capture; ensure same shape |
| Triton autotune picks "best" but slow | Cache key missed shape; recompiling each call | Pass shape explicitly in `key=` |
| `flash-attn` bench segfaults | Built against wrong torch version | Reinstall `flash-attn --no-build-isolation` in the image |
