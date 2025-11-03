# Module 2: CUDA Programming - Exercises

This directory contains 6 comprehensive hands-on exercises for implementing custom CUDA kernels for machine learning operations. You'll build production-ready kernels and integrate them with PyTorch.

## Prerequisites

- Completed Module 1 (Performance Fundamentals)
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- CUDA Toolkit 12.0+ installed
- PyTorch 2.1+ with CUDA support
- C++ compiler (g++ or clang++)
- Basic understanding of PyTorch autograd

## Environment Setup

```bash
# Create Python environment
conda create -n cuda-dev python=3.10
conda activate cuda-dev

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install development tools
pip install ninja pytest pytest-benchmark matplotlib pandas
pip install pycuda cupy-cuda12x

# Verify setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
nvcc --version
```

## Exercises Overview

| Exercise | Topic | Difficulty | Time | Key Skills |
|----------|-------|------------|------|------------|
| 01 | Fused Elementwise Operations | Medium | 4h | Kernel fusion, vectorization, PyTorch integration |
| 02 | Optimized Reductions | Hard | 5h | Warp primitives, online algorithms, multi-dimensional |
| 03 | Custom Activation Functions | Medium | 4h | GELU, SiLU, fast math, gradient kernels |
| 04 | LayerNorm Implementation | Hard | 6h | Welford's algorithm, fusion, backward pass |
| 05 | Flash Attention Basics | Expert | 8h | Tiling, online softmax, memory efficiency |
| 06 | End-to-End Integration | Hard | 5h | PyTorch C++ extension, testing, benchmarking |

**Total Time**: ~30 hours

---

## Exercise 01: Fused Elementwise Operations

**Difficulty**: Medium | **Time**: 4 hours

### Learning Objectives

By completing this exercise, you will:
- Implement kernel fusion to eliminate memory transfers
- Use vectorized memory access (float4) for performance
- Integrate custom CUDA kernels with PyTorch
- Benchmark and validate against PyTorch native operations
- Understand the performance impact of kernel fusion

### Background

Elementwise operations are memory-bound. Chaining multiple operations causes excessive memory traffic:

```python
# 5 kernel launches, 10 memory transfers
x = torch.relu(x)           # Load x, store x
x = x * 0.5                 # Load x, store x
x = x + bias                # Load x, load bias, store x
x = torch.dropout(x, 0.1)   # Load x, store x, store mask
```

**Goal**: Fuse into single kernel → 1 launch, 3 memory transfers (load x, load bias, store x + mask)

### Part 1: Implement Fused ReLU-Scale-BiasAdd (2 hours)

**Task**: Implement a fused kernel combining ReLU, scaling, and bias addition.

```cuda
// TODO: Create csrc/fused_ops.cu

#include <cuda_runtime.h>
#include <torch/extension.h>

/**
 * Fused kernel: ReLU → Scale → BiasAdd
 *
 * Performance target: >80% of memory bandwidth
 * Expected speedup: 3-4x vs separate operations
 */
__global__ void fused_relu_scale_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float scale,
    const int n
) {
    // TODO: Implement using vectorized loads (float4)
    // TODO: Process 4 elements per thread
    // TODO: Apply operations in registers without intermediate stores

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx4 = idx * 4;

    if (idx4 + 3 < n) {
        // TODO: Vectorized load of input (float4)
        // TODO: Vectorized load of bias (float4)
        // TODO: Apply ReLU: max(0, x)
        // TODO: Apply scale: x * scale
        // TODO: Apply bias: x + bias
        // TODO: Vectorized store of output (float4)
    } else {
        // TODO: Handle remainder elements (idx4 to n)
    }
}

// Host function
torch::Tensor fused_relu_scale_bias_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scale
) {
    // TODO: Validate inputs
    // TODO: Allocate output tensor
    // TODO: Calculate launch configuration
    // TODO: Launch kernel
    // TODO: Check CUDA errors
    // TODO: Return output
}
```

**Python wrapper**:

```python
# TODO: Create python/fused_ops.py

import torch
import fused_ops_cuda  # Compiled extension

def fused_relu_scale_bias(input: torch.Tensor,
                          bias: torch.Tensor,
                          scale: float) -> torch.Tensor:
    """
    Fused ReLU → Scale → BiasAdd operation

    Args:
        input: Input tensor [N]
        bias: Bias tensor [N]
        scale: Scaling factor

    Returns:
        output: Processed tensor [N]
    """
    assert input.is_cuda, "Input must be on CUDA"
    assert bias.is_cuda, "Bias must be on CUDA"
    assert input.shape == bias.shape, "Shape mismatch"

    return fused_ops_cuda.fused_relu_scale_bias(input, bias, scale)
```

**Build configuration**:

```python
# TODO: Create setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_ops',
    ext_modules=[
        CUDAExtension(
            name='fused_ops_cuda',
            sources=[
                'csrc/fused_ops.cu',
                'csrc/bindings.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-use_fast_math',
                    '--expt-relaxed-constexpr',
                    '-gencode', 'arch=compute_80,code=sm_80',  # A100
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### Part 2: Benchmark Against PyTorch (1 hour)

**Task**: Measure speedup vs unfused PyTorch operations.

```python
# TODO: Create tests/test_fused_ops.py

import torch
import time
import fused_ops

def benchmark_unfused(input, bias, scale, num_iters=100):
    """Benchmark unfused PyTorch operations"""
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_iters):
        x = torch.relu(input)
        x = x * scale
        x = x + bias

    torch.cuda.synchronize()
    return (time.time() - start) / num_iters

def benchmark_fused(input, bias, scale, num_iters=100):
    """Benchmark fused custom kernel"""
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_iters):
        x = fused_ops.fused_relu_scale_bias(input, bias, scale)

    torch.cuda.synchronize()
    return (time.time() - start) / num_iters

def test_performance():
    sizes = [1000, 10000, 100000, 1000000, 10000000]

    print("Size       | Unfused | Fused   | Speedup | BW (Fused)")
    print("-----------|---------|---------|---------|------------")

    for n in sizes:
        input = torch.randn(n, device='cuda')
        bias = torch.randn(n, device='cuda')
        scale = 0.5

        # TODO: Run benchmarks
        # TODO: Calculate bandwidth utilization
        # TODO: Print results

        # Expected: 3-4x speedup, >80% bandwidth utilization

# TODO: Run test
if __name__ == '__main__':
    test_performance()
```

### Part 3: Add Dropout to Fusion (1 hour)

**Task**: Extend the kernel to include dropout.

```cuda
// TODO: Add to csrc/fused_ops.cu

#include <curand_kernel.h>

__global__ void fused_relu_scale_bias_dropout_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    uint8_t* __restrict__ dropout_mask,
    const float scale,
    const float dropout_prob,
    const int n,
    const unsigned long long seed
) {
    // TODO: Initialize cuRAND state per thread
    // TODO: Generate random number
    // TODO: Apply dropout mask
    // TODO: Scale by 1/(1-p) for training
    // TODO: Store mask for backward pass
}
```

### Success Criteria

- [ ] **Correctness**: Output matches PyTorch within 1e-5
- [ ] **Performance**: 3-4x speedup vs unfused operations
- [ ] **Bandwidth**: >80% of theoretical peak (1640+ GB/s on A100)
- [ ] **Vectorization**: Properly uses float4 loads/stores
- [ ] **Code quality**: Clean, documented, error-checked
- [ ] **Integration**: Successfully builds with PyTorch

### Testing Procedure

```bash
# Build extension
python setup.py develop

# Run tests
pytest tests/test_fused_ops.py -v

# Profile with Nsight Compute
ncu --set full -o fused_ops python tests/test_fused_ops.py
```

### Expected Output

```
Size       | Unfused | Fused   | Speedup | BW (Fused)
-----------|---------|---------|---------|------------
1K         | 0.012ms | 0.005ms | 2.4x    | 720 GB/s
10K        | 0.015ms | 0.006ms | 2.5x    | 1200 GB/s
100K       | 0.038ms | 0.012ms | 3.2x    | 1580 GB/s
1M         | 0.185ms | 0.048ms | 3.9x    | 1750 GB/s
10M        | 1.820ms | 0.465ms | 3.9x    | 1810 GB/s
```

---

## Exercise 02: Optimized Reductions

**Difficulty**: Hard | **Time**: 5 hours

### Learning Objectives

- Implement warp-level reduction using shuffle intrinsics
- Master block-level reduction with shared memory
- Handle multi-dimensional reductions efficiently
- Compare performance: naive vs warp-optimized vs atomic

### Part 1: Warp-Optimized Sum Reduction (2 hours)

**Task**: Implement efficient sum reduction using modern CUDA primitives.

```cuda
// TODO: Create csrc/reductions.cu

/**
 * Warp-level reduction using shuffle intrinsics
 *
 * Performance: 32x faster than shared memory for intra-warp
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // TODO: Implement using __shfl_down_sync
    // Loop: offset = 16, 8, 4, 2, 1
    // Each iteration: val += __shfl_down_sync(0xffffffff, val, offset)
    // Return: val (lane 0 has the sum)
}

/**
 * Block-level reduction using warp primitives
 */
__global__ void reduce_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    // TODO: Declare shared memory for warp sums
    __shared__ float warp_sums[32];  // Max 32 warps per block

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    // TODO: Each thread accumulates multiple elements (grid-stride loop)
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // TODO: Reduce within warp
    sum = warp_reduce_sum(sum);

    // TODO: First thread of each warp writes to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // TODO: First warp reduces all warp sums
    if (warp_id == 0) {
        float warp_sum = (tid < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        warp_sum = warp_reduce_sum(warp_sum);

        if (tid == 0) {
            // TODO: Atomic add for multi-block reduction
            atomicAdd(output, warp_sum);
        }
    }
}
```

### Part 2: Multi-Dimensional Reductions (2 hours)

**Task**: Implement spatial mean reduction for [B, C, H, W] → [B, C, 1, 1].

```cuda
/**
 * Reduce along spatial dimensions (H, W)
 * Input:  [B, C, H, W]
 * Output: [B, C, 1, 1]
 *
 * Launch: One block per (batch, channel)
 */
__global__ void reduce_spatial_mean_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int B, const int C, const int H, const int W
) {
    // TODO: Calculate batch and channel indices
    int bc = blockIdx.x;
    int b = bc / C;
    int c = bc % C;

    if (b >= B || c >= C) return;

    // TODO: Accumulate over spatial dimensions
    // TODO: Use warp reduction
    // TODO: Divide by H*W for mean
    // TODO: Write result to output[b, c, 0, 0]
}
```

### Part 3: Comparison and Benchmarking (1 hour)

Compare three approaches:
1. **Naive**: Tree reduction with shared memory (has warp divergence)
2. **Warp-optimized**: Uses shuffle intrinsics
3. **Atomic**: Direct atomic addition

```python
# TODO: Create tests/test_reductions.py

def benchmark_reduction(method: str, input: torch.Tensor):
    """Benchmark reduction method"""
    # TODO: Implement for all three methods
    # TODO: Measure time and achieved FLOPS
    pass

# Expected results:
# Naive:          2.45 ms,  98 GFLOPS
# Warp-optimized: 0.18 ms, 1333 GFLOPS (13.6x faster!)
# PyTorch:        0.15 ms, 1600 GFLOPS
```

### Success Criteria

- [ ] **Warp reduction**: Correct implementation with __shfl_down_sync
- [ ] **Performance**: Within 20% of PyTorch native reduction
- [ ] **Speedup**: 10x+ faster than naive tree reduction
- [ ] **Multi-dimensional**: Correct reduction along spatial dims
- [ ] **Validation**: Results match torch.mean within 1e-5

---

## Exercise 03: Custom Activation Functions

**Difficulty**: Medium | **Time**: 4 hours

### Learning Objectives

- Implement GELU and SiLU activation functions
- Use fast math intrinsics for performance
- Implement backward passes for autograd integration
- Benchmark accuracy vs performance trade-offs

### Part 1: GELU Forward and Backward (2 hours)

```cuda
// TODO: Create csrc/activations.cu

/**
 * GELU (Gaussian Error Linear Unit)
 * Forward: f(x) = 0.5x(1 + tanh(√(2/π) * (x + 0.044715x³)))
 */
__global__ void gelu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    // TODO: Implement using __tanf for fast approximation
    // TODO: Use vectorized loads (float4)
}

/**
 * GELU Backward
 * Gradient: ∂L/∂x = ∂L/∂y * ∂y/∂x
 * where ∂y/∂x = 0.5(1 + tanh(...)) + 0.5x * sech²(...) * (...)′
 */
__global__ void gelu_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    const int n
) {
    // TODO: Implement gradient calculation
    // TODO: Recompute forward pass values (cheaper than storing)
}
```

### Part 2: SiLU/Swish Implementation (1 hour)

```cuda
/**
 * SiLU (Sigmoid Linear Unit) / Swish
 * Forward: f(x) = x * sigmoid(x) = x / (1 + e^(-x))
 */
__device__ __forceinline__ float silu_forward(float x) {
    // TODO: Implement
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_backward(float x, float grad_out) {
    // TODO: Implement gradient
    // ∂y/∂x = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
}
```

### Part 3: Accuracy vs Performance Analysis (1 hour)

Compare three GELU implementations:

```python
# TODO: Create tests/test_activations.py

def compare_gelu_implementations():
    """Compare exact, approx, and fast GELU"""
    x = torch.linspace(-3, 3, 1000, device='cuda')

    # 1. Exact (torch.nn.functional.gelu)
    # 2. Tanh approximation
    # 3. Fast math (__tanf)

    # TODO: Measure accuracy (max error)
    # TODO: Measure performance (time per 1M elements)
    # TODO: Plot error distribution

# Expected:
# Exact:  0.85 ms, 0.0 error (baseline)
# Approx: 0.32 ms, 1.2e-5 max error (2.7x faster)
# Fast:   0.12 ms, 9.1e-5 max error (7.1x faster) ← Best for ML
```

### Success Criteria

- [ ] **GELU forward**: Within 1e-4 of PyTorch GELU
- [ ] **GELU backward**: Gradients match PyTorch within 1e-4
- [ ] **Performance**: 5-7x faster than PyTorch GELU
- [ ] **SiLU**: Correct implementation and gradients
- [ ] **PyTorch integration**: Works with autograd

---

## Exercise 04: LayerNorm Implementation

**Difficulty**: Hard | **Time**: 6 hours

### Learning Objectives

- Implement Welford's online algorithm for single-pass statistics
- Fuse LayerNorm with residual connection
- Implement correct backward pass with parameter gradients
- Achieve near-cuDNN performance

### Part 1: Forward Pass with Welford's Algorithm (2 hours)

```cuda
// TODO: Create csrc/layernorm.cu

/**
 * LayerNorm: y = (x - mean) / std * gamma + beta
 *
 * Single-pass using Welford's online algorithm:
 * - Compute mean and variance in one pass
 * - Normalize and apply affine transform
 */
__global__ void layer_norm_forward_kernel(
    const float* __restrict__ input,      // [B, N]
    const float* __restrict__ gamma,      // [N]
    const float* __restrict__ beta,       // [N]
    float* __restrict__ output,           // [B, N]
    float* __restrict__ mean,             // [B] - save for backward
    float* __restrict__ inv_std,          // [B] - save for backward
    const int B, const int N,
    const float eps = 1e-5f
) {
    int batch = blockIdx.x;
    if (batch >= B) return;

    __shared__ float mean_shared;
    __shared__ float var_shared;

    // TODO: Welford's online algorithm for mean and variance
    // For each thread:
    //   - Accumulate mean and M2 (sum of squared differences)
    //   - count += 1
    //   - delta = x - mean
    //   - mean += delta / count
    //   - delta2 = x - mean
    //   - M2 += delta * delta2

    // TODO: Reduce across threads
    // TODO: Compute variance = M2 / N
    // TODO: Normalize: (x - mean) / sqrt(var + eps)
    // TODO: Apply affine: x * gamma + beta
}
```

### Part 2: Backward Pass (2 hours)

```cuda
/**
 * LayerNorm Backward
 *
 * Gradients:
 * ∂L/∂x = (∂L/∂y * γ / σ) - mean(∂L/∂y * γ / σ)
 *         - (x - μ) * mean(∂L/∂y * γ * (x - μ) / σ³)
 * ∂L/∂γ = sum(∂L/∂y * (x - μ) / σ)
 * ∂L/∂β = sum(∂L/∂y)
 */
__global__ void layer_norm_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    float* __restrict__ grad_input,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    const int B, const int N
) {
    // TODO: Implement backward pass
    // TODO: Use atomics for grad_gamma and grad_beta
    // TODO: Efficient reduction for intermediate sums
}
```

### Part 3: Fused LayerNorm + Residual (2 hours)

```cuda
/**
 * Fused: LayerNorm(x + residual)
 * Common pattern in transformers
 */
__global__ void layer_norm_residual_kernel(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ mean,
    float* __restrict__ inv_std,
    const int B, const int N,
    const float eps = 1e-5f
) {
    // TODO: Load and add residual in first pass
    // TODO: Continue with LayerNorm
}
```

### Success Criteria

- [ ] **Correctness**: Output matches torch.nn.LayerNorm within 1e-5
- [ ] **Backward pass**: All gradients match PyTorch within 1e-5
- [ ] **Performance**: Within 10% of PyTorch LayerNorm (~0.3ms for [128, 768])
- [ ] **Fusion speedup**: 1.5-2x faster than unfused version
- [ ] **Single pass**: Uses Welford's algorithm correctly

---

## Exercise 05: Flash Attention Basics

**Difficulty**: Expert | **Time**: 8 hours

### Learning Objectives

- Understand Flash Attention algorithm
- Implement tiled attention computation
- Use online softmax to avoid materializing attention matrix
- Achieve significant memory savings (10-100x)

### Background

Standard attention materializes N×N attention matrix:
```python
scores = Q @ K.T  # [B, H, N, N] - N² memory!
attn = softmax(scores, dim=-1)
output = attn @ V
```

For N=2048: 2048² × 4 bytes = 16 MB per head → 192 MB for 12 heads

**Flash Attention**: Process in tiles, never materialize full matrix.

### Part 1: Tiled Attention (4 hours)

```cuda
// TODO: Create csrc/flash_attention.cu

/**
 * Flash Attention: Memory-efficient attention
 *
 * Algorithm:
 * 1. Tile Q, K, V into blocks that fit in shared memory
 * 2. For each Q tile:
 *    - For each K tile:
 *      - Compute attention scores in shared memory
 *      - Update running max and sum (online softmax)
 *      - Multiply with V tile and accumulate
 * 3. Final normalization
 *
 * Memory: O(N) instead of O(N²)
 */
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,    // [B, H, N, D]
    const float* __restrict__ K,    // [B, H, N, D]
    const float* __restrict__ V,    // [B, H, N, D]
    float* __restrict__ O,          // [B, H, N, D]
    const int B, const int H, const int N, const int D,
    const float scale  // 1/sqrt(D)
) {
    const int TILE_SIZE = 64;  // Tune based on shared memory

    // TODO: Shared memory for Q, K, V tiles
    __shared__ float Q_tile[TILE_SIZE][64];  // Assume D=64
    __shared__ float K_tile[TILE_SIZE][64];
    __shared__ float V_tile[TILE_SIZE][64];

    // TODO: Load Q tile (done once per block)
    // TODO: Initialize output accumulator and online softmax state
    // TODO: Loop over K, V tiles
    //   - Load K tile
    //   - Compute scores: Q_tile @ K_tile.T
    //   - Update online softmax: new_max, renormalize
    //   - Load V tile
    //   - Accumulate: output += softmax(scores) @ V_tile
    // TODO: Final normalization and write output
}
```

### Part 2: Online Softmax Algorithm (2 hours)

```cuda
/**
 * Online softmax: Compute softmax incrementally
 *
 * Algorithm:
 * - Keep running max and sum
 * - For each new tile:
 *   - new_max = max(old_max, tile_max)
 *   - correction = exp(old_max - new_max)
 *   - old_sum *= correction
 *   - new_sum = old_sum + sum(exp(x - new_max))
 */
__device__ void online_softmax_update(
    float* scores,           // [TILE_SIZE]
    float& row_max,          // Running max
    float& row_sum,          // Running sum
    const int tile_size
) {
    // TODO: Find max in current tile
    // TODO: Update global max
    // TODO: Renormalize previous sum
    // TODO: Add new contributions
}
```

### Part 3: Benchmarking and Validation (2 hours)

```python
# TODO: Create tests/test_flash_attention.py

def test_flash_attention_memory():
    """Compare memory usage: standard vs flash"""
    B, H, N, D = 1, 12, 2048, 64

    # Standard attention
    Q = torch.randn(B, H, N, D, device='cuda')
    K = torch.randn(B, H, N, D, device='cuda')
    V = torch.randn(B, H, N, D, device='cuda')

    torch.cuda.reset_peak_memory_stats()
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    standard_memory = torch.cuda.max_memory_allocated()

    # Flash attention
    torch.cuda.reset_peak_memory_stats()
    output_flash = flash_attention(Q, K, V)
    flash_memory = torch.cuda.max_memory_allocated()

    print(f"Standard: {standard_memory / 1e6:.1f} MB")
    print(f"Flash:    {flash_memory / 1e6:.1f} MB")
    print(f"Speedup:  {standard_memory / flash_memory:.1f}x memory savings")

    # Expected: 192 MB → 8 MB (24x savings)

def test_flash_attention_performance():
    """Benchmark speed vs standard attention"""
    # TODO: Measure time for different sequence lengths
    # TODO: Plot: N vs Time
    # TODO: Show Flash Attention enables longer sequences
```

### Success Criteria

- [ ] **Correctness**: Output matches standard attention within 1e-3
- [ ] **Memory**: 10-100x less memory than standard attention
- [ ] **Performance**: 2-5x faster than standard attention
- [ ] **Long sequences**: Handles N=8192+ without OOM
- [ ] **Online softmax**: Correctly implements incremental softmax

---

## Exercise 06: End-to-End Integration

**Difficulty**: Hard | **Time**: 5 hours

### Learning Objectives

- Build complete PyTorch C++ extension with multiple kernels
- Implement comprehensive testing suite
- Profile and optimize based on bottlenecks
- Create production-ready package

### Part 1: Complete Extension Package (2 hours)

Create a full PyTorch extension with all custom ops:

```
custom_ml_kernels/
├── csrc/
│   ├── fused_ops.cu
│   ├── reductions.cu
│   ├── activations.cu
│   ├── layernorm.cu
│   ├── flash_attention.cu
│   └── bindings.cpp
├── python/
│   ├── __init__.py
│   ├── ops.py
│   └── nn.py  # nn.Module wrappers
├── tests/
│   ├── test_correctness.py
│   ├── test_performance.py
│   └── test_gradients.py
├── benchmarks/
│   └── benchmark_suite.py
├── setup.py
└── README.md
```

### Part 2: Comprehensive Testing (2 hours)

```python
# TODO: Create tests/test_correctness.py

import torch
import pytest
import custom_ml_kernels as cmk

class TestCorrectness:
    """Test all ops match PyTorch native"""

    @pytest.mark.parametrize("size", [100, 1000, 10000, 100000])
    def test_gelu(self, size):
        x = torch.randn(size, device='cuda', requires_grad=True)

        # Forward
        y_torch = torch.nn.functional.gelu(x)
        y_custom = cmk.gelu(x)
        assert torch.allclose(y_torch, y_custom, atol=1e-4)

        # Backward
        grad = torch.randn_like(x)
        y_torch.backward(grad)
        grad_torch = x.grad.clone()

        x.grad = None
        y_custom.backward(grad)
        grad_custom = x.grad.clone()

        assert torch.allclose(grad_torch, grad_custom, atol=1e-4)

    def test_layernorm(self):
        # TODO: Test LayerNorm correctness
        pass

    def test_flash_attention(self):
        # TODO: Test Flash Attention correctness
        pass

# TODO: Create tests/test_performance.py
class TestPerformance:
    """Benchmark all ops vs PyTorch native"""

    def test_gelu_speedup(self):
        # TODO: Measure speedup
        # Target: 5-7x faster
        pass

    def test_layernorm_speedup(self):
        # TODO: Measure speedup
        # Target: 3-4x faster (with fusion)
        pass
```

### Part 3: Production Model Integration (1 hour)

Build a simple transformer block using custom kernels:

```python
# TODO: Create python/nn.py

import torch.nn as nn
import custom_ml_kernels as cmk

class CustomTransformerBlock(nn.Module):
    """Transformer block using custom kernels"""

    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.ln1 = cmk.nn.LayerNorm(d_model)  # Custom LayerNorm
        self.attn = cmk.nn.FlashAttention(d_model, n_heads)  # Flash Attention
        self.ln2 = cmk.nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            cmk.nn.GELU(),  # Custom GELU
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        # Attention block with fused residual
        x = cmk.ops.layer_norm_residual(self.ln1, x, x)

        # MLP block with fused residual
        mlp_out = self.mlp(x)
        x = cmk.ops.layer_norm_residual(self.ln2, mlp_out, x)

        return x

# TODO: Benchmark vs standard transformer
def benchmark_transformer():
    custom_block = CustomTransformerBlock().cuda()
    standard_block = StandardTransformerBlock().cuda()

    # TODO: Measure end-to-end speedup
    # Expected: 2-3x faster with custom kernels
```

### Success Criteria

- [ ] **Complete package**: All ops implemented and integrated
- [ ] **Tests pass**: 100% pass rate on correctness tests
- [ ] **Performance targets**: All ops meet speedup goals
- [ ] **Gradient checking**: torch.autograd.gradcheck passes
- [ ] **Documentation**: README with usage examples
- [ ] **Production ready**: Can be pip installed and used in models

---

## Submission Guidelines

### Required Deliverables

For each exercise:
1. **Source code**: Complete `.cu`, `.cpp`, `.py` files
2. **Build system**: Working `setup.py`
3. **Tests**: Passing correctness and performance tests
4. **Benchmarks**: Detailed performance comparisons
5. **Report**: Written analysis with:
   - Implementation approach
   - Performance results
   - Optimization techniques used
   - Lessons learned

### Code Quality Standards

```python
# Type hints
def fused_ops(input: torch.Tensor, bias: torch.Tensor, scale: float) -> torch.Tensor:
    """Docstring with performance characteristics"""
    pass

# Error handling
assert input.is_cuda, "Input must be CUDA tensor"
assert input.is_contiguous(), "Input must be contiguous"

# CUDA error checking
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}
```

### Evaluation Criteria

1. **Correctness** (30%): Passes all tests, matches PyTorch
2. **Performance** (30%): Meets speedup targets
3. **Code Quality** (20%): Clean, documented, type-hinted
4. **Analysis** (20%): Thorough written report with insights

## Resources

### Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch C++ Extension Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)

### Tools
- NVIDIA Nsight Compute: Kernel profiling
- NVIDIA Nsight Systems: System-wide profiling
- PyTorch profiler: Integration with Python code

### Example Code
- [Flash Attention Implementation](https://github.com/Dao-AILab/flash-attention)
- [Triton Tutorials](https://github.com/openai/triton)
- [PyTorch Extensions Examples](https://github.com/pytorch/extension-cpp)

---

**Ready to build production-grade CUDA kernels! These skills are essential for ML performance engineering.**
