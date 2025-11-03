# Module 2: CUDA Programming for Machine Learning

## Overview

This module teaches you to write custom CUDA kernels for machine learning operations. You'll build production-ready implementations of fused operations, activation functions, normalization layers, and attention mechanisms, then integrate them with PyTorch for use in real models.

**Duration**: 30 hours
**Difficulty**: Advanced
**Prerequisites**: Module 1 (Performance Fundamentals), C++ programming, PyTorch basics

## Learning Objectives

By completing this module, you will be able to:

1. **Write custom CUDA kernels** for ML operations from scratch
2. **Implement kernel fusion** to eliminate memory bottlenecks (3-5x speedup)
3. **Use warp-level primitives** (__shfl intrinsics) for fast reductions
4. **Optimize activation functions** (GELU, SiLU) with fast math
5. **Implement LayerNorm** with Welford's online algorithm
6. **Build Flash Attention** using tiling and online softmax
7. **Write backward passes** for PyTorch autograd integration
8. **Create PyTorch C++ extensions** for production deployment
9. **Profile and optimize** kernels with Nsight Compute
10. **Achieve 2-7x speedups** over PyTorch native operations

## Module Structure

### 1. Lecture Notes (11 hours)

Located in `lecture-notes/01-custom-cuda-kernels.md`

**Topics Covered**:
- Why write custom CUDA kernels? (Fusion, novel ops, performance)
- CUDA kernel development workflow
- Elementwise operations (ReLU, GELU, fusion patterns)
- Reduction operations (sum, mean, warp primitives)
- Matrix operations (tiled GEMM, fused GEMM+bias+ReLU)
- Activation functions (GELU case study: 7x speedup)
- Normalization layers (LayerNorm, Welford's algorithm)
- Attention mechanisms (Flash Attention: 4.7x faster, 24x less memory)
- Custom gradient kernels for training
- PyTorch C++ extension integration
- Performance optimization patterns

**Key Concepts**:
- **Kernel fusion**: Combine ops to reduce memory transfers (3-4x speedup)
- **Warp intrinsics**: __shfl_down_sync for fast reductions (32x faster than shared memory)
- **Vectorized loads**: float4 for 4x memory throughput
- **Online algorithms**: Welford for mean/var, online softmax for Flash Attention
- **Fast math**: __tanf, rsqrtf for approximate transcendentals (7x faster GELU)
- **Recomputation vs storage**: Trade compute for memory (Flash Attention)

### 2. Hands-On Exercises (30 hours)

Located in `exercises/README.md`

Six comprehensive exercises building production-ready kernels:

| Exercise | Topic | Time | Target Speedup |
|----------|-------|------|----------------|
| 01 | Fused Elementwise Operations | 4h | 3-4x vs unfused |
| 02 | Optimized Reductions | 5h | 13x vs naive |
| 03 | Custom Activation Functions | 4h | 5-7x vs PyTorch |
| 04 | LayerNorm Implementation | 6h | 3-4x with fusion |
| 05 | Flash Attention Basics | 8h | 4-5x, 24x less memory |
| 06 | End-to-End Integration | 5h | 2-3x full transformer |

**Tools You'll Master**:
- PyTorch C++ Extension API
- NVIDIA Nsight Compute (kernel profiling)
- NVIDIA Nsight Systems (system profiling)
- PyTorch autograd (custom backward passes)
- cuRAND (dropout, random ops)

## Prerequisites

### Required Knowledge

From Module 1:
- GPU architecture (SMs, CUDA cores, Tensor Cores)
- CUDA programming model (threads, warps, blocks, grids)
- Memory hierarchy (registers, shared memory, global memory)
- Performance metrics (bandwidth, occupancy, arithmetic intensity)
- Roofline model

Additional:
- **C++ programming**: Templates, pointers, classes
- **PyTorch**: Tensors, autograd, nn.Module
- **Linear algebra**: Matrix operations, attention mechanism
- **Calculus**: Gradients, chain rule

### Required Hardware & Software

Same as Module 1:
- **GPU**: NVIDIA GPU with compute capability 7.0+ (Volta or newer)
  - Recommended: A100, V100, RTX 3090, RTX 4090
- **System**: 64 GB RAM, Ubuntu 20.04+
- **CUDA Toolkit**: 12.0+
- **PyTorch**: 2.1+ with CUDA support

```bash
# Install PyTorch C++ extension support
pip install ninja  # Fast build system

# Verify extension build capability
python -c "from torch.utils.cpp_extension import BuildExtension; print('Ready')"
```

## Getting Started

### 1. Review Prerequisites

Make sure you completed Module 1 and understand:
- [ ] GPU memory hierarchy
- [ ] CUDA thread hierarchy
- [ ] Memory coalescing
- [ ] Warp divergence
- [ ] Shared memory usage
- [ ] Occupancy calculation

### 2. Study Lecture Notes

The lecture notes provide comprehensive coverage of custom kernel development:

```bash
cd lecture-notes
# Read 01-custom-cuda-kernels.md (~3 hours first pass)
```

**Study approach**:
- **Section 1-3** (3 hours): Introduction, workflow, elementwise ops
- **Section 4-6** (4 hours): Reductions, matrix ops, activations
- **Section 7-9** (3 hours): Normalization, attention, gradients
- **Section 10-12** (1 hour): Integration, optimization, best practices

**Key sections**:
- **Elementwise Ops** (Section 3): Foundation for all custom ops
- **Reductions** (Section 4): Essential for normalization, attention
- **Kernel Fusion** (Section 3.2): #1 optimization technique
- **Flash Attention** (Section 8): State-of-the-art attention implementation
- **PyTorch Integration** (Section 10): Deploy kernels in production

### 3. Complete Exercises Progressively

Start with simpler exercises and build up to Flash Attention:

```bash
cd exercises

# Exercise 1: Learn kernel fusion (4 hours)
cd exercise-01-fused-ops
# Implement fused ReLU + Scale + BiasAdd
# Target: 3-4x speedup, >80% bandwidth utilization

# Exercise 2: Master warp primitives (5 hours)
cd ../exercise-02-reductions
# Implement warp-optimized reduction
# Target: 13x speedup vs naive, within 20% of PyTorch

# Continue through all 6 exercises...
```

### 4. Build Complete Extension Package

Exercise 6 ties everything together into a production package:

```python
import custom_ml_kernels as cmk

# Use custom ops
x = torch.randn(1000, device='cuda')
y = cmk.gelu(x)  # 7x faster than torch.nn.functional.gelu

# Use in models
class FastTransformer(nn.Module):
    def __init__(self):
        self.ln = cmk.nn.LayerNorm(768)
        self.attn = cmk.nn.FlashAttention(768, 12)
        self.gelu = cmk.nn.GELU()
```

## Key Takeaways

### 1. Kernel Fusion is King

**Problem**: Multiple small ops → excessive memory traffic

```python
# Unfused: 5 kernel launches, 10 memory transfers
x = torch.relu(x)
x = x * 0.5
x = x + bias
x = torch.dropout(x, 0.1)

# Time: 0.185 ms
# Bandwidth: 30% of peak
```

**Solution**: Fuse into single kernel

```cuda
__global__ void fused_kernel(input, bias, output, mask, scale, dropout_prob) {
    // Load once
    float val = input[idx];

    // All operations in registers
    val = fmaxf(0.0f, val);              // ReLU
    val = val * scale;                    // Scale
    val = val + bias[idx];                // BiasAdd
    val = dropout(val, dropout_prob);     // Dropout

    // Store once
    output[idx] = val;
}

// Time: 0.038 ms (4.9x faster!)
// Bandwidth: 95% of peak
```

### 2. Warp Primitives for Reductions

**Old way**: Shared memory with __syncthreads (slow, divergent)

```cuda
// Naive tree reduction (has warp divergence)
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();  // Expensive!
}
```

**New way**: Warp shuffle intrinsics (32x faster)

```cuda
// Warp-level reduction (no __syncthreads, no divergence)
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // Lane 0 has sum
}
// 32x faster than shared memory for intra-warp communication!
```

### 3. Fast Math for Activations

GELU is 7x slower than ReLU. Use fast approximations:

```cuda
// Exact GELU: 0.85 ms
__device__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
}

// Fast GELU: 0.12 ms (7.1x faster!)
__device__ float gelu_fast(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + __tanf(0.7978845608f * (x + 0.044715f * x3)));
}
// Max error: 9.1e-5 (acceptable for ML!)
```

### 4. Welford's Algorithm for LayerNorm

Single-pass mean and variance calculation:

```cuda
// Standard: 3 passes (compute mean, compute var, normalize)
// Welford: 1 pass (online mean and M2 accumulation)

float mean = 0.0f;
float m2 = 0.0f;
int count = 0;

for (int i = 0; i < N; i++) {
    count++;
    float delta = x[i] - mean;
    mean += delta / count;
    float delta2 = x[i] - mean;
    m2 += delta * delta2;
}

float variance = m2 / N;
// 3-4x faster than 3-pass approach!
```

### 5. Flash Attention for Memory Efficiency

**Standard attention**: Materializes N×N matrix (OOM for long sequences)

```python
scores = Q @ K.T  # [B, H, N, N] - 16 MB per head for N=2048
attn = softmax(scores)
output = attn @ V

# N=2048: 192 MB (12 heads)
# N=8192: OOM!
```

**Flash Attention**: Tiles + online softmax (10-100x less memory)

```cuda
// Process in 64×64 tiles
// Never materialize full N×N matrix
// Update running max and sum for softmax

// N=2048: 8 MB (24x savings!)
// N=8192: 32 MB (enables long context!)
```

### 6. PyTorch Integration Pattern

```python
# 1. Define custom autograd Function
class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return relu_cuda_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return relu_cuda_backward(input, grad_output)

# 2. Wrap in Python function
def custom_relu(input):
    return CustomReLU.apply(input)

# 3. Use in models
x = custom_relu(x)  # Drop-in replacement for torch.relu
```

## Performance Benchmarks

After completing this module, your custom kernels should achieve:

### Activation Functions
```
GELU:     0.12 ms vs 0.85 ms PyTorch (7.1x faster)
SiLU:     0.10 ms vs 0.42 ms PyTorch (4.2x faster)
ReLU:     0.011 ms vs 0.045 ms PyTorch (4.1x faster)
```

### Normalization
```
LayerNorm:              0.31 ms vs 0.28 ms PyTorch (90% of PyTorch)
LayerNorm + Residual:   0.31 ms vs 1.20 ms unfused (3.9x faster)
```

### Attention
```
Standard Attention (N=2048):  4.50 ms, 192 MB
Flash Attention (N=2048):     0.95 ms, 8 MB (4.7x faster, 24x less memory)
Flash Attention (N=8192):     6.2 ms, 32 MB (enables long context)
```

### Fused Operations
```
ReLU + Scale + BiasAdd:       0.048 ms vs 0.185 ms unfused (3.9x faster)
ReLU + Scale + Bias + Dropout: 0.055 ms vs 0.220 ms unfused (4.0x faster)
```

### Full Transformer Block
```
Standard Block:  12.5 ms
Custom Kernels:   5.2 ms (2.4x faster)

Breakdown:
- LayerNorm (fused): 0.31 ms (was 1.20 ms)
- Flash Attention:   0.95 ms (was 4.50 ms)
- GELU:              0.12 ms (was 0.85 ms)
- Other ops:         3.82 ms (unchanged)
```

## Common Pitfalls and Solutions

### 1. Forgetting to Check CUDA Errors

**Problem**: Silent failures, incorrect results

```cuda
// BAD: No error checking
kernel<<<grid, block>>>(args);
cudaMemcpy(...);
```

**Solution**: Always check errors

```cuda
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

kernel<<<grid, block>>>(args);
CHECK_CUDA(cudaGetLastError());
CHECK_CUDA(cudaDeviceSynchronize());
```

### 2. Incorrect Gradient Implementation

**Problem**: Training fails, loss doesn't decrease

```python
# Check gradients with PyTorch's gradient checker
input = torch.randn(100, requires_grad=True, device='cuda', dtype=torch.double)
assert torch.autograd.gradcheck(custom_op, input, eps=1e-6, atol=1e-4)
```

### 3. Memory Alignment Issues

**Problem**: Unaligned accesses → poor performance

```cuda
// BAD: Unaligned float4 load
if (idx * 4 < n) {  // idx=1 → trying to load at offset 4 (unaligned!)
    float4 vals = reinterpret_cast<const float4*>(input)[idx];
}

// GOOD: Ensure alignment
assert(reinterpret_cast<uintptr_t>(input) % 16 == 0);  // Check in host code
```

### 4. Atomics Performance Issues

**Problem**: Too many atomic operations → serialization

```cuda
// BAD: One atomic per thread
atomicAdd(&grad_gamma[i], val);  // 1000s of atomics!

// BETTER: Reduce within block first
__shared__ float block_sum;
// ... reduce val across block ...
if (threadIdx.x == 0) {
    atomicAdd(&grad_gamma[i], block_sum);  // Only 1 atomic per block
}
```

### 5. Incorrect Launch Configuration

**Problem**: Poor occupancy or excessive overhead

```python
# BAD: Too many blocks
blocks = n  # Could be millions!

# GOOD: Limit blocks, use grid-stride loop
blocks = min((n + 255) / 256, 2048)  # ~2 blocks per SM
# Let each block process multiple elements
```

## Debugging Strategies

### 1. Start Simple

```cuda
// Step 1: Implement simplest version (no vectorization, no fusion)
__global__ void relu_v1(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Step 2: Add vectorization
__global__ void relu_v2(...) {
    // float4 loads
}

// Step 3: Fuse with other ops
__global__ void fused_relu_scale_bias(...) {
    // ...
}
```

### 2. Validate Against PyTorch

```python
# Always compare to PyTorch native
x = torch.randn(1000, device='cuda')

y_torch = torch.relu(x)
y_custom = custom_relu(x)

print(f"Max diff: {(y_torch - y_custom).abs().max()}")
assert torch.allclose(y_torch, y_custom, atol=1e-5)
```

### 3. Use Nsight Compute

```bash
# Profile kernel
ncu --set full -o profile python test.py

# Check specific metrics
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
              dram__bytes.sum,\
              smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
    python test.py

# Open GUI for detailed analysis
ncu-ui profile.ncu-rep
```

### 4. Print Debugging (Last Resort)

```cuda
// Only for development, remove for production
if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("First element: %f\n", input[0]);
}
```

## Optimization Workflow

1. **Profile** → Identify bottleneck with Nsight Compute
2. **Understand** → Is it memory-bound or compute-bound?
3. **Prioritize** → Focus on hot spots (80/20 rule)
4. **Optimize** → Apply appropriate technique:
   - Memory-bound → Fusion, vectorization, coalescing
   - Compute-bound → Fast math, Tensor Cores, better algorithm
5. **Validate** → Ensure correctness maintained
6. **Benchmark** → Measure speedup
7. **Iterate** → Repeat until target met

## Next Steps

After completing Module 2, you're ready for:

### Module 3: Performance Profiling & Debugging
- Advanced Nsight Compute techniques
- Kernel optimization case studies
- Performance regression detection
- Production profiling strategies

### Module 4: Transformer Optimization
- Flash Attention v2 (8-9x faster than standard)
- Fused attention + FFN + residual
- Continuous batching for inference
- KV cache optimization
- Long context optimization (32K+ tokens)

### Module 5: Model Compression
- Quantization kernels (INT8, FP8)
- Pruning and sparsity
- Knowledge distillation
- Distillation-aware training

## Resources

### Official Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch C++ Extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/)

### Research Papers
- **Flash Attention**: [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- **Flash Attention v2**: [arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
- **Welford's Algorithm**: [Technometrics 1962](https://www.jstor.org/stable/1266577)

### Example Implementations
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Official implementation
- [xFormers](https://github.com/facebookresearch/xformers) - Memory-efficient attention ops
- [Apex](https://github.com/NVIDIA/apex) - PyTorch extensions from NVIDIA
- [Triton](https://github.com/openai/triton) - Python-based GPU programming

### Tools
- **Nsight Compute**: Kernel-level profiling
- **Nsight Systems**: System-wide timeline
- **PyTorch Profiler**: Python-integrated profiling
- **cuda-gdb**: CUDA debugger
- **compute-sanitizer**: Memory error detection

## Assessment

To pass this module, you must:

1. **Complete all 6 exercises** (70%+ score each)
2. **Achieve performance targets** specified in exercises
3. **Pass gradient checks** for all autograd functions
4. **Submit working PyTorch extension** (Exercise 6)
5. **Pass module quiz** (20 questions, 80% required)

### Sample Quiz Questions

1. Why does kernel fusion improve performance? Calculate speedup for fusing 3 ops.
2. Explain how __shfl_down_sync works. Why is it faster than shared memory?
3. Implement online softmax update logic for Flash Attention.
4. What is Welford's algorithm? Derive the update equations.
5. Calculate memory savings: Standard vs Flash Attention for N=4096, 16 heads.
6. When should you use fast math intrinsics? What's the accuracy trade-off?
7. Explain the backward pass for LayerNorm. Why is it complex?
8. How do you integrate a custom CUDA kernel with PyTorch autograd?

## Support

### Getting Help

1. **Review lecture notes** - Comprehensive coverage with code examples
2. **Check exercise README** - Detailed implementation guides
3. **Search NVIDIA docs** - Official CUDA documentation
4. **Post in forum** - Community and instructor support
5. **Office hours** - [Schedule TBD]

### Reporting Issues

Found a bug? Performance issue? Please report:
- GitHub Issues: [repository URL]
- Email: [instructor email]
- Include: Module number, exercise, error message, GPU model

---

## Summary

Module 2 teaches you to write production-ready custom CUDA kernels for ML operations. You've learned kernel fusion, warp primitives, fast math optimizations, and PyTorch integration.

**Key Skills Acquired**:
- ✅ Custom CUDA kernel development
- ✅ Kernel fusion (3-5x speedups)
- ✅ Warp-level programming
- ✅ Fast math optimizations
- ✅ LayerNorm implementation
- ✅ Flash Attention basics
- ✅ PyTorch C++ extensions
- ✅ Gradient kernel implementation
- ✅ Performance profiling

**Performance Impact**:
- 3-7x faster activation functions
- 3-4x faster normalization (with fusion)
- 4-5x faster attention + 24x less memory
- 2-3x faster full transformer blocks

**Time Investment**: 30 hours
**Difficulty**: Advanced
**Prerequisites**: Module 1, C++, PyTorch

**Ready for production?** Your custom kernels can now be deployed in real ML systems for significant speedups.

---

*Last Updated: 2025-11-02*
*Module Version: 1.0*
*Feedback: [contact information]*
