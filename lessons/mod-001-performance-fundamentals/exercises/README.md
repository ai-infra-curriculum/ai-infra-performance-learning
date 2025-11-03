# Module 1: Performance Fundamentals - Exercises

This directory contains 6 comprehensive hands-on exercises designed to build practical skills in GPU architecture understanding, CUDA programming, and performance optimization fundamentals.

## Prerequisites

- NVIDIA GPU with CUDA support (compute capability 7.0+)
- CUDA Toolkit 12.0+ installed
- Python 3.9+ with PyTorch 2.1+
- Basic understanding of Python and C/C++
- Familiarity with Linux command line

## Environment Setup

```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Install required Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib pandas triton
pip install pycuda

# Clone exercise materials
cd mod-001-performance-fundamentals/exercises
```

## Exercise Structure

Each exercise includes:
- **Learning Objectives**: What you'll learn
- **Setup Instructions**: Environment and dependencies
- **Implementation Guide**: Step-by-step instructions
- **Success Criteria**: Performance targets and validation
- **Testing Procedures**: How to verify your implementation
- **Stretch Goals**: Advanced challenges

## Exercises Overview

| Exercise | Topic | Difficulty | Time | Key Skills |
|----------|-------|------------|------|------------|
| 01 | GPU Architecture Analysis | Easy | 2h | Hardware specs, theoretical performance |
| 02 | Memory Bandwidth Measurement | Medium | 3h | Memory transfers, coalescing |
| 03 | CUDA Thread Hierarchy | Medium | 3h | Grid/block configuration, parallel algorithms |
| 04 | Shared Memory Optimization | Hard | 4h | Tiled algorithms, memory hierarchies |
| 05 | Roofline Analysis | Hard | 4h | Performance modeling, profiling |
| 06 | Occupancy Tuning | Expert | 4h | Resource management, register optimization |

**Total Time**: ~20 hours

---

## Exercise 01: GPU Architecture Analysis

**Difficulty**: Easy | **Time**: 2 hours

### Learning Objectives

By completing this exercise, you will:
- Analyze GPU specifications to calculate theoretical performance
- Understand the relationship between hardware specs and performance capabilities
- Calculate memory bandwidth, compute throughput, and arithmetic intensity requirements
- Compare different GPU architectures (A100 vs H100)

### Setup Instructions

```bash
cd exercise-01-gpu-architecture
pip install pandas matplotlib

# Query your GPU
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

### Part 1: Theoretical Performance Calculation (30 min)

**Task**: Calculate theoretical peak performance for NVIDIA A100 GPU.

Given specifications:
- 108 Streaming Multiprocessors (SMs)
- 64 FP32 CUDA cores per SM
- 4 Tensor Cores per SM
- Boost clock: 1.41 GHz
- Memory bandwidth: 2 TB/s (2048 GB/s)
- HBM2e memory: 80 GB

**Calculate**:
1. **Peak FP32 performance** (TFLOPS)
   ```
   FP32_TFLOPS = SMs × CUDA_cores_per_SM × clock_GHz × 2 (FMA)
   ```

2. **Peak FP16 performance with Tensor Cores** (TFLOPS)
   ```
   # Tensor Core does 4×4 matrix multiply-accumulate
   # Each Tensor Core: 512 FP16 FMA ops per cycle
   FP16_TC_TFLOPS = SMs × TC_per_SM × ops_per_TC × clock_GHz
   ```

3. **Memory bandwidth** (GB/s and TB/s)

4. **Theoretical ridge point** (FLOPs/byte)
   ```
   Ridge_Point = Peak_Compute_TFLOPS / Peak_Bandwidth_TB/s
   ```

**Implementation**:

```python
# TODO: Create a Python script `gpu_specs.py`
# - Define GPU specifications as dataclass or dictionary
# - Implement calculation functions
# - Generate comparison table for A100 vs H100
# - Visualize results with matplotlib

import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class GPUSpecs:
    name: str
    sms: int
    cuda_cores_per_sm: int
    tensor_cores_per_sm: int
    clock_ghz: float
    memory_bandwidth_gbps: float
    memory_size_gb: int

    def calculate_fp32_tflops(self) -> float:
        """Calculate peak FP32 performance in TFLOPS"""
        # TODO: Implement calculation
        pass

    def calculate_fp16_tensor_tflops(self) -> float:
        """Calculate peak FP16 performance with Tensor Cores"""
        # TODO: Implement calculation
        pass

    def calculate_ridge_point(self) -> float:
        """Calculate arithmetic intensity ridge point (FLOPs/byte)"""
        # TODO: Implement calculation
        pass

# TODO: Create specs for A100 and H100
# TODO: Generate comparison DataFrame
# TODO: Create bar charts comparing performance
```

### Part 2: Memory Hierarchy Analysis (45 min)

**Task**: Analyze memory hierarchy latencies and calculate effective bandwidth.

Given A100 memory specifications:
- Registers: 256 KB per SM, 1 cycle latency
- L1 Cache: 128 KB per SM, ~28 cycles latency
- Shared Memory: 164 KB per SM, ~28 cycles latency
- L2 Cache: 40 MB total, ~200 cycles latency
- HBM2e: 80 GB, ~350 cycles latency
- PCIe 4.0: 64 GB/s, ~20,000 cycles latency

**Calculate**:
1. **Effective bandwidth** for each memory level (TB/s)
   ```
   Effective_BW = (bytes_per_access / latency_cycles) × clock_frequency
   ```

2. **Capacity per SM** for shared resources

3. **Memory hierarchy diagram** showing relative speeds

**Implementation**:

```python
# TODO: Create `memory_hierarchy.py`
# - Define memory level specifications
# - Calculate effective bandwidth for each level
# - Generate waterfall chart showing latency differences
# - Calculate breakeven points for data reuse

def calculate_effective_bandwidth(bytes_per_access: int,
                                   latency_cycles: int,
                                   clock_ghz: float) -> float:
    """Calculate effective bandwidth in GB/s"""
    # TODO: Implement
    pass

def memory_hierarchy_chart():
    """Create visualization of memory hierarchy"""
    # TODO: Use matplotlib to create:
    # - Latency waterfall chart
    # - Bandwidth comparison bars
    # - Capacity pyramid
    pass
```

### Part 3: Architecture Comparison (45 min)

**Task**: Compare A100 (Ampere) and H100 (Hopper) architectures.

Research and compare:
- SM count and structure
- CUDA core and Tensor Core counts
- Memory bandwidth and capacity
- New features (H100: Transformer Engine, FP8, etc.)
- Performance improvements

**Deliverable**: Create comparison table and report.

```markdown
# TODO: Create `ARCHITECTURE_COMPARISON.md`
## Ampere (A100) vs Hopper (H100)

### Architectural Changes
- [Compare SM architecture]
- [Discuss Tensor Core improvements]
- [Explain new features: FP8, Transformer Engine]

### Performance Improvements
- [Quantify compute improvements]
- [Analyze memory bandwidth gains]
- [Calculate expected speedup for different workloads]

### Use Case Recommendations
- [When to choose A100 vs H100]
- [Cost-performance trade-offs]
```

### Success Criteria

- [ ] **Accurate calculations**: FP32, FP16, memory bandwidth within 5% of published specs
- [ ] **Code quality**: Type hints, docstrings, error handling
- [ ] **Visualizations**: Clear charts comparing A100 vs H100
- [ ] **Memory hierarchy**: Correct latency and bandwidth calculations
- [ ] **Documentation**: Complete comparison report with analysis

### Testing Procedure

```bash
# Run your implementation
python gpu_specs.py
python memory_hierarchy.py

# Verify outputs
# - Check console output for calculated values
# - Inspect generated PNG charts
# - Review ARCHITECTURE_COMPARISON.md
```

### Expected Output

```
=== NVIDIA A100 Specifications ===
FP32 Performance: 19.5 TFLOPS
FP16 (Tensor Core): 312 TFLOPS
Memory Bandwidth: 2048 GB/s (2.0 TB/s)
Ridge Point: 156 FLOPs/byte

=== Memory Hierarchy ===
Level          | Capacity  | Latency (cycles) | Effective BW
---------------|-----------|------------------|-------------
Registers      | 256 KB/SM | 1                | 350 TB/s
Shared Memory  | 164 KB/SM | 28               | 12.5 TB/s
L2 Cache       | 40 MB     | 200              | 1.75 TB/s
HBM2e          | 80 GB     | 350              | 2.0 TB/s
```

### Stretch Goals

1. **Multi-GPU comparison**: Add V100, A100, H100, and GH200 comparisons
2. **TCO analysis**: Calculate cost per TFLOPS for different GPUs
3. **Workload profiling**: Estimate performance for specific workloads (LLM training, inference, etc.)
4. **Interactive dashboard**: Create Streamlit app for exploring GPU specs

---

## Exercise 02: Memory Bandwidth Measurement

**Difficulty**: Medium | **Time**: 3 hours

### Learning Objectives

- Measure actual vs theoretical memory bandwidth
- Understand memory coalescing and its impact on performance
- Implement memory transfer benchmarks with different patterns
- Analyze PCIe vs GPU memory bandwidth

### Setup Instructions

```bash
cd exercise-02-memory-bandwidth
pip install pycuda numpy matplotlib
```

### Part 1: Simple Memory Copy (45 min)

**Task**: Implement and benchmark device-to-device memory copy.

```python
# TODO: Create `memory_copy.py`
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

def benchmark_memcpy(size_mb: int, num_iterations: int = 100):
    """
    Benchmark device-to-device memory copy

    Args:
        size_mb: Size of data in MB
        num_iterations: Number of iterations for averaging

    Returns:
        bandwidth_gbps: Achieved bandwidth in GB/s
    """
    # TODO: Allocate device memory
    # TODO: Perform warmup
    # TODO: Time memory copies
    # TODO: Calculate bandwidth
    pass

# TODO: Benchmark different sizes: 1MB, 10MB, 100MB, 1GB
# TODO: Plot bandwidth vs transfer size
# TODO: Compare to theoretical peak (2048 GB/s for A100)
```

### Part 2: Memory Coalescing Benchmark (1 hour)

**Task**: Demonstrate the impact of coalesced vs strided memory access.

Implement three access patterns:
1. **Coalesced**: Thread i accesses element i (consecutive)
2. **Strided**: Thread i accesses element i * stride (gaps)
3. **Random**: Thread i accesses random element

```cuda
// TODO: Create `coalescing.cu`
// Coalesced access pattern
__global__ void coalesced_read(float *data, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = data[idx];  // Sequential access
    }
}

// Strided access pattern
__global__ void strided_read(float *data, float *output, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * stride < N) {
        output[idx] = data[idx * stride];  // Strided access
    }
}

// Random access pattern
__global__ void random_read(float *data, float *output, int *indices, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = data[indices[idx]];  // Random access
    }
}
```

**Python wrapper**:

```python
# TODO: Create `coalescing_benchmark.py`
# - Compile CUDA kernels using pycuda
# - Run benchmarks with different patterns
# - Measure bandwidth for each pattern
# - Calculate slowdown vs coalesced access

def benchmark_access_pattern(pattern: str, size: int, stride: int = 1):
    """Benchmark memory access pattern"""
    # TODO: Implement
    pass

# TODO: Test strides: 1, 2, 4, 8, 16, 32
# TODO: Generate chart showing bandwidth degradation
```

### Part 3: PCIe Bandwidth Measurement (1 hour)

**Task**: Measure host-to-device and device-to-host transfer bandwidth.

```python
# TODO: Create `pcie_bandwidth.py`
def benchmark_host_to_device(size_mb: int):
    """Benchmark H2D transfer"""
    # TODO: Allocate pinned host memory
    # TODO: Allocate device memory
    # TODO: Time cudaMemcpy H2D
    # TODO: Calculate bandwidth
    pass

def benchmark_device_to_host(size_mb: int):
    """Benchmark D2H transfer"""
    # TODO: Similar to H2D
    pass

def benchmark_pinned_vs_pageable(size_mb: int):
    """Compare pinned vs pageable host memory"""
    # TODO: Test both memory types
    # TODO: Show pinned memory advantage
    pass

# TODO: Test transfer sizes: 1KB to 1GB
# TODO: Compare to PCIe 4.0 x16 theoretical: 64 GB/s
# TODO: Analyze transfer size vs bandwidth relationship
```

### Success Criteria

- [ ] **Memory copy bandwidth**: Achieves >90% of theoretical peak (1843+ GB/s for A100)
- [ ] **Coalescing impact**: Demonstrates 10-32x slowdown for strided access
- [ ] **PCIe bandwidth**: Achieves >50 GB/s for large transfers
- [ ] **Pinned memory**: Shows 2-3x speedup over pageable memory
- [ ] **Visualizations**: Clear charts for all benchmarks
- [ ] **Analysis**: Written report explaining results

### Testing Procedure

```bash
# Compile CUDA code
nvcc -o coalescing coalescing.cu

# Run benchmarks
python memory_copy.py
python coalescing_benchmark.py
python pcie_bandwidth.py

# Generate report
python generate_report.py  # Creates BANDWIDTH_REPORT.md
```

### Expected Results

```
=== Device Memory Copy ===
Transfer Size | Bandwidth
1 MB          | 1200 GB/s
10 MB         | 1800 GB/s
100 MB        | 1950 GB/s
1 GB          | 2000 GB/s (98% of peak)

=== Access Pattern Performance ===
Pattern       | Bandwidth | Slowdown
Coalesced     | 1950 GB/s | 1.0x
Stride 2      | 975 GB/s  | 2.0x
Stride 4      | 487 GB/s  | 4.0x
Stride 32     | 61 GB/s   | 32.0x (!)

=== PCIe Bandwidth ===
Direction | Pageable | Pinned
H2D       | 12 GB/s  | 52 GB/s
D2H       | 11 GB/s  | 55 GB/s
```

### Stretch Goals

1. **Async transfers**: Implement overlapping compute and transfer
2. **Multi-GPU**: Measure GPU-to-GPU transfer bandwidth (NVLink vs PCIe)
3. **Unified Memory**: Benchmark managed memory performance
4. **Cache effects**: Measure L1/L2 cache hit rates

---

## Exercise 03: CUDA Thread Hierarchy

**Difficulty**: Medium | **Time**: 3 hours

### Learning Objectives

- Master CUDA thread hierarchy (grid, block, warp, thread)
- Implement parallel algorithms with optimal configuration
- Understand warp divergence and its performance impact
- Optimize block and grid dimensions for different workloads

### Setup Instructions

```bash
cd exercise-03-thread-hierarchy
nvcc --version  # Verify CUDA compiler
```

### Part 1: Vector Operations (1 hour)

**Task**: Implement vector addition, scaling, and dot product with optimal launch configuration.

```cuda
// TODO: Create `vector_ops.cu`
// Vector addition: C = A + B
__global__ void vector_add(float *A, float *B, float *C, int N) {
    // TODO: Calculate global thread index
    // TODO: Bounds check
    // TODO: Perform addition
}

// Vector scaling: B = alpha * A
__global__ void vector_scale(float *A, float *B, float alpha, int N) {
    // TODO: Implement
}

// Dot product (partial): partial_sum = sum(A[i] * B[i])
__global__ void dot_product_partial(float *A, float *B, float *partial, int N) {
    // TODO: Each block computes partial sum
    // TODO: Use shared memory for block-level reduction
    // TODO: Write result to partial[blockIdx.x]
}
```

**Launch configuration analysis**:

```python
# TODO: Create `launch_config.py`
def calculate_optimal_config(N: int, max_threads_per_block: int = 1024):
    """
    Calculate optimal grid and block dimensions

    Returns:
        threads_per_block, blocks_per_grid
    """
    # TODO: Choose threads_per_block (multiple of 32, power of 2)
    # TODO: Calculate blocks_per_grid
    # TODO: Consider occupancy
    pass

# TODO: Test configurations: 32, 64, 128, 256, 512, 1024 threads/block
# TODO: Measure performance for each
# TODO: Identify optimal configuration
```

### Part 2: Warp Divergence Analysis (1 hour)

**Task**: Demonstrate warp divergence and its performance impact.

```cuda
// TODO: Create `warp_divergence.cu`
// No divergence: all threads take same path
__global__ void no_divergence(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f;  // All threads execute
    }
}

// Full divergence: odd/even threads take different paths
__global__ void full_divergence(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (idx % 2 == 0) {
            // Expensive computation (even threads)
            for (int i = 0; i < 100; i++) {
                data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
            }
        } else {
            // Simple computation (odd threads)
            data[idx] = data[idx] * 2.0f;
        }
        // ⚠️ Both paths execute serially!
    }
}

// Warp-aligned divergence: better than full divergence
__global__ void warp_aligned_divergence(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / 32;

    if (idx < N) {
        if (warp_id % 2 == 0) {
            // Even warps: expensive path
            for (int i = 0; i < 100; i++) {
                data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
            }
        } else {
            // Odd warps: simple path
            data[idx] = data[idx] * 2.0f;
        }
        // ✓ Warps remain coherent
    }
}
```

**Benchmark divergence**:

```python
# TODO: Create `divergence_benchmark.py`
# - Run all three kernels
# - Measure execution time
# - Calculate overhead from divergence
# - Expected: full_divergence ~2x slower than warp_aligned

def benchmark_divergence():
    """Compare divergence patterns"""
    # TODO: Implement
    pass
```

### Part 3: Parallel Reduction (1 hour)

**Task**: Implement efficient parallel reduction using shared memory.

```cuda
// TODO: Create `reduction.cu`
// Naive reduction: suffers from warp divergence
__global__ void reduce_naive(float *input, float *output, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // TODO: Implement tree reduction (naive)
    // Problem: Warp divergence at every step!
}

// Optimized reduction: warp-aware
__global__ void reduce_optimized(float *input, float *output, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load with coalesced access
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // TODO: Implement warp-aware reduction
    // - Reduce within warps first (no __syncthreads needed)
    // - Then reduce across warps
    // - Use __shfl_down_sync for warp-level reduction
}
```

**Warp shuffle implementation**:

```cuda
// TODO: Add to `reduction.cu`
__device__ float warp_reduce_sum(float val) {
    // TODO: Use __shfl_down_sync to reduce within warp
    // Loop: offset = 16, 8, 4, 2, 1
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

### Success Criteria

- [ ] **Vector ops**: Implement all three operations correctly
- [ ] **Optimal config**: Achieve >90% of theoretical performance
- [ ] **Divergence demo**: Show 2x slowdown for full divergence
- [ ] **Reduction**: Optimized version 3-5x faster than naive
- [ ] **Launch config analysis**: Chart showing performance vs block size
- [ ] **Code quality**: Proper error checking, timing code

### Testing Procedure

```bash
# Compile
nvcc -o vector_ops vector_ops.cu -O3
nvcc -o warp_divergence warp_divergence.cu -O3
nvcc -o reduction reduction.cu -O3

# Run benchmarks
python launch_config.py
python divergence_benchmark.py
./reduction

# Verify correctness
python test_kernels.py
```

### Expected Performance

```
=== Vector Addition (N=1M) ===
Block Size | Time (ms) | Bandwidth
32         | 0.150     | 160 GB/s
128        | 0.052     | 461 GB/s
256        | 0.042     | 571 GB/s ← Optimal
512        | 0.045     | 533 GB/s
1024       | 0.048     | 500 GB/s

=== Warp Divergence Impact ===
Kernel               | Time (ms) | Slowdown
No Divergence        | 10.2      | 1.0x
Full Divergence      | 21.8      | 2.1x
Warp-Aligned         | 11.5      | 1.1x

=== Reduction (N=1M) ===
Implementation | Time (μs) | Speedup
Naive          | 450       | 1.0x
Optimized      | 95        | 4.7x
```

### Stretch Goals

1. **2D/3D grids**: Implement matrix operations with 2D thread blocks
2. **Dynamic parallelism**: Launch child kernels from device
3. **Cooperative groups**: Use cooperative_groups API for flexible thread cooperation
4. **Multi-kernel pipeline**: Overlap multiple kernel launches

---

## Exercise 04: Shared Memory Optimization

**Difficulty**: Hard | **Time**: 4 hours

### Learning Objectives

- Implement tiled algorithms using shared memory
- Optimize matrix multiplication with shared memory blocking
- Avoid shared memory bank conflicts
- Balance shared memory usage with occupancy

### Setup Instructions

```bash
cd exercise-04-shared-memory
pip install torch numpy matplotlib
```

### Part 1: Tiled Matrix Multiplication (2 hours)

**Task**: Implement optimized matrix multiplication using shared memory tiling.

**Algorithm**: Compute C = A × B where A is M×K, B is K×N, C is M×N.

```cuda
// TODO: Create `matmul.cu`
#define TILE_SIZE 16

// Naive matmul: Global memory only
__global__ void matmul_naive(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
            // ⚠️ K memory accesses to global memory
        }
        C[row * N + col] = sum;
    }
}

// Tiled matmul: Shared memory optimization
__global__ void matmul_tiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // TODO: Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // TODO: Load tile of A into shared memory
        // TODO: Load tile of B into shared memory
        // TODO: Synchronize to ensure tile is loaded
        // TODO: Compute partial product using shared memory
        // TODO: Synchronize before loading next tile
    }

    // TODO: Write result to C
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Optimized matmul: Additional optimizations
__global__ void matmul_optimized(float *A, float *B, float *C, int M, int N, int K) {
    // TODO: Add bank conflict avoidance (padding)
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid conflicts
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // TODO: Add register tiling (compute multiple elements per thread)
    // TODO: Add vectorized loads (float4)
    // TODO: Implement similar to tiled version with optimizations
}
```

**Benchmark**:

```python
# TODO: Create `matmul_benchmark.py`
import torch
import numpy as np
from pycuda import compiler, gpuarray
import pycuda.autoinit

def benchmark_matmul(M, N, K):
    """Benchmark all three implementations"""
    # TODO: Generate random matrices
    # TODO: Run naive, tiled, optimized versions
    # TODO: Compare to PyTorch GEMM
    # TODO: Calculate achieved TFLOPS
    pass

# TODO: Test sizes: 512, 1024, 2048, 4096
# TODO: Compare speedup: naive → tiled → optimized → PyTorch
```

### Part 2: Bank Conflict Analysis (1 hour)

**Task**: Understand and avoid shared memory bank conflicts.

Background: Shared memory is divided into 32 banks. Simultaneous access to the same bank by different threads causes serialization.

```cuda
// TODO: Create `bank_conflicts.cu`
// No conflict: Each thread accesses different bank
__global__ void no_conflict(float *output) {
    __shared__ float sdata[32 * 32];

    int tid = threadIdx.x;
    int wid = tid / 32;  // Warp ID
    int lane = tid % 32;  // Lane within warp

    // Each lane accesses a different bank
    float value = sdata[wid * 32 + lane];  // ✓ No conflict
    output[tid] = value;
}

// 32-way conflict: All threads access same bank
__global__ void conflict_32way(float *output) {
    __shared__ float sdata[32 * 32];

    int tid = threadIdx.x;
    int lane = tid % 32;

    // All threads in warp access bank 0
    float value = sdata[lane * 32];  // ⚠️ 32-way conflict!
    output[tid] = value;
}

// 2-way conflict: Stride causes conflicts
__global__ void conflict_2way(float *output) {
    __shared__ float sdata[32 * 32];

    int tid = threadIdx.x;
    int lane = tid % 32;

    // Every other thread accesses same bank
    float value = sdata[lane * 2];  // ⚠️ 2-way conflict
    output[tid] = value;
}

// Avoiding conflict with padding
__global__ void avoid_conflict_padding(float *output) {
    // Add 1 extra column to shift banks
    __shared__ float sdata[32][33];  // 33 instead of 32

    int tid = threadIdx.x;
    int row = tid / 32;
    int col = tid % 32;

    float value = sdata[col][row];  // ✓ No conflict due to padding
    output[tid] = value;
}
```

**Measure bank conflicts**:

```bash
# TODO: Use nvprof or Nsight Compute
nvprof --metrics shared_load_transactions_per_request ./bank_conflicts

# Expected output:
# no_conflict: 1.0 (perfect coalescing)
# conflict_32way: 32.0 (maximum serialization)
# conflict_2way: 2.0
# avoid_conflict_padding: 1.0
```

### Part 3: Occupancy vs Shared Memory Trade-off (1 hour)

**Task**: Analyze the trade-off between shared memory usage and occupancy.

A100 specs:
- Max shared memory per SM: 164 KB
- Max threads per SM: 2048
- Max blocks per SM: 32

```python
# TODO: Create `occupancy_analysis.py`
def calculate_occupancy(threads_per_block: int,
                        shared_mem_per_block: int,
                        registers_per_thread: int = 64):
    """
    Calculate theoretical occupancy given resource usage

    Returns:
        occupancy: Percentage of max threads active
        blocks_per_sm: Number of blocks per SM
        limiting_factor: 'threads', 'shared_mem', or 'registers'
    """
    # TODO: Implement occupancy calculator
    # - Max blocks limited by threads: 2048 / threads_per_block
    # - Max blocks limited by shared mem: 168960 / shared_mem_per_block
    # - Max blocks limited by registers: ...
    # - Occupancy = min(limits) * threads_per_block / 2048
    pass

# TODO: Analyze different TILE_SIZE values
# TILE_SIZE 8:  8×8×4 bytes = 256 bytes shared mem
# TILE_SIZE 16: 16×16×4 bytes = 1024 bytes shared mem
# TILE_SIZE 32: 32×32×4 bytes = 4096 bytes shared mem
# TILE_SIZE 64: 64×64×4 bytes = 16384 bytes shared mem

# TODO: Find optimal TILE_SIZE balancing shared mem usage and occupancy
```

### Success Criteria

- [ ] **Matmul correctness**: Results match PyTorch within 1e-5
- [ ] **Tiled speedup**: 5-10x faster than naive implementation
- [ ] **Performance**: Achieve >50% of PyTorch performance
- [ ] **Bank conflicts**: Demonstrate and avoid them
- [ ] **Occupancy analysis**: Identify optimal TILE_SIZE
- [ ] **Documentation**: Detailed report with profiling results

### Testing Procedure

```bash
# Compile kernels
nvcc -o matmul matmul.cu -O3
nvcc -o bank_conflicts bank_conflicts.cu -O3

# Run benchmarks
python matmul_benchmark.py

# Profile with Nsight Compute
ncu --set full -o matmul_profile ./matmul
ncu --set full -o bank_conflicts_profile ./bank_conflicts

# Analyze occupancy
python occupancy_analysis.py
```

### Expected Performance

```
=== Matrix Multiplication (M=N=K=2048) ===
Implementation | Time (ms) | TFLOPS | Speedup | vs PyTorch
Naive          | 450.2     | 0.38   | 1.0x    | 1.2%
Tiled (16×16)  | 52.8      | 3.25   | 8.5x    | 10.2%
Optimized      | 28.1      | 6.11   | 16.0x   | 19.1%
PyTorch        | 5.4       | 31.8   | 83.4x   | 100%

=== Bank Conflicts ===
Kernel              | Transactions/Request | Slowdown
No Conflict         | 1.0                  | 1.0x
2-Way Conflict      | 2.0                  | 1.9x
32-Way Conflict     | 32.0                 | 28.7x
Padding (Avoided)   | 1.0                  | 1.0x

=== Occupancy Analysis ===
TILE_SIZE | Shared Mem | Blocks/SM | Occupancy | Performance
8         | 256 B      | 32        | 100%      | 2.1 TFLOPS
16        | 1 KB       | 32        | 100%      | 6.1 TFLOPS ← Best
32        | 4 KB       | 32        | 100%      | 5.8 TFLOPS
64        | 16 KB      | 10        | 62.5%     | 3.2 TFLOPS
```

### Stretch Goals

1. **Tensor Core matmul**: Use wmma API for FP16 Tensor Core operations
2. **Double buffering**: Overlap shared memory loads with computation
3. **Rectangular tiles**: Optimize for non-square matrices
4. **cuBLAS integration**: Compare to highly optimized NVIDIA library

---

## Exercise 05: Roofline Analysis

**Difficulty**: Hard | **Time**: 4 hours

### Learning Objectives

- Profile GPU kernels to measure performance
- Calculate arithmetic intensity for different algorithms
- Plot kernels on the Roofline model
- Identify performance bottlenecks (compute vs memory bound)
- Optimize based on Roofline analysis

### Setup Instructions

```bash
cd exercise-05-roofline
pip install matplotlib pandas
# Install NVIDIA Nsight Compute (included with CUDA Toolkit)
```

### Part 1: Roofline Model Construction (1 hour)

**Task**: Create the Roofline model for your GPU.

```python
# TODO: Create `roofline.py`
import numpy as np
import matplotlib.pyplot as plt

class RooflineModel:
    def __init__(self, peak_compute_tflops: float, peak_bandwidth_gbps: float):
        """
        Initialize Roofline model

        Args:
            peak_compute_tflops: Peak compute performance (TFLOPS)
            peak_bandwidth_gbps: Peak memory bandwidth (GB/s)
        """
        self.peak_compute = peak_compute_tflops * 1e12  # Convert to FLOPS
        self.peak_bandwidth = peak_bandwidth_gbps * 1e9  # Convert to bytes/s
        self.ridge_point = self.peak_compute / self.peak_bandwidth  # FLOPs/byte

    def theoretical_performance(self, arithmetic_intensity: float) -> float:
        """
        Calculate theoretical performance for given arithmetic intensity

        Returns:
            performance in FLOPS
        """
        # TODO: Implement roofline formula
        # If AI < ridge_point: memory bound (performance = AI × bandwidth)
        # If AI >= ridge_point: compute bound (performance = peak_compute)
        pass

    def plot_roofline(self, kernels: list = None):
        """
        Plot the Roofline model with optional kernel data points

        Args:
            kernels: List of (name, AI, achieved_flops) tuples
        """
        # TODO: Create log-log plot
        # X-axis: Arithmetic Intensity (FLOPs/byte), log scale
        # Y-axis: Performance (TFLOPS), log scale
        # Plot two regions: memory bound and compute bound
        # Mark ridge point
        # Add kernel data points if provided
        pass

# TODO: Create Roofline for A100
# Peak FP32: 19.5 TFLOPS
# Peak FP16 (Tensor Core): 312 TFLOPS
# Peak Bandwidth: 2048 GB/s
# Ridge Point FP32: 19.5/2.048 = 9.5 FLOPs/byte
# Ridge Point FP16: 312/2.048 = 152 FLOPs/byte
```

### Part 2: Kernel Profiling (2 hours)

**Task**: Profile various kernels and calculate their arithmetic intensity.

**Kernels to profile**:
1. Vector addition
2. Vector scaling (SAXPY)
3. Dot product
4. Matrix-vector multiply
5. Matrix-matrix multiply
6. Convolution

```python
# TODO: Create `profile_kernels.py`
import subprocess
import re

class KernelProfiler:
    def __init__(self, executable: str):
        self.executable = executable

    def profile_with_ncu(self, kernel_name: str) -> dict:
        """
        Profile kernel using NVIDIA Nsight Compute

        Returns:
            metrics: Dict with runtime, flops, bytes, achieved_occupancy
        """
        # TODO: Run ncu command
        cmd = [
            'ncu',
            '--metrics', 'dram__bytes.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,'
                        'sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__cycles_elapsed.avg',
            '--csv',
            self.executable,
            kernel_name
        ]

        # TODO: Parse ncu output
        # TODO: Extract metrics
        # TODO: Calculate arithmetic intensity = FLOPs / bytes_transferred
        pass

    def calculate_arithmetic_intensity(self, flops: int, bytes_transferred: int) -> float:
        """Calculate AI in FLOPs/byte"""
        return flops / bytes_transferred if bytes_transferred > 0 else 0

# TODO: Profile all kernels
# TODO: Create DataFrame with results
```

**Alternative manual calculation**:

```python
# TODO: Create `calculate_ai.py`
def vector_add_ai(N: int) -> float:
    """
    Vector addition: C = A + B

    FLOPs: N (1 add per element)
    Bytes: 3×N×4 (read A, B, write C; 4 bytes per float)
    AI = N / (3×N×4) = 1/12 = 0.083 FLOPs/byte
    """
    flops = N
    bytes_transferred = 3 * N * 4
    return flops / bytes_transferred

def matmul_ai(M: int, N: int, K: int) -> float:
    """
    Matrix multiply: C = A × B

    FLOPs: 2×M×N×K (multiply-add for each element)
    Bytes (naive): (M×K + K×N + M×N) × 4
    AI = 2×M×N×K / ((M×K + K×N + M×N) × 4)

    For square matrices (M=N=K):
    AI = 2×N³ / (3×N²×4) = N/6 FLOPs/byte
    """
    # TODO: Implement
    pass

# TODO: Calculate AI for all kernel types
# TODO: Compare to profiled values
```

### Part 3: Performance Optimization (1 hour)

**Task**: Optimize a memory-bound kernel based on Roofline analysis.

**Example: Vector addition is memory-bound (AI = 0.083 < ridge point)**

Strategy: Fuse operations to increase AI.

```cuda
// TODO: Create `fusion.cu`
// Memory-bound: Two separate kernels
__global__ void vector_add(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
    // AI = 1/12 = 0.083 FLOPs/byte
}

__global__ void vector_scale(float *C, float *D, float alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        D[idx] = alpha * C[idx];
    }
    // AI = 1/8 = 0.125 FLOPs/byte
}

// Optimized: Fused kernel
__global__ void fused_add_scale(float *A, float *B, float *D, float alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = A[idx] + B[idx];  // Reuse in register
        D[idx] = alpha * sum;
    }
    // FLOPs: 2 (add + mul)
    // Bytes: 3×N×4 (read A, B; write D)
    // AI = 2/12 = 0.167 FLOPs/byte (2x improvement!)
}
```

**Benchmark fusion**:

```python
# TODO: Create `fusion_benchmark.py`
# - Run separate kernels vs fused kernel
# - Measure total time
# - Calculate achieved bandwidth
# - Plot on Roofline model
# - Expected: Fused kernel ~1.8x faster (less memory traffic)
```

### Success Criteria

- [ ] **Roofline plot**: Clear visualization with memory and compute bounds
- [ ] **Ridge point**: Correctly calculated and marked
- [ ] **Kernel profiling**: AI calculated for at least 5 kernels
- [ ] **Validation**: Profiled AI matches manual calculation within 10%
- [ ] **Optimization**: Demonstrate kernel improvement via fusion
- [ ] **Analysis report**: Detailed interpretation of results

### Testing Procedure

```bash
# Profile kernels with Nsight Compute
ncu --set full -o vector_add_profile ./kernels vector_add
ncu --set full -o matmul_profile ./kernels matmul

# Run profiling script
python profile_kernels.py

# Generate Roofline plot
python roofline.py

# Test optimization
python fusion_benchmark.py
```

### Expected Results

```
=== Roofline Analysis (A100 FP32) ===
Peak Compute: 19.5 TFLOPS
Peak Bandwidth: 2048 GB/s
Ridge Point: 9.5 FLOPs/byte

Kernel               | AI (FLOPs/byte) | Achieved (TFLOPS) | % of Roofline
---------------------|-----------------|-------------------|---------------
Vector Add           | 0.083           | 0.17              | 99%
SAXPY (αx+y)         | 0.125           | 0.25              | 98%
Dot Product          | 0.167           | 0.34              | 100%
Matrix-Vec (4096)    | 2.0             | 4.0               | 97%
MatMul (2048)        | 341             | 18.2              | 93%

=== Kernel Fusion Optimization ===
Implementation    | Time (ms) | Bandwidth | Speedup
Separate Kernels  | 0.084     | 1905 GB/s | 1.0x
Fused Kernel      | 0.047     | 1702 GB/s | 1.8x
(Lower bandwidth but fewer transfers!)
```

### Stretch Goals

1. **Multi-precision Roofline**: Plot FP32, FP16, and INT8 rooflines
2. **Cache-aware Roofline**: Add L1/L2 cache bandwidths
3. **Optimization case study**: Take one kernel from 20% to 90% of roofline
4. **Automated profiler**: Tool that profiles any kernel and plots on roofline

---

## Exercise 06: Occupancy Tuning

**Difficulty**: Expert | **Time**: 4 hours

### Learning Objectives

- Understand GPU occupancy and its impact on performance
- Optimize register usage to increase occupancy
- Balance occupancy with per-thread work for maximum throughput
- Use CUDA Occupancy Calculator and compiler tools

### Setup Instructions

```bash
cd exercise-06-occupancy
# Install CUDA Occupancy Calculator (spreadsheet or Python API)
pip install cuda-python
```

### Part 1: Occupancy Analysis (1.5 hours)

**Task**: Analyze how resource usage affects occupancy.

**A100 SM limits**:
- Max threads per SM: 2048
- Max blocks per SM: 32
- Max registers per SM: 65,536
- Max shared memory per SM: 164 KB

```python
# TODO: Create `occupancy_calculator.py`
class OccupancyCalculator:
    def __init__(self, gpu_arch: str = "sm_80"):  # A100
        # SM limits for A100
        self.max_threads_per_sm = 2048
        self.max_blocks_per_sm = 32
        self.max_registers_per_sm = 65536
        self.max_shared_mem_per_sm = 164 * 1024  # bytes
        self.max_threads_per_block = 1024
        self.warp_size = 32

    def calculate_occupancy(self,
                            threads_per_block: int,
                            registers_per_thread: int,
                            shared_mem_per_block: int) -> dict:
        """
        Calculate theoretical occupancy and limiting factors

        Returns:
            {
                'occupancy': 0.75,  # 75% of max threads
                'active_warps': 48,  # out of 64 max
                'active_blocks': 12,
                'limiting_factor': 'registers',
                'blocks_limited_by': {
                    'threads': 16,
                    'registers': 12,
                    'shared_mem': 20
                }
            }
        """
        # TODO: Calculate max blocks limited by each resource
        # TODO: Determine limiting factor
        # TODO: Calculate final occupancy
        pass

    def optimization_suggestions(self, config: dict) -> list:
        """Suggest optimizations to improve occupancy"""
        # TODO: Analyze bottleneck
        # TODO: Return suggestions like:
        # - "Reduce registers per thread from 64 to 48"
        # - "Increase threads per block from 128 to 256"
        pass

# TODO: Create comparison table for different configurations
```

**Generate occupancy heatmap**:

```python
# TODO: Create `occupancy_heatmap.py`
import matplotlib.pyplot as plt
import numpy as np

def generate_occupancy_heatmap():
    """
    Create heatmap showing occupancy for different:
    - X-axis: Threads per block (32, 64, 128, 256, 512, 1024)
    - Y-axis: Registers per thread (16, 32, 48, 64, 80, 96)
    - Color: Occupancy percentage
    """
    # TODO: Calculate occupancy for all combinations
    # TODO: Plot heatmap with seaborn/matplotlib
    # TODO: Highlight optimal configurations
    pass
```

### Part 2: Register Pressure Optimization (1.5 hours)

**Task**: Reduce register usage to improve occupancy.

```cuda
// TODO: Create `register_optimization.cu`
// High register usage: Low occupancy
__global__ void matrix_ops_high_registers(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Many local variables → high register usage
    float a0 = A[idx];
    float a1 = A[idx + N];
    float a2 = A[idx + 2*N];
    float a3 = A[idx + 3*N];
    float b0 = B[idx];
    float b1 = B[idx + N];
    float b2 = B[idx + 2*N];
    float b3 = B[idx + 3*N];

    // Complex computation
    float result = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    result = sqrtf(result * result + 1.0f);
    result = expf(result / 10.0f);

    C[idx] = result;
    // Compiler may allocate 20+ registers
}

// Optimized: Reduced register usage
__global__ void matrix_ops_low_registers(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Reuse variables
    float accumulator = 0.0f;

    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        float a = A[idx + i*N];
        float b = B[idx + i*N];
        accumulator += a * b;  // Reuse registers
    }

    accumulator = sqrtf(accumulator * accumulator + 1.0f);
    accumulator = expf(accumulator / 10.0f);

    C[idx] = accumulator;
    // Compiler may allocate only 10 registers
}

// Force register limit with compiler flag
// Compile with: nvcc -maxrregcount=32 ...
```

**Check register usage**:

```bash
# TODO: Compile with --ptxas-options=-v to see register usage
nvcc -o high_reg register_optimization.cu --ptxas-options=-v

# Output will show:
# ptxas info    : Used 42 registers, 0+0 bytes smem
# Occupancy: 75% (limited by registers)

nvcc -o low_reg register_optimization.cu -O3 --ptxas-options=-v

# Output:
# ptxas info    : Used 24 registers, 0+0 bytes smem
# Occupancy: 100%
```

**Benchmark impact**:

```python
# TODO: Create `register_benchmark.py`
# - Run both kernels with identical workload
# - Measure achieved occupancy (use ncu)
# - Measure execution time
# - Show that higher occupancy → better latency hiding → better performance
```

### Part 3: Occupancy vs Work-per-Thread Trade-off (1 hour)

**Task**: Find optimal balance between occupancy and work per thread.

**Insight**: Higher occupancy isn't always better! Sometimes doing more work per thread with lower occupancy performs better.

```cuda
// TODO: Create `work_per_thread.cu`
// Strategy 1: High occupancy, little work per thread
__global__ void high_occupancy_v1(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Each thread processes 1 element
        data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
    }
    // High occupancy (100%), but more kernel launches needed
}

// Strategy 2: Medium occupancy, more work per thread
__global__ void medium_occupancy_v2(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread processes 4 elements (loop unrolling)
    for (int i = idx; i < N; i += stride) {
        data[i] = sqrtf(data[i] * data[i] + 1.0f);

        if (i + stride < N) {
            data[i + stride] = sqrtf(data[i + stride] * data[i + stride] + 1.0f);
        }
        // More registers used, ~75% occupancy, but fewer launches
    }
}

// Strategy 3: Lower occupancy, vectorized loads
__global__ void low_occupancy_v3(float *data, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < N) {
        // Vectorized load: float4
        float4 values = reinterpret_cast<float4*>(data)[idx/4];

        values.x = sqrtf(values.x * values.x + 1.0f);
        values.y = sqrtf(values.y * values.y + 1.0f);
        values.z = sqrtf(values.z * values.z + 1.0f);
        values.w = sqrtf(values.w * values.w + 1.0f);

        reinterpret_cast<float4*>(data)[idx/4] = values;
        // Even more registers, ~50% occupancy, but better memory efficiency
    }
}
```

**Comprehensive benchmark**:

```python
# TODO: Create `work_per_thread_benchmark.py`
# - Test all three strategies
# - Measure achieved occupancy with ncu
# - Measure execution time
# - Measure achieved bandwidth
# - Plot occupancy vs performance
# - Find optimal strategy for the workload
```

### Success Criteria

- [ ] **Occupancy calculator**: Correctly computes occupancy for given config
- [ ] **Heatmap**: Clear visualization of occupancy space
- [ ] **Register optimization**: Reduce register usage by >30%
- [ ] **Performance improvement**: Show register-optimized kernel is faster
- [ ] **Work-per-thread analysis**: Demonstrate optimal balance point
- [ ] **Profiling**: Use ncu to validate theoretical calculations
- [ ] **Report**: Comprehensive analysis with recommendations

### Testing Procedure

```bash
# Compile with verbose output
nvcc -o register_optimization register_optimization.cu --ptxas-options=-v -lineinfo

# Profile occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./register_optimization

# Run benchmarks
python register_benchmark.py
python work_per_thread_benchmark.py

# Generate visualizations
python occupancy_heatmap.py
```

### Expected Results

```
=== Occupancy Analysis ===
Configuration | Threads | Regs | Shared | Occupancy | Limiting
Config 1      | 256     | 64   | 0      | 50%       | Registers
Config 2      | 256     | 32   | 0      | 100%      | None
Config 3      | 512     | 32   | 8 KB   | 100%      | None
Config 4      | 1024    | 32   | 16 KB  | 62.5%     | Shared Mem

=== Register Optimization ===
Kernel              | Registers | Occupancy | Time (ms) | Speedup
High Register       | 48        | 66.7%     | 1.25      | 1.0x
Optimized           | 28        | 100%      | 0.87      | 1.44x
With -maxrregcount  | 24        | 100%      | 0.82      | 1.52x

=== Work Per Thread Strategy ===
Strategy            | Occupancy | Time (ms) | Elements/Thread
High Occupancy      | 100%      | 1.05      | 1
Medium Occupancy    | 75%       | 0.78      | 4 ← Best
Low Occupancy       | 50%       | 0.92      | 4 (vectorized)

Conclusion: 75% occupancy with 4 elements/thread is optimal for this workload
```

### Stretch Goals

1. **Automated tuner**: Auto-tune kernel launch parameters for max performance
2. **Multi-kernel pipeline**: Optimize occupancy for concurrent kernel execution
3. **Dynamic occupancy**: Adjust launch config based on runtime conditions
4. **Instruction-level profiling**: Use SASS analysis to understand register allocation

---

## Submission Guidelines

### Required Deliverables

For each exercise:
1. **Source code**: All `.cu`, `.py` files with implementations
2. **Build scripts**: `Makefile` or `build.sh` for compilation
3. **Test results**: Console output, screenshots, or logs
4. **Visualizations**: Charts, plots saved as PNG/PDF
5. **Written report**: Markdown file with:
   - Implementation approach
   - Results analysis
   - Performance metrics
   - Lessons learned
   - Answers to discussion questions

### Directory Structure

```
exercise-XX-name/
├── README.md               # Your implementation notes
├── src/
│   ├── kernel.cu          # CUDA kernels
│   ├── benchmark.py       # Python benchmarking code
│   └── utils.py           # Helper functions
├── results/
│   ├── output.txt         # Console output
│   ├── plots/             # Generated charts
│   └── profiles/          # Nsight Compute reports
├── report.md              # Written analysis
├── Makefile
└── requirements.txt
```

### Code Quality Standards

- **Style**: Follow PEP 8 (Python) and Google C++ Style Guide (CUDA)
- **Documentation**: Docstrings for all functions, inline comments for complex logic
- **Error handling**: Check CUDA errors, validate inputs
- **Type hints**: Use Python type annotations
- **Testing**: Include unit tests where applicable

### Evaluation Criteria

Each exercise is graded on:
1. **Correctness** (30%): Does the implementation work correctly?
2. **Performance** (30%): Does it meet the performance targets?
3. **Code Quality** (20%): Is the code clean, readable, and well-documented?
4. **Analysis** (20%): Is the written report thorough and insightful?

### Getting Help

- Review lecture notes in `../lecture-notes/`
- Check NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Post questions in discussion forum
- Office hours: [TBD]

---

## Additional Resources

### Official Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [PyCUDA Documentation](https://documen.tician.de/pycuda/)

### Books
- "Programming Massively Parallel Processors" by Hwu, Kirk, and Hajj
- "CUDA by Example" by Sanders and Kandrot
- "Professional CUDA C Programming" by Cheng et al.

### Online Courses
- NVIDIA DLI: Fundamentals of Accelerated Computing with CUDA C/C++
- Coursera: GPU Programming Specialization
- Udacity: Intro to Parallel Programming

### Tools
- NVIDIA Nsight Compute (profiling)
- NVIDIA Nsight Systems (system-wide profiling)
- CUDA-GDB (debugging)
- CUDA-MEMCHECK (memory error detection)

---

**Good luck with the exercises! Remember: Understanding GPU architecture is the foundation for all high-performance ML/AI work.**
