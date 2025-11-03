# Module 1: Performance Fundamentals

## Overview

This module provides the foundational knowledge required for AI/ML performance engineering. You'll learn GPU architecture, CUDA programming fundamentals, memory hierarchies, and performance modeling. By the end of this module, you'll understand why GPUs are essential for deep learning and how to analyze GPU performance using the Roofline model.

**Duration**: 20 hours
**Difficulty**: Intermediate to Advanced
**Prerequisites**: Python programming, basic C/C++, linear algebra, understanding of neural networks

## Learning Objectives

By completing this module, you will be able to:

1. **Explain GPU architecture** and its advantages for deep learning workloads
2. **Understand CUDA programming model** including threads, warps, blocks, and grids
3. **Analyze memory hierarchy** and calculate effective bandwidth for each level
4. **Apply the Roofline model** to identify compute vs memory bottlenecks
5. **Calculate arithmetic intensity** for different algorithms
6. **Optimize memory access patterns** for coalescing and reduced latency
7. **Measure GPU performance** using theoretical calculations and profiling tools
8. **Identify performance bottlenecks** and apply appropriate optimization strategies

## Module Structure

### 1. Lecture Notes (7 hours)

Located in `lecture-notes/01-gpu-architecture.md`

**Topics Covered**:
- Why GPUs for Deep Learning? (10,000x speedup potential)
- GPU Architecture Evolution (Pascal → Volta → Ampere → Hopper)
- Streaming Multiprocessors and CUDA Cores
- Tensor Cores and Mixed Precision
- CUDA Programming Model (threads, warps, blocks, grids)
- Memory Hierarchy (registers, shared memory, L1/L2 cache, HBM)
- Performance Metrics (FLOPS, bandwidth, occupancy)
- Roofline Model (compute vs memory bound analysis)
- Optimization Principles

**Key Concepts**:
- NVIDIA A100: 108 SMs, 6,912 CUDA cores, 432 Tensor Cores, 2 TB/s bandwidth
- Memory latency: Registers (1 cycle) → Shared (28 cycles) → Global (350 cycles)
- Ridge Point: Peak Compute / Peak Bandwidth = 156 FLOPs/byte for A100 FP16
- Memory coalescing: 32x performance difference between optimal and strided access
- Warp divergence: SIMT execution model requires branch coherence

### 2. Hands-On Exercises (13 hours)

Located in `exercises/README.md`

Six comprehensive exercises building practical skills:

| Exercise | Topic | Time | Key Skills |
|----------|-------|------|------------|
| 01 | GPU Architecture Analysis | 2h | Spec analysis, performance calculations |
| 02 | Memory Bandwidth Measurement | 3h | Benchmarking, coalescing, PCIe transfers |
| 03 | CUDA Thread Hierarchy | 3h | Kernel launch config, parallel algorithms |
| 04 | Shared Memory Optimization | 4h | Tiled matmul, bank conflicts, occupancy |
| 05 | Roofline Analysis | 4h | Profiling, arithmetic intensity, bottleneck identification |
| 06 | Occupancy Tuning | 4h | Register optimization, resource management |

**Tools You'll Use**:
- CUDA Toolkit 12.0+
- Python with PyTorch, PyCUDA
- NVIDIA Nsight Compute (profiling)
- NVIDIA Nsight Systems (system-wide analysis)

## Prerequisites

### Required Knowledge

- **Programming**:
  - Python (intermediate level)
  - C/C++ (basic level - pointers, functions, structs)

- **Mathematics**:
  - Linear algebra (matrix operations, vectors)
  - Basic calculus (derivatives, chain rule)

- **Machine Learning**:
  - Neural network fundamentals
  - Training vs inference
  - Backpropagation basics

### Required Hardware

- **GPU**: NVIDIA GPU with compute capability 7.0+ (Volta or newer)
  - Recommended: A100, V100, RTX 3090, RTX 4090, or H100
  - Minimum: GTX 1080 Ti, RTX 2060

- **System**:
  - 64 GB RAM (32 GB minimum)
  - Ubuntu 20.04+ or equivalent Linux distribution
  - 50 GB free disk space

### Required Software

```bash
# CUDA Toolkit 12.0+
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
sudo sh cuda_12.3.0_545.23.06_linux.run

# Python environment
conda create -n perf-eng python=3.10
conda activate perf-eng

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install profiling and benchmarking tools
pip install pycuda numpy matplotlib pandas jupyter
pip install nvidia-pyindex
pip install nvidia-nsight-compute

# Verify installation
nvidia-smi
nvcc --version
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Getting Started

### 1. Start with Lecture Notes

Read through the comprehensive lecture notes to build theoretical understanding:

```bash
cd lecture-notes
# Read 01-gpu-architecture.md
```

**Time Allocation**:
- First pass: 3 hours (read everything)
- Take notes, sketch diagrams
- Second pass: 2 hours (focus on key concepts)
- Review code examples
- Final review: 2 hours (before starting exercises)

**Study Tips**:
- Draw the GPU architecture hierarchy on paper
- Calculate examples by hand (memory bandwidth, arithmetic intensity)
- Run the code examples in a Jupyter notebook
- Watch your GPU in action with `nvidia-smi -l 1`

### 2. Complete Exercises Sequentially

The exercises build upon each other, so complete them in order:

```bash
cd exercises

# Exercise 1: Understand the hardware
cd exercise-01-gpu-architecture
python gpu_specs.py

# Exercise 2: Measure memory bandwidth
cd ../exercise-02-memory-bandwidth
python memory_copy.py

# Continue through all 6 exercises...
```

**Each Exercise Includes**:
- Clear learning objectives
- Step-by-step implementation guide
- TODO markers in code for hands-on work
- Success criteria with performance targets
- Testing procedures
- Stretch goals for deeper learning

### 3. Validate Your Understanding

After completing the module, you should be able to:

- [ ] Explain why GPUs are 100-10,000x faster than CPUs for ML workloads
- [ ] Calculate theoretical peak performance for any GPU
- [ ] Identify whether a kernel is compute-bound or memory-bound
- [ ] Write CUDA kernels with optimal thread configuration
- [ ] Optimize memory access patterns for coalescing
- [ ] Use shared memory to reduce global memory traffic
- [ ] Profile kernels with Nsight Compute
- [ ] Plot kernel performance on the Roofline model
- [ ] Optimize occupancy without sacrificing per-thread work

## Key Takeaways

### 1. GPU Architecture Essentials

**Why GPUs?**
```
CPU: 8-64 cores, ~200 GB/s memory bandwidth
GPU: 7,000+ CUDA cores, 2,000+ GB/s memory bandwidth

Matrix multiply (4096×4096):
CPU (64-core): ~5 seconds
GPU (A100):    ~5 milliseconds
Speedup:       1,000x
```

**Memory Hierarchy Performance**:
```
Registers:      ~20 TB/s    (1 cycle)
Shared Memory:  ~12 TB/s    (28 cycles)
L2 Cache:       ~8 TB/s     (200 cycles)
HBM (Global):   ~2 TB/s     (350 cycles)
PCIe:           ~0.06 TB/s  (20,000 cycles)

Optimization Goal: Keep data in registers and shared memory!
```

### 2. CUDA Programming Model

**Thread Hierarchy**:
```
Grid (device)
├── Block 0 (CTA - Cooperative Thread Array)
│   ├── Warp 0 (32 threads)
│   │   ├── Thread 0
│   │   ├── Thread 1
│   │   └── ...
│   └── Warp 1 (32 threads)
│       └── ...
└── Block 1
    └── ...

Key Insight: Warps execute in lockstep (SIMT)
- Threads in a warp must take the same path
- Divergence causes serialization
- Always keep warps coherent!
```

**Thread ID Calculation**:
```c
// 1D grid
int global_id = blockIdx.x * blockDim.x + threadIdx.x;

// 2D grid (for images)
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// 3D grid (for volumes)
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

### 3. Roofline Model

The Roofline model is your most important tool for performance analysis.

**Formula**:
```
Achievable Performance = min(Peak Compute, Arithmetic Intensity × Peak Bandwidth)

Ridge Point = Peak Compute / Peak Bandwidth

If AI < Ridge Point: Memory Bound → Optimize memory access
If AI >= Ridge Point: Compute Bound → Optimize compute
```

**Example (A100 FP16 with Tensor Cores)**:
```
Peak Compute: 312 TFLOPS
Peak Bandwidth: 2 TB/s
Ridge Point: 312 / 2 = 156 FLOPs/byte

Vector Addition:
AI = 1/12 = 0.083 FLOPs/byte < 156 → Memory Bound!
Max Performance = 0.083 × 2000 = 166 GFLOPS (0.05% of peak compute)

Matrix Multiply (N=4096):
AI = N/6 = 682 FLOPs/byte > 156 → Compute Bound!
Max Performance = 312 TFLOPS (100% of peak compute)
```

### 4. Optimization Checklist

When optimizing GPU code, follow this priority order:

1. **Choose the right algorithm** (10-1000x impact)
   - Matrix multiply beats element-wise ops for large matrices
   - FFT beats naive convolution for large kernels
   - Parallel reduction beats sequential sum

2. **Maximize parallelism** (2-10x impact)
   - Launch enough blocks to saturate GPU
   - Keep all SMs busy
   - Use all available CUDA cores

3. **Optimize memory access** (2-32x impact)
   - Coalesce global memory accesses
   - Minimize global memory transactions
   - Use shared memory for data reuse
   - Avoid strided access patterns

4. **Minimize divergence** (1.5-2x impact)
   - Keep threads in a warp on the same path
   - Restructure conditionals to be warp-aligned
   - Use __ballot_sync, __any_sync for warp-level ops

5. **Tune occupancy** (1.2-2x impact)
   - Balance threads per block with resource usage
   - Reduce register pressure
   - Optimize shared memory usage
   - Use occupancy calculator

6. **Use specialized hardware** (5-16x impact)
   - Tensor Cores for matrix operations (16x for FP16)
   - Warp-level intrinsics (__shfl, __ballot)
   - Fast math functions (expf, sqrtf)

## Performance Benchmarks

After completing this module, your code should achieve:

### Memory Operations
```
Device Memory Copy:     >90% of peak bandwidth (1,843+ GB/s on A100)
Coalesced Access:       >95% of peak bandwidth
Strided Access (×32):   ~3% of peak (demonstrates importance!)
PCIe Transfer (pinned): >50 GB/s
```

### Compute Kernels
```
Vector Addition:        >85% of theoretical (memory bound)
Matrix Multiply (2K):   >50% of PyTorch cuBLAS performance
Shared Memory MatMul:   5-10x speedup vs naive implementation
Optimized Reduction:    >4x speedup vs naive tree reduction
```

### Profiling Metrics
```
Achieved Occupancy:     >75% for most kernels
Memory Efficiency:      >80% for coalesced access
Compute Utilization:    >50% for compute-bound kernels
Warp Execution Eff:     >95% (low divergence)
```

## Common Pitfalls and How to Avoid Them

### 1. Ignoring Memory Coalescing
**Problem**: Strided or random memory access → 32x slowdown
```cuda
// ❌ BAD: Strided access
float value = data[threadIdx.x * 32];  // Each thread accesses 32 elements apart

// ✅ GOOD: Coalesced access
float value = data[threadIdx.x];  // Consecutive threads access consecutive elements
```

### 2. Excessive Shared Memory Usage
**Problem**: High shared memory usage → low occupancy → underutilized GPU
```cuda
// ❌ BAD: 64 KB shared memory → only 2 blocks per SM
__shared__ float cache[16384];  // 64 KB

// ✅ GOOD: 16 KB shared memory → 10 blocks per SM
__shared__ float cache[4096];  // 16 KB, or use smaller tiles
```

### 3. Unoptimized Launch Configuration
**Problem**: Wrong block size → poor occupancy or excessive overhead
```cuda
// ❌ BAD: 37 threads per block (not multiple of 32, wastes resources)
kernel<<<numBlocks, 37>>>(...);

// ✅ GOOD: 256 threads per block (8 warps, good occupancy)
kernel<<<numBlocks, 256>>>(...);
```

### 4. Warp Divergence
**Problem**: Threads in warp take different paths → serialization
```cuda
// ❌ BAD: Even/odd threads diverge
if (threadIdx.x % 2 == 0) {
    // Expensive path
} else {
    // Cheap path
}
// Both paths execute serially!

// ✅ GOOD: Separate warps take different paths
if (threadIdx.x / 32 == 0) {  // First warp
    // Path A
} else {  // Other warps
    // Path B
}
// Warps remain coherent
```

### 5. Ignoring Arithmetic Intensity
**Problem**: Memory-bound kernel treated as compute-bound → wasted optimization effort
```python
# ❌ BAD: Optimizing compute when memory is the bottleneck
# Vector add is memory-bound (AI = 0.083)
# Optimizing the addition operation won't help!

# ✅ GOOD: Fuse operations to increase AI
# Combined add + scale increases AI from 0.083 to 0.167
# Reduces memory traffic by 33%
```

## Debugging Tips

### CUDA Error Checking
Always check for CUDA errors:
```cuda
// Wrap all CUDA calls
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Use it
CHECK_CUDA(cudaMalloc(&d_ptr, size));
CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
kernel<<<grid, block>>>();
CHECK_CUDA(cudaGetLastError());  // Check kernel launch
CHECK_CUDA(cudaDeviceSynchronize());  // Wait and check execution
```

### cuda-memcheck
Detect memory errors:
```bash
cuda-memcheck ./my_program
# Detects: out-of-bounds access, race conditions, uninitialized memory
```

### Nsight Compute
Profile individual kernels:
```bash
# Basic profiling
ncu --set full -o profile ./my_program

# Check specific metrics
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
               dram__bytes.sum,\
               smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
    ./my_program

# Open GUI
ncu-ui profile.ncu-rep
```

### Nsight Systems
System-wide timeline:
```bash
nsys profile --stats=true -o timeline ./my_program
nsys-ui timeline.nsys-rep
```

## Next Steps

After completing Module 1, you're ready for:

### Module 2: Advanced GPU Optimization
- Custom CUDA kernels for ML operations
- Warp-level programming with cooperative groups
- Tensor Core programming with WMMA API
- Multi-GPU programming with NCCL

### Module 3: Performance Profiling & Debugging
- Advanced profiling with Nsight Compute
- Performance regression detection
- Kernel optimization workflows
- CUDA best practices deep dive

### Module 4: Transformer Optimization
- Flash Attention implementation
- Fused kernels for LayerNorm, GELU
- KV cache optimization
- Long context optimization

## Resources

### Official Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - Comprehensive reference
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - Optimization patterns
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/) - Profiling tool docs

### Books
- **"Programming Massively Parallel Processors" by Hwu, Kirk, and Hajj**
  - The definitive textbook on GPU programming
  - Covers CUDA fundamentals through advanced topics
  - Includes exercises and case studies

- **"CUDA by Example" by Sanders and Kandrot**
  - Beginner-friendly introduction
  - Lots of code examples

- **"Professional CUDA C Programming" by Cheng et al.**
  - Advanced optimization techniques
  - Performance tuning strategies

### Online Courses
- [NVIDIA DLI: Fundamentals of Accelerated Computing](https://www.nvidia.com/en-us/training/)
  - Official NVIDIA training
  - Hands-on labs with cloud GPUs

- [Coursera: GPU Programming Specialization](https://www.coursera.org/)
  - University-level course
  - Theory and practice

### Community
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [CUDA subreddit](https://www.reddit.com/r/CUDA/)
- [Stack Overflow - CUDA tag](https://stackoverflow.com/questions/tagged/cuda)

### Papers
- **"Roofline: An Insightful Visual Performance Model"** (Williams et al., 2009)
- **"Understanding the GPU Microarchitecture"** (Jia et al., 2018)
- **"Dissecting the NVIDIA Volta GPU Architecture"** (NVIDIA, 2017)

## Assessment

To pass this module, you must:

1. **Complete all 6 exercises** with passing scores (70%+)
2. **Achieve performance targets** specified in each exercise
3. **Submit written reports** analyzing your results
4. **Pass the module quiz** (20 questions, 80% required)

### Sample Quiz Questions

1. What is the arithmetic intensity of vector addition? Why is it memory-bound?
2. Calculate the ridge point for NVIDIA H100 (989 TFLOPS FP16, 3.3 TB/s bandwidth)
3. Explain why memory coalescing matters and how to achieve it
4. What is warp divergence and how do you avoid it?
5. Given a kernel using 48 registers/thread with 256 threads/block, what is the occupancy on A100?

## Support

### Getting Help

1. **Review lecture notes** - Most questions are answered there
2. **Check exercise README** - Detailed implementation guides
3. **Search NVIDIA docs** - Official documentation is comprehensive
4. **Post in forum** - Community support available
5. **Office hours** - [Schedule TBD]

### Reporting Issues

Found a bug or typo? Please report:
- GitHub Issues: [repository URL]
- Email: [instructor email]
- Include: Module number, exercise number, description, screenshots

---

## Summary

Module 1 provides the essential foundation for AI/ML performance engineering. You've learned GPU architecture, CUDA programming, memory optimization, and performance modeling. These concepts are critical for all subsequent modules.

**Key Skills Acquired**:
- ✅ GPU architecture understanding
- ✅ CUDA kernel development
- ✅ Memory hierarchy optimization
- ✅ Performance profiling
- ✅ Roofline analysis
- ✅ Bottleneck identification

**Time Investment**: 20 hours
**Difficulty**: Intermediate to Advanced
**Prerequisites**: Programming, linear algebra, basic ML

**Ready to continue?** Proceed to Module 2: Advanced GPU Optimization

---

*Last Updated: 2025-11-02*
*Module Version: 1.0*
*Feedback: [contact information]*
