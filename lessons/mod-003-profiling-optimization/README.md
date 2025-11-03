# Module 3: Performance Profiling and Optimization

## Overview

This module teaches advanced GPU profiling techniques and systematic optimization workflows used in production ML systems. You'll master industry-standard tools (NVIDIA Nsight Compute, Nsight Systems) and learn to identify, analyze, and resolve performance bottlenecks in real-world ML workloads.

**Duration**: 5-6 weeks (25-30 hours)

## Prerequisites

- Completed Module 1 (Performance Fundamentals)
- Completed Module 2 (CUDA Programming)
- Strong understanding of GPU architecture and CUDA programming
- Experience writing custom CUDA kernels
- Familiarity with PyTorch or TensorFlow
- Basic understanding of performance metrics

## Learning Objectives

By the end of this module, you will be able to:

1. **Master Profiling Tools**
   - Use NVIDIA Nsight Compute for kernel-level profiling
   - Use NVIDIA Nsight Systems for system-wide performance analysis
   - Integrate PyTorch Profiler into training pipelines
   - Choose appropriate profiling tools for different scenarios

2. **Interpret Performance Metrics**
   - Understand Speed of Light (SOL) metrics
   - Analyze memory bandwidth utilization
   - Interpret occupancy and warp efficiency
   - Identify memory coalescing issues
   - Understand warp stall reasons

3. **Identify Bottlenecks**
   - Classify kernels as compute-bound, memory-bound, or latency-bound
   - Identify synchronization bottlenecks
   - Find kernel fusion opportunities
   - Detect memory allocation hotspots
   - Recognize cache thrashing patterns

4. **Apply Optimization Workflows**
   - Follow systematic profiling and optimization processes
   - Prioritize optimizations by impact
   - Validate improvements with profiling
   - Document optimization decisions
   - Balance optimization effort vs impact

5. **Implement Production Profiling**
   - Build lightweight profiling with <5% overhead
   - Create continuous performance monitoring
   - Implement automated anomaly detection
   - Set up performance regression testing
   - Integrate profiling with CI/CD pipelines

## Module Structure

### Lecture Notes

1. **Advanced Profiling Techniques** (`lecture-notes/01-advanced-profiling.md`)
   - Introduction to GPU profiling philosophy
   - NVIDIA Nsight Compute detailed walkthrough
   - NVIDIA Nsight Systems for system-wide analysis
   - Performance metrics deep dive
   - Bottleneck identification patterns
   - Optimization workflows and methodologies
   - Case studies: LayerNorm, Flash Attention, GELU
   - Production profiling best practices
   - Performance regression detection

### Exercises

The exercises provide hands-on experience with real-world profiling scenarios:

1. **Nsight Compute Deep Dive** (4 hours)
   - Profile reference kernels (matmul, elementwise, reduction)
   - Analyze custom GELU implementations
   - Deep dive into performance metrics
   - Extract and visualize profiling data

2. **Nsight Systems Timeline Analysis** (4 hours)
   - Profile training loops and identify GPU idle time
   - Optimize DataLoader configurations
   - Analyze transformer training pipelines
   - Identify kernel fusion opportunities

3. **Bottleneck Identification and Resolution** (5 hours)
   - Classify bottlenecks from profiling data
   - Optimize transformer attention kernels
   - Debug performance regressions
   - Apply systematic optimization workflows

See `exercises/README.md` for complete exercise descriptions.

## Key Concepts

### Profiling Tools Comparison

| Tool | Use Case | Overhead | Granularity | Production-Ready |
|------|----------|----------|-------------|------------------|
| Nsight Compute | Kernel optimization | 100x | Per-kernel | No (dev only) |
| Nsight Systems | System bottlenecks | 5% | System-wide | Yes (with care) |
| PyTorch Profiler | ML pipeline analysis | 10-20% | Python + CUDA | Development |
| Custom Profiling | Production monitoring | <5% | Configurable | Yes |

### Performance Metrics

**Speed of Light (SOL)**: Percentage of theoretical peak achieved

```
Compute SOL = (Actual FLOP/s) / (Peak FLOP/s) × 100%
Memory SOL = (Actual Bandwidth) / (Peak Bandwidth) × 100%
```

**Occupancy**: Active warps as percentage of maximum possible

```
Occupancy = (Active Warps) / (Max Warps per SM) × 100%
```

**Warp Efficiency**: Percentage of threads in a warp that are active

```
Warp Efficiency = (Active Threads) / (32 threads per warp) × 100%
```

**Memory Coalescing Efficiency**: Percentage of memory transactions that are coalesced

```
Coalescing Efficiency = (Theoretical Transactions) / (Actual Transactions) × 100%
```

### Bottleneck Classification

1. **Compute-Bound**
   - High Compute SOL (>60%)
   - Low Memory SOL (<40%)
   - **Optimization**: Improve instruction mix, use Tensor Cores, increase arithmetic intensity

2. **Memory-Bound**
   - Low Compute SOL (<40%)
   - High Memory SOL (>60%)
   - **Optimization**: Improve memory coalescing, use shared memory, fuse kernels

3. **Latency-Bound**
   - Low Compute SOL (<40%)
   - Low Memory SOL (<40%)
   - High warp stalls on memory/sync
   - **Optimization**: Increase occupancy, reduce divergence, optimize launch config

4. **Bandwidth-Limited**
   - High Memory SOL (>80%)
   - **Optimization**: Reduce memory traffic, increase data reuse, compress data

5. **Synchronization-Bound**
   - Frequent CPU-GPU sync points
   - GPU idle time in timeline
   - **Optimization**: Remove synchronization, use async operations, batch operations

### Optimization Workflow

```
1. Profile ’ Identify Hotspot
   “
2. Classify Bottleneck Type
   “
3. Prioritize by Impact × Effort
   “
4. Implement Optimization
   “
5. Profile Again ’ Validate
   “
6. Repeat Until Target Met
```

## Performance Benchmarks

Expected profiling and optimization results on A100 GPU:

### Nsight Compute Profiling

| Kernel Type | Baseline | Optimized | Speedup | Key Metric |
|-------------|----------|-----------|---------|------------|
| GELU | 0.85 ms | 0.12 ms | 7.1x | Fast math, vectorization |
| LayerNorm | 1.20 ms | 0.31 ms | 3.9x | Welford's algorithm, fusion |
| Attention QK | 2.50 ms | 0.95 ms | 2.6x | Shared memory, coalescing |
| Softmax | 0.45 ms | 0.18 ms | 2.5x | Warp reductions |
| Embedding | 0.60 ms | 0.35 ms | 1.7x | L2 cache optimization |

### Nsight Systems Profiling

| Pipeline | Baseline GPU % | Optimized GPU % | Improvement |
|----------|---------------|-----------------|-------------|
| Training Loop | 45% | 92% | +47 pp |
| DataLoader | 35% | 88% | +53 pp |
| Transformer | 60% | 90% | +30 pp |

### Production Profiling

| Requirement | Target | Typical Result |
|-------------|--------|----------------|
| Overhead | <5% | 1-2% |
| Latency | <1 ms per operation | 0.1-0.5 ms |
| Memory | <100 MB | 20-50 MB |
| Anomaly Detection | >95% accuracy | 97-99% |

## Tools and Setup

### Required Software

1. **NVIDIA CUDA Toolkit 12.0+**
   ```bash
   # Includes Nsight Compute and Nsight Systems
   # Download from: https://developer.nvidia.com/cuda-downloads
   ```

2. **NVIDIA Nsight Compute**
   ```bash
   ncu --version
   # Should show version 2023.3.0 or later
   ```

3. **NVIDIA Nsight Systems**
   ```bash
   nsys --version
   # Should show version 2023.4.1 or later
   ```

4. **Python Environment**
   ```bash
   pip install torch torchvision pytest pytest-benchmark
   pip install matplotlib pandas seaborn  # For visualizations
   ```

### Recommended Hardware

- **Minimum**: NVIDIA GPU with compute capability 7.0+ (V100, T4, RTX 2000 series)
- **Recommended**: A100, H100, or RTX 4090 for realistic performance testing

### Environment Setup

```bash
# Create profiling workspace
mkdir -p ~/ml-profiling
cd ~/ml-profiling

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Best Practices

### Profiling Methodology

1. **Start Broad, Then Narrow**
   - Begin with Nsight Systems to find hotspots
   - Use Nsight Compute for specific kernel optimization
   - Focus on top 20% of time-consuming operations (80/20 rule)

2. **Profile Realistically**
   - Use production-like input sizes
   - Include warm-up iterations
   - Profile multiple runs for statistical significance
   - Consider cold vs warm cache effects

3. **Compare Apples to Apples**
   - Same hardware configuration
   - Same input data
   - Same CUDA version
   - Control for GPU clock throttling

4. **Document Everything**
   - Save profile reports (.ncu-rep, .qdrep)
   - Record system configuration
   - Note optimization rationale
   - Track performance progression

### Optimization Guidelines

1. **Validate Correctness First**
   ```python
   # Always check numerical accuracy after optimization
   assert torch.allclose(output_optimized, output_baseline, atol=1e-3, rtol=1e-3)
   ```

2. **One Change at a Time**
   - Make incremental optimizations
   - Profile after each change
   - Roll back if regression occurs

3. **Consider Diminishing Returns**
   - Stop when target performance is met
   - Balance optimization time vs benefit
   - Document remaining opportunities

4. **Production Readiness Checklist**
   - [ ] Numerical accuracy validated
   - [ ] Performance meets requirements
   - [ ] No memory leaks
   - [ ] Graceful error handling
   - [ ] Monitoring integrated
   - [ ] Regression tests added

## Common Pitfalls and Solutions

### Pitfall 1: Over-Profiling

**Problem**: Profiling overhead slows down training significantly

**Solution**:
- Use Nsight Systems (5% overhead) for production
- Profile representative samples, not entire runs
- Disable profiling in production after validation

### Pitfall 2: Premature Optimization

**Problem**: Optimizing operations that aren't bottlenecks

**Solution**:
- Always profile first to identify actual bottlenecks
- Use Amdahl's Law to estimate maximum speedup
- Focus on operations consuming >5% of total time

### Pitfall 3: Ignoring Data Dependencies

**Problem**: Optimizations that work in isolation fail in pipelines

**Solution**:
- Profile end-to-end workflows
- Consider memory layout compatibility
- Test with realistic data dependencies

### Pitfall 4: Hardware-Specific Optimizations

**Problem**: Code optimized for A100 performs poorly on V100

**Solution**:
- Use runtime dispatch based on GPU capability
- Test on target hardware
- Use architecture-agnostic optimizations when possible

### Pitfall 5: Neglecting Numerical Stability

**Problem**: Fast math optimizations cause training divergence

**Solution**:
```python
# Always validate numerical accuracy
max_error = torch.max(torch.abs(output_fast - output_exact))
assert max_error < 1e-4, f"Accuracy loss too large: {max_error}"
```

## Case Study: Optimizing GPT-2 Inference

**Scenario**: Reduce GPT-2 inference latency from 180ms to <50ms

**Workflow**:

1. **Initial Profiling** (Nsight Systems)
   - GPU utilization: 45%
   - Identify: Attention (40%), MLP (30%), LayerNorm (15%)

2. **Attention Optimization** (Nsight Compute)
   - Baseline: 72ms, Memory SOL: 85%, Compute SOL: 30%
   - Bottleneck: Memory-bound, no KV cache
   - Implement KV cache ’ 18ms (4x speedup)
   - Implement Flash Attention ’ 15ms (4.8x speedup)

3. **MLP Optimization**
   - Baseline: 54ms, unfused operations
   - Fuse GELU + Linear ’ 38ms (1.4x speedup)
   - Custom GELU kernel ’ 32ms (1.7x speedup)

4. **LayerNorm Optimization**
   - Baseline: 27ms, standard implementation
   - Welford's algorithm + fusion ’ 12ms (2.25x speedup)

5. **Final Result**
   - Total latency: 42ms (4.3x speedup vs baseline)
   - GPU utilization: 88%
   - Met SLA target of <50ms

## Additional Resources

### Official Documentation

- [NVIDIA Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [CUDA Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)

### Tutorials and Guides

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/)
- [Mixed Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)

### Research Papers

- [Roofline Model](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Performance Analysis of GPU Applications](https://dl.acm.org/doi/10.1145/3431921)

### Community Resources

- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [PyTorch Discuss - Performance](https://discuss.pytorch.org/c/performance/)
- [/r/MachineLearning Performance Threads](https://www.reddit.com/r/MachineLearning/)

## Assessment

### Knowledge Check

After completing this module, you should be able to answer:

1. When would you use Nsight Compute vs Nsight Systems?
2. What does a Memory SOL of 85% indicate? How would you optimize?
3. What are the three main types of bottlenecks? How do you identify each?
4. What profiling overhead is acceptable for production monitoring?
5. How do you validate that an optimization doesn't harm numerical accuracy?

### Practical Assessment

Complete the following tasks:

1. Profile a custom CUDA kernel and identify its primary bottleneck
2. Optimize a transformer attention implementation to achieve >3x speedup
3. Implement lightweight production profiling with <5% overhead
4. Build a performance regression testing suite for CI/CD
5. Debug a 30% performance regression in production code

### Expected Competencies

By module completion, you should demonstrate:

- **Profiling Expertise**: Efficiently use Nsight tools to identify bottlenecks
- **Metrics Interpretation**: Correctly interpret SOL, occupancy, and bandwidth metrics
- **Optimization Skills**: Apply systematic optimization workflows
- **Production Readiness**: Implement profiling suitable for production systems
- **Problem Solving**: Debug performance issues using profiling data

## Next Steps

After completing this module:

1. **Module 4: Transformer Optimization**
   - Apply profiling skills to optimize transformer models
   - Learn Flash Attention v2, fused operations, and KV cache optimization
   - Optimize long-context attention mechanisms

2. **Module 5: Model Compression and Quantization**
   - Profile quantized models
   - Measure compression-performance trade-offs
   - Optimize INT8 inference kernels

3. **Module 6: Distributed Inference**
   - Profile multi-GPU systems
   - Identify communication bottlenecks
   - Optimize tensor parallel inference

## Getting Help

If you encounter issues:

1. **Profiling Tool Issues**
   - Check CUDA version compatibility
   - Verify driver version (nvidia-smi)
   - Consult NVIDIA Developer Forums

2. **Performance Questions**
   - Review lecture notes for similar cases
   - Compare your metrics with expected results
   - Check hardware specifications

3. **Exercise Problems**
   - Verify environment setup
   - Check input data shapes and types
   - Compare with baseline implementations

## Summary

This module provides comprehensive training in GPU profiling and optimization techniques used in production ML systems. By mastering tools like Nsight Compute and Nsight Systems, understanding performance metrics, and applying systematic optimization workflows, you'll be equipped to identify and resolve performance bottlenecks in real-world AI infrastructure.

The skills learned here are foundational for advanced performance engineering topics and are directly applicable to optimizing production ML systems at scale.

**Time Investment**: 25-30 hours
**Expected Outcome**: Ability to profile and optimize GPU workloads with 3-5x speedups
**Industry Relevance**: High - These skills are core requirements for ML Performance Engineering roles

---

**Module Maintainer**: AI Infrastructure Curriculum Team
**Last Updated**: 2025
**Feedback**: Please report issues or suggestions via GitHub Issues
