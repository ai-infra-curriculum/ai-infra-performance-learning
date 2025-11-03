# Advanced Performance Profiling and Optimization

## Table of Contents
1. [Introduction](#introduction)
2. [Profiling Tools Overview](#profiling-tools-overview)
3. [NVIDIA Nsight Compute](#nvidia-nsight-compute)
4. [NVIDIA Nsight Systems](#nvidia-nsight-systems)
5. [Performance Metrics Deep Dive](#performance-metrics-deep-dive)
6. [Bottleneck Identification](#bottleneck-identification)
7. [Optimization Workflows](#optimization-workflows)
8. [Case Studies](#case-studies)
9. [Production Profiling](#production-profiling)
10. [Performance Regression Detection](#performance-regression-detection)

---

## Introduction

### Why Profile?

**Don't guess, measure.** Profiling is essential for:

1. **Finding bottlenecks** - Where is time actually spent?
2. **Validating optimizations** - Did your changes help?
3. **Understanding hardware utilization** - Are you using the GPU efficiently?
4. **Catching regressions** - Did recent changes slow things down?

**Common misconceptions**:
- L "I know where the bottleneck is" ’ Profile anyway, you might be surprised
- L "Optimizing this function will help" ’ It might only be 0.1% of runtime
- L "This should be faster" ’ Hardware behavior is complex, measure it

**Real-world example**:
```
Developer: "The matmul is slow, let's optimize it"
Profiler:   "Matmul is 5% of runtime, data loading is 60%"
Result:     Optimizing wrong bottleneck wastes weeks
```

### Profiling Levels

Different tools for different needs:

| Level | Tool | Use Case | Overhead |
|-------|------|----------|----------|
| **System** | Nsight Systems | Timeline, CPU”GPU interaction | ~5% |
| **Kernel** | Nsight Compute | Detailed kernel analysis | ~100x |
| **Python** | PyTorch Profiler | Python + CUDA integration | ~10% |
| **Code** | Manual timers | Quick checks | <1% |

**Rule of thumb**: Start broad (Nsight Systems), then zoom in (Nsight Compute).

---

## Profiling Tools Overview

### NVIDIA Nsight Suite

**Nsight Systems** (`nsys`):
- **Purpose**: System-wide timeline profiling
- **Shows**: CPU activity, GPU kernels, memory transfers, API calls
- **Overhead**: ~5%, suitable for production
- **Use for**: Understanding overall system behavior, finding idle time

**Nsight Compute** (`ncu`):
- **Purpose**: Detailed single-kernel profiling
- **Shows**: All GPU metrics, warp efficiency, memory transactions, occupancy
- **Overhead**: ~100x slower, not for production
- **Use for**: Optimizing specific kernels, understanding bottlenecks

**Visual Profiler** (deprecated):
- Replaced by Nsight Systems + Nsight Compute
- Don't use for new work

### PyTorch Profiler

Built-in PyTorch profiling:
```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
```

**Pros**: Python-aware, easy to use, integrates with TensorBoard
**Cons**: Less detailed than Nsight tools

### When to Use Which Tool

```
Step 1: PyTorch Profiler ’ Identify hot ops in Python
    “
Step 2: Nsight Systems ’ Understand system timeline
    “
Step 3: Nsight Compute ’ Optimize specific kernels
    “
Step 4: Validate ’ Re-profile to confirm improvement
```

---

## NVIDIA Nsight Compute

### Basic Usage

```bash
# Profile single kernel launch
ncu python script.py

# Profile with all metrics (slow but comprehensive)
ncu --set full -o profile python script.py

# Profile specific kernel by name
ncu --kernel-name "matmul_kernel" python script.py

# Profile only first 10 launches of each kernel
ncu --launch-skip 0 --launch-count 10 python script.py
```

### Understanding the Output

**Key sections** in Nsight Compute report:

1. **GPU Speed Of Light (SOL)** - How close to theoretical max?
2. **Memory Workload Analysis** - Memory bandwidth utilization
3. **Compute Workload Analysis** - Compute throughput utilization
4. **Scheduler Statistics** - Thread blocks, warps, occupancy
5. **Warp State Statistics** - What are warps doing?
6. **Instruction Statistics** - FLOPs, memory operations

**Example output**:
```
Kernel: matmul_kernel
Duration: 0.523 ms
Blocks:  1024
Threads: 256

GPU Speed Of Light
  SM %:                     45.2%
  Memory %:                 89.3%   Memory bound!

Memory Workload Analysis
  Global Load:              1850 GB/s (90% of peak)
  Global Store:             45 GB/s
  L2 Hit Rate:              12.5%

Compute Workload Analysis
  FP32 FLOPs:               8.2 TFLOPS (42% of peak)

Conclusion: Memory bound. Optimize memory access patterns.
```

### Important Metrics

#### 1. SM Utilization

**SM % (SOL)** - What percentage of peak compute is achieved?

```
If SM% is low (<50%):
- Check occupancy
- Check warp execution efficiency
- Look for idle time between kernels
```

#### 2. Memory Utilization

**Memory % (SOL)** - What percentage of peak bandwidth is achieved?

```
If Memory% is high (>80%):
- Kernel is memory bound
- Optimize memory access (coalescing, caching, fusion)
- Consider algorithmic changes to increase arithmetic intensity
```

#### 3. Occupancy

**Achieved Occupancy** - What percentage of max warps are active?

```
Occupancy = Active warps / Max possible warps

Low occupancy (<50%):
- Too many registers per thread?
- Too much shared memory per block?
- Too few blocks launched?

High occupancy (>75%) but still slow:
- Occupancy isn't the problem
- Look at memory bandwidth or instruction throughput
```

**Important**: High occupancy doesn't guarantee high performance!

```
Kernel A: 25% occupancy, 1800 GB/s bandwidth ’ Good (memory bound)
Kernel B: 100% occupancy, 400 GB/s bandwidth ’ Bad (memory underutilized)
```

#### 4. Warp Execution Efficiency

**Warp Exec Efficiency** - Percentage of active threads in executed warps

```
Low efficiency (<80%):
- Warp divergence (if statements)
- Unbalanced thread work
- Tail effects (last block doesn't fill all threads)
```

#### 5. Memory Efficiency

**Global Memory Efficiency** - How many bytes are used per transaction?

```
Ideal: 100% (all bytes in cache line are used)
Poor:  <50% (scattered access, poor coalescing)

Metric: dram__bytes_requested / dram__bytes_transferred

Example:
Requested: 4 MB
Transferred: 8 MB
Efficiency: 50%  Wasting half the bandwidth!
```

### Metric Groups

**Full report** includes hundreds of metrics. Focus on these groups:

```bash
# Memory metrics
ncu --metrics dram__bytes.sum,\
             dram__sectors_read.sum,\
             l2_cache_hit_rate,\
             smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
    python script.py

# Compute metrics
ncu --metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
             sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
             smsp__sass_average_inst_executed_per_warp.ratio \
    python script.py

# Occupancy metrics
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
             sm__maximum_warps_per_active_cycle,\
             launch__registers_per_thread \
    python script.py
```

### Roofline Plot in Nsight Compute

Modern Nsight Compute includes roofline analysis:

```bash
ncu --set roofline -o profile python script.py
# Open in GUI: ncu-ui profile.ncu-rep
# View: Roofline chart
```

Shows:
- Kernel position on roofline
- Memory vs compute bound
- Optimization opportunities

### Command-Line Workflow

```bash
# 1. Quick check - which kernels are slow?
ncu --print-kernel-summary python train.py

# 2. Profile slowest kernel
ncu --kernel-name "slow_kernel" --set full -o profile python train.py

# 3. Compare before/after optimization
ncu --kernel-name "slow_kernel" -o baseline python train.py
# ... make changes ...
ncu --kernel-name "slow_kernel" -o optimized python train.py

# 4. Compare in GUI
ncu-ui baseline.ncu-rep optimized.ncu-rep
```

### GUI Workflow

Launch GUI: `ncu-ui`

**Key views**:
1. **Details** - Overview with SOL metrics
2. **Source** - Line-by-line metrics (requires `-lineinfo` compile flag)
3. **Memory Chart** - Memory hierarchy utilization
4. **Scheduler** - Timeline of warp execution
5. **Speed Of Light** - Visual comparison to theoretical peak

**Pro tip**: Compare multiple runs side-by-side in GUI.

---

## NVIDIA Nsight Systems

### Basic Usage

```bash
# Profile entire application
nsys profile -o timeline python train.py

# Profile with statistics
nsys profile --stats=true -o timeline python train.py

# Profile GPU and CPU
nsys profile --trace=cuda,cudnn,cublas,nvtx -o timeline python train.py

# Profile for specific duration
nsys profile --duration=60 -o timeline python train.py
```

### Understanding the Timeline

Open with: `nsys-ui timeline.qdrep`

**Timeline rows**:
- **CUDA HW**: GPU kernel execution (what the GPU is doing)
- **CUDA API**: CPU calls to CUDA runtime (cudaMalloc, cudaMemcpy, kernel launches)
- **Threads**: CPU thread activity
- **NVTX**: Custom markers (added by you)

**What to look for**:

1. **GPU idle time** - Gaps in CUDA HW row
   ```
   CPU Thread: [Launch] ... [compute] ... [Launch]
   CUDA HW:            [Kernel]    [idle gap]    [Kernel]

   Problem: CPU not keeping GPU fed
   Solution: Asynchronous launches, CUDA streams
   ```

2. **Memory transfer bottlenecks**
   ```
   CUDA HW: [H2D copy] [Kernel] [D2H copy] [idle] [H2D copy] ...

   Problem: Synchronous transfers blocking GPU
   Solution: Pinned memory, async transfers, overlap with compute
   ```

3. **Kernel launch overhead**
   ```
   CUDA API: [launch][launch][launch][launch][launch]...

   Problem: Too many small kernel launches
   Solution: Kernel fusion, larger work per kernel
   ```

### NVTX Markers

Add custom markers to your code:

```python
import torch.cuda.nvtx as nvtx

# Mark regions
nvtx.range_push("data_loading")
data = load_batch()
nvtx.range_pop()

nvtx.range_push("forward_pass")
output = model(data)
nvtx.range_pop()

nvtx.range_push("backward_pass")
loss.backward()
nvtx.range_pop()
```

Shows up as colored regions in timeline ’ easy to see where time is spent.

### Multi-GPU Profiling

```bash
# Profile all GPUs
nsys profile --trace=cuda,mpi -o timeline python distributed_train.py

# In timeline:
# - See NCCL communication
# - Identify load imbalance across GPUs
# - Find GPU synchronization issues
```

### Analyzing CUDA Streams

Multiple streams enable overlap:

```python
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    out1 = model1(input1)

with torch.cuda.stream(stream2):
    out2 = model2(input2)

# Both execute concurrently!
```

In Nsight Systems timeline:
- Each stream appears as separate row
- Can see concurrent execution
- Identify stream dependencies

---

## Performance Metrics Deep Dive

### Memory Metrics

#### 1. Global Memory Throughput

```
Achieved Bandwidth = Bytes Transferred / Time

Metric: dram__bytes.sum / duration

Goal: >80% of theoretical peak (1640+ GB/s on A100)
```

**Example**:
```
Kernel duration: 0.5 ms
Bytes transferred: 800 MB
Bandwidth: 800 MB / 0.5 ms = 1600 GB/s (80% of 2048 GB/s) 
```

#### 2. Memory Coalescing

```
Coalescing Efficiency = Bytes Used / Bytes Fetched

Metric: dram__bytes_requested / dram__bytes_transferred

Goal: >90%
```

**Example**:
```
Strided access (stride=32):
  Requested: 4 MB (1M floats)
  Fetched: 128 MB (32x more due to cache line waste)
  Efficiency: 4/128 = 3.1% 

Coalesced access:
  Requested: 4 MB
  Fetched: 4.2 MB (minimal overhead)
  Efficiency: 4/4.2 = 95% 
```

#### 3. Cache Hit Rates

```
L1 Hit Rate = l1tex__t_sectors_hit / l1tex__t_sectors_lookup
L2 Hit Rate = lts__t_sectors_hit / lts__t_sectors_lookup

Higher is better (less global memory traffic)
```

**Typical values**:
```
L1 Cache:
  Streaming access (matmul): 20-40% hit rate
  Reuse patterns: 70-90% hit rate

L2 Cache:
  First access: 0% hit rate
  Temporal reuse: 40-80% hit rate
```

### Compute Metrics

#### 1. FLOPs Achieved

```
FLOPs = Count of FP operations

Metrics:
  sm__sass_thread_inst_executed_op_fadd_pred_on.sum  (adds)
  sm__sass_thread_inst_executed_op_fmul_pred_on.sum  (multiplies)
  sm__sass_thread_inst_executed_op_ffma_pred_on.sum  (FMAs, count as 2)

Total FLOPs = adds + multiplies + (2 × FMAs)
```

**Example**:
```
Adds: 1M
Multiplies: 1M
FMAs: 10M
Total FLOPs: 1M + 1M + (2 × 10M) = 22M FLOPs

Duration: 0.01 ms
TFLOPS: 22M / 0.01ms = 2.2 TFLOPS
```

#### 2. Arithmetic Intensity

```
AI = FLOPs / Bytes Transferred

Compare to ridge point to determine if memory or compute bound
```

**Example (A100 FP32)**:
```
Ridge Point = 19.5 TFLOPS / 2 TB/s = 9.75 FLOPs/byte

Vector Add:
  AI = 0.083 FLOPs/byte < 9.75 ’ Memory bound

MatMul (N=2048):
  AI = 682 FLOPs/byte > 9.75 ’ Compute bound
```

### Occupancy Metrics

#### 1. Theoretical vs Achieved Occupancy

```
Theoretical Occupancy: Based on resource usage
  - Calculated from registers, shared memory, threads per block

Achieved Occupancy: Measured during execution
  - Can be lower due to:
    - Warp divergence
    - Memory latency not hidden
    - Unbalanced workload
```

**Calculation**:
```python
def theoretical_occupancy(threads_per_block, regs_per_thread, shared_mem_per_block):
    # A100 limits
    MAX_THREADS_PER_SM = 2048
    MAX_BLOCKS_PER_SM = 32
    MAX_REGS_PER_SM = 65536
    MAX_SHARED_MEM_PER_SM = 164 * 1024  # 164 KB

    # Calculate max blocks limited by each resource
    blocks_by_threads = MAX_THREADS_PER_SM // threads_per_block
    blocks_by_regs = MAX_REGS_PER_SM // (regs_per_thread * threads_per_block)
    blocks_by_shared = MAX_SHARED_MEM_PER_SM // shared_mem_per_block

    blocks_per_sm = min(blocks_by_threads, blocks_by_regs, blocks_by_shared, MAX_BLOCKS_PER_SM)
    active_warps = blocks_per_sm * (threads_per_block // 32)
    max_warps = MAX_THREADS_PER_SM // 32  # 64 warps

    return active_warps / max_warps

# Example
threads = 256  # 8 warps
regs = 48
shared = 8 * 1024  # 8 KB

occ = theoretical_occupancy(threads, regs, shared)
print(f"Occupancy: {occ:.1%}")  # 100%
```

#### 2. Warp Stall Reasons

Nsight Compute shows why warps are stalled:

```
Stall Reasons:
  Memory Throttle:     45%   Waiting for memory
  Execution Dependency: 15%   Waiting for ALU
  Synchronization:     8%    Waiting for __syncthreads
  Not Selected:        32%   Other warps running

If Memory Throttle is high ’ Memory bound
If Execution Dependency high ’ Compute bound
```

---

## Bottleneck Identification

### The Profiling Workflow

```
1. Measure baseline performance
2. Identify bottleneck
3. Hypothesize cause
4. Optimize
5. Measure again
6. Validate improvement
7. Repeat until target met
```

### Common Bottlenecks and Solutions

#### 1. Low Memory Bandwidth

**Symptoms**:
- Memory % (SOL) > 80%
- SM % (SOL) < 50%
- High stalls on memory throttle

**Solutions**:
```
 Kernel fusion (reduce memory transfers)
 Coalesced access patterns
 Increase arithmetic intensity
 Use shared memory for reuse
 Vectorized loads (float4)
```

**Example**:
```cuda
// Before: 3 kernels, 6 memory transfers
kernel1<<<>>>(); // load x, store x
kernel2<<<>>>(); // load x, store x
kernel3<<<>>>(); // load x, store x

// After: 1 kernel, 2 memory transfers
fused_kernel<<<>>>(); // load x, store x
// 3x speedup!
```

#### 2. Low Compute Utilization

**Symptoms**:
- SM % (SOL) < 50%
- Memory % (SOL) < 50%
- Low occupancy or warp efficiency

**Solutions**:
```
 Increase threads per block
 Reduce register usage
 Launch more blocks
 Eliminate warp divergence
 Balance work across threads
```

**Example**:
```cuda
// Before: 64 threads per block (2 warps), 50% occupancy
kernel<<<numBlocks, 64>>>();

// After: 256 threads per block (8 warps), 100% occupancy
kernel<<<numBlocks, 256>>>();
// 2x speedup (better latency hiding)
```

#### 3. Warp Divergence

**Symptoms**:
- Warp execution efficiency < 80%
- High variation in thread execution time
- Nsight Compute shows divergent branches

**Solutions**:
```
 Restructure conditionals (warp-aligned)
 Sort data by condition
 Use __ballot_sync to handle divergence explicitly
```

**Example**:
```cuda
// Before: 50% warp efficiency
if (threadIdx.x % 2 == 0) {
    // Even threads
} else {
    // Odd threads
}
// Both paths execute serially

// After: 100% warp efficiency
if (threadIdx.x < 16) {
    // First half warp
} else {
    // Second half warp
}
// Warps remain coherent
```

#### 4. Launch Overhead

**Symptoms**:
- Nsight Systems shows many small kernel launches
- CPU time high in CUDA API calls
- GPU idle between kernels

**Solutions**:
```
 Kernel fusion
 CUDA streams for overlap
 Asynchronous launches
 Larger work per kernel
```

**Example**:
```python
# Before: 1000 small kernels (sequential)
for i in range(1000):
    small_kernel(data[i])
# Total: 150 ms (100 ms overhead + 50 ms compute)

# After: 1 large kernel or batched
batched_kernel(data)  # Process all at once
# Total: 55 ms (5 ms overhead + 50 ms compute)
# 2.7x speedup!
```

#### 5. Memory Transfer Bottleneck

**Symptoms**:
- Nsight Systems shows large H2D or D2H transfers
- GPU idle during transfers
- cudaMemcpy in critical path

**Solutions**:
```
 Pinned memory (2-3x faster)
 Async transfers with compute overlap
 Keep data on GPU (avoid round-trips)
 Use CUDA streams
```

**Example**:
```python
# Before: Synchronous, no overlap
cudaMemcpy(d_data, h_data, size, H2D)  # 50 ms, GPU idle
kernel()                                # 30 ms
cudaMemcpy(h_result, d_result, size, D2H)  # 50 ms, GPU idle
# Total: 130 ms

# After: Async + pinned + overlap
cudaMemcpyAsync(d_data, h_data_pinned, size, H2D, stream1)
cudaMemcpyAsync(d_data2, h_data2_pinned, size, H2D, stream2)
# GPU starts computing as soon as data arrives
kernel(stream1)  # Overlap with stream2 transfer
kernel(stream2)  # Overlap with next transfer
# Total: 65 ms (2x speedup!)
```

---

## Optimization Workflows

### Optimization Priority Order

Follow this order for maximum impact:

1. **Algorithm** (10-1000x) - Is there a better algorithm?
2. **Kernel Fusion** (2-10x) - Combine operations
3. **Memory Access** (2-5x) - Coalescing, caching
4. **Occupancy** (1.2-2x) - If it's low and fixable
5. **Instruction Optimization** (1.1-1.5x) - Fast math, loop unrolling

**Example priority**:
```
Baseline: 100 ms

1. Switch from naive to tiled matmul ’ 50 ms (2x)
2. Fuse matmul + bias + relu ’ 35 ms (1.4x)
3. Fix coalescing ’ 25 ms (1.4x)
4. Increase occupancy 50% ’ 75% ’ 22 ms (1.1x)
5. Use fast math ’ 20 ms (1.1x)

Total: 5x speedup
```

### The Optimize-Measure-Validate Loop

```bash
# 1. Profile baseline
ncu --set full -o baseline python script.py

# 2. Identify bottleneck from baseline report
# Example: Memory bound, poor coalescing

# 3. Implement fix
# ... edit code ...

# 4. Profile optimized version
ncu --set full -o optimized python script.py

# 5. Compare
ncu-ui baseline.ncu-rep optimized.ncu-rep

# 6. Validate improvement
# - Bandwidth increased? 
# - Still correct results? 
# - No new bottlenecks? 

# 7. Repeat for next bottleneck
```

### Optimization Checklist

Before declaring a kernel "optimized", check:

- [ ] **Memory bandwidth**: >80% of theoretical peak
- [ ] **Compute utilization**: >70% for compute-bound kernels
- [ ] **Occupancy**: >75% (if memory/compute-bound)
- [ ] **Coalescing**: >90% memory efficiency
- [ ] **Warp efficiency**: >95%
- [ ] **Bank conflicts**: None (1.0 transactions/request)
- [ ] **Results**: Still correct (validate!)

---

## Case Studies

### Case Study 1: Optimizing LayerNorm

**Initial implementation**: 3-pass (mean, variance, normalize)

**Profile baseline**:
```
Duration: 1.20 ms
Memory %: 85% (memory bound)
SM %: 42%
Kernel launches: 3
Conclusion: Too many passes, memory bound
```

**Optimization 1**: Single-pass with Welford's algorithm
```
Duration: 0.62 ms (1.9x speedup)
Memory %: 88%
Kernel launches: 1
```

**Optimization 2**: Fuse with residual connection
```cuda
// LayerNorm(x + residual) in one kernel
Duration: 0.31 ms (3.9x total speedup)
Memory %: 92%
```

**Profile final**:
```bash
ncu --set full -o final python layernorm_test.py

Results:
  Memory bandwidth: 1882 GB/s (92% of peak) 
  Coalescing: 96% 
  Occupancy: 100% 
  Within 10% of PyTorch native 
```

### Case Study 2: Flash Attention

**Standard attention**: Materializes N×N scores

**Profile standard** (N=2048, 12 heads):
```
Duration: 4.50 ms
Memory: 192 MB allocated
Memory %: 65%
Bottleneck: Memory allocation + N² writes
```

**Flash Attention**: Tiled with online softmax

**Profile Flash** (N=2048):
```
Duration: 0.95 ms (4.7x speedup)
Memory: 8 MB allocated (24x less)
Memory %: 82%
Computation increased but memory saved
```

**Bottleneck shift**:
```
Standard: Memory bound by N² storage
Flash: More compute (recomputing scores), less memory

Trade-off: 2x more FLOPs for 24x less memory
Result: 4.7x faster (memory was bottleneck)
```

**Enables long context**:
```
N=8192, Standard attention: OOM (out of memory)
N=8192, Flash attention: 6.2 ms 
```

### Case Study 3: GELU Optimization

**Exact GELU**: Uses `erf()` function

**Profile exact**:
```
Duration: 0.85 ms (1M elements)
SM %: 78% (compute bound)
Memory %: 32%
Bottleneck: Expensive `erf()` function
```

**Optimization**: Fast approximation with `__tanf`

```cuda
// Before
float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
}

// After
float gelu_fast(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + __tanf(0.7978845608f * (x + 0.044715f * x3)));
}
```

**Profile fast**:
```
Duration: 0.12 ms (7.1x speedup)
SM %: 85%
Memory %: 82%
Max error: 9.1e-5 (acceptable for ML)
```

**Validation**:
```python
# Test accuracy
x = torch.linspace(-3, 3, 10000)
exact = torch.nn.functional.gelu(x)
fast = gelu_fast_cuda(x)

max_error = (exact - fast).abs().max()
print(f"Max error: {max_error:.2e}")  # 9.1e-5

# Test in training
model_exact = train(gelu='exact')  # Acc: 94.2%
model_fast = train(gelu='fast')    # Acc: 94.1%
# Negligible difference, 7x faster!
```

---

## Production Profiling

### Challenges

Profiling in production is different:

1. **Can't slow down** - Nsight Compute 100x overhead unacceptable
2. **Need continuous monitoring** - Not one-time analysis
3. **Representative workloads** - Must profile real data
4. **Minimal overhead** - <5% is acceptable

### Lightweight Profiling

**Use Nsight Systems** (not Nsight Compute):
```bash
# Production profiling (5% overhead)
nsys profile --stats=true --duration=60 -o prod_profile python serve.py

# Analyze offline
nsys stats prod_profile.qdrep

# Look for:
# - GPU utilization over time
# - Kernel launch patterns
# - Memory transfer patterns
```

**Manual timers** for specific sections:
```python
import time
import torch

class PerformanceMonitor:
    def __init__(self):
        self.timings = {}

    def time_section(self, name):
        torch.cuda.synchronize()
        start = time.time()

        yield

        torch.cuda.synchronize()
        elapsed = time.time() - start

        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)

    def report(self):
        for name, times in self.timings.items():
            avg = sum(times) / len(times)
            print(f"{name}: {avg*1000:.2f} ms average")

# Usage
monitor = PerformanceMonitor()

with monitor.time_section("data_loading"):
    batch = load_data()

with monitor.time_section("forward"):
    output = model(batch)

with monitor.time_section("backward"):
    loss.backward()

monitor.report()
```

### CUDA Events for Accurate Timing

```python
def benchmark_kernel(func, *args, num_iters=100, warmup=10):
    """Accurate GPU timing using CUDA events"""

    # Warmup
    for _ in range(warmup):
        func(*args)

    # Time with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        func(*args)
    end.record()

    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)  # milliseconds

    return elapsed / num_iters

# Usage
time_ms = benchmark_kernel(custom_gelu, x)
print(f"Average time: {time_ms:.3f} ms")
```

### Production Metrics

Track these metrics in production:

```python
class ProductionMetrics:
    def __init__(self):
        self.metrics = {
            'throughput': [],      # samples/sec
            'latency': [],         # ms per batch
            'gpu_utilization': [], # %
            'memory_used': [],     # GB
        }

    def record(self, batch_size, time_ms):
        throughput = batch_size / (time_ms / 1000)
        self.metrics['throughput'].append(throughput)
        self.metrics['latency'].append(time_ms)
        self.metrics['gpu_utilization'].append(torch.cuda.utilization())
        self.metrics['memory_used'].append(torch.cuda.memory_allocated() / 1e9)

    def alert_if_degraded(self, baseline_throughput):
        current = sum(self.metrics['throughput'][-100:]) / 100
        if current < 0.9 * baseline_throughput:
            alert(f"Performance degradation: {current} < {0.9*baseline_throughput}")

# Log to monitoring system (Prometheus, Grafana, etc.)
```

---

## Performance Regression Detection

### Why Regressions Happen

Common causes:
1. **Code changes** - New feature slows down hot path
2. **Dependency updates** - PyTorch/CUDA version change
3. **Model changes** - Larger model, different architecture
4. **Data changes** - Different input shapes or distributions
5. **Hardware changes** - Different GPU model

### Automated Regression Testing

```python
# tests/test_performance_regressions.py

import pytest
import torch
from my_kernels import custom_gelu

BASELINE_TIMES = {
    'custom_gelu_1M': 0.12,  # ms
    'layernorm_128_768': 0.31,
    'flash_attention_2048': 0.95,
}

TOLERANCE = 1.10  # Allow 10% slowdown

@pytest.mark.parametrize("size", [1000000])
def test_gelu_performance(size, benchmark):
    x = torch.randn(size, device='cuda')

    # Benchmark with pytest-benchmark
    result = benchmark.pedantic(
        custom_gelu, args=(x,),
        iterations=100, rounds=10
    )

    time_ms = result.mean * 1000
    baseline = BASELINE_TIMES['custom_gelu_1M']

    assert time_ms < baseline * TOLERANCE, \
        f"Regression: {time_ms:.3f}ms > {baseline * TOLERANCE:.3f}ms"

# Run with: pytest tests/test_performance_regressions.py -v
```

### Continuous Performance Monitoring

```bash
# .github/workflows/performance.yml

name: Performance Tests

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]

    steps:
      - uses: actions/checkout@v2

      - name: Run benchmarks
        run: |
          pytest tests/test_performance_regressions.py --benchmark-json=output.json

      - name: Compare to baseline
        run: |
          python scripts/compare_benchmarks.py output.json baseline.json

      - name: Comment on PR if regression
        if: github.event_name == 'pull_request'
        run: |
          python scripts/post_perf_comment.py
```

### Profiling CI/CD

**Best practice**: Profile before deploying

```bash
# deploy.sh

echo "Profiling production build..."
nsys profile --stats=true --duration=60 -o deploy_profile python serve.py &
PID=$!

# Let it run for 60 seconds
wait $PID

# Analyze
nsys stats deploy_profile.qdrep > deploy_stats.txt

# Check GPU utilization
GPU_UTIL=$(grep "GPU Utilization" deploy_stats.txt | awk '{print $3}')
if [ "$GPU_UTIL" -lt 80 ]; then
    echo "Warning: Low GPU utilization ($GPU_UTIL%)"
    exit 1
fi

echo "Profile looks good, deploying..."
```

---

## Summary

### Key Takeaways

1. **Profile before optimizing** - Don't guess bottlenecks
2. **Use the right tool**:
   - Nsight Systems: System-wide timeline
   - Nsight Compute: Kernel-level optimization
   - PyTorch Profiler: Python integration
3. **Focus on hot spots** - 80/20 rule applies
4. **Validate changes** - Always measure before/after
5. **Memory is often the bottleneck** - Not compute
6. **High occupancy ` fast** - It's just one factor

### Profiling Workflow

```
1. Baseline measurement (Nsight Systems)
2. Identify slowest operations
3. Deep dive on hot kernels (Nsight Compute)
4. Hypothesize bottleneck
5. Implement optimization
6. Profile again
7. Validate improvement
8. Repeat
```

### Common Optimizations by Bottleneck

| Bottleneck | Solution | Expected Gain |
|------------|----------|---------------|
| Memory bandwidth | Kernel fusion | 3-5x |
| Low occupancy | More threads/block | 1.5-2x |
| Warp divergence | Restructure conditionals | 1.5-2x |
| Launch overhead | Fewer larger kernels | 2-3x |
| Coalescing | Fix access patterns | 2-10x |
| Slow math | Fast intrinsics | 5-7x |

### Tools Quick Reference

```bash
# Nsight Compute
ncu --set full -o profile python script.py
ncu-ui profile.ncu-rep

# Nsight Systems
nsys profile --stats=true -o timeline python script.py
nsys-ui timeline.qdrep

# PyTorch Profiler
# (in Python code)
with torch.profiler.profile() as prof:
    model(input)
prof.export_chrome_trace("trace.json")
```

---

*Module 3, Lecture 1: Advanced Performance Profiling*
*Last updated: 2025-11-02*
