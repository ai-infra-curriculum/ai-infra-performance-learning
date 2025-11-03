# Module 3 Exercises: Performance Profiling and Optimization

These exercises provide hands-on experience with production GPU profiling tools, performance analysis techniques, and systematic optimization workflows used in real ML infrastructure.

## Prerequisites

- Completed Module 1 (Performance Fundamentals) and Module 2 (CUDA Programming)
- NVIDIA GPU with CUDA Toolkit 12.0+
- NVIDIA Nsight Compute and Nsight Systems installed
- PyTorch 2.0+, Python 3.10+
- Understanding of custom CUDA kernels
- Basic knowledge of performance metrics

## Setup

```bash
# Install profiling tools (if not already installed)
# Nsight Compute and Nsight Systems are included with CUDA Toolkit
# Verify installation
ncu --version
nsys --version

# Install Python dependencies
pip install torch torchvision pytest pytest-benchmark matplotlib pandas seaborn

# Create exercise workspace
mkdir -p mod-003-exercises
cd mod-003-exercises
```

## Learning Objectives

By completing these exercises, you will:

1. Master NVIDIA Nsight Compute for kernel-level profiling
2. Use Nsight Systems for system-wide performance analysis
3. Interpret performance metrics and identify bottlenecks
4. Apply systematic optimization workflows
5. Implement production profiling with minimal overhead
6. Build automated performance regression testing

## Exercise Overview

| Exercise | Topic | Time | Difficulty |
|----------|-------|------|------------|
| 1 | Nsight Compute Deep Dive | 4 hours | Intermediate |
| 2 | Nsight Systems Analysis | 4 hours | Intermediate |
| 3 | Bottleneck Identification | 5 hours | Advanced |
| 4 | End-to-End Optimization | 8 hours | Advanced |
| 5 | Production Profiling | 4 hours | Advanced |
| 6 | Regression Testing | 4 hours | Intermediate |

**Total Time**: ~29 hours

---

## Exercise 1: Nsight Compute Deep Dive (4 hours)

### Objective

Master NVIDIA Nsight Compute for detailed kernel profiling, metrics interpretation, and performance analysis.

### Background

Nsight Compute provides kernel-level profiling with detailed metrics about compute, memory, and execution characteristics. Understanding these metrics is essential for identifying optimization opportunities.

### Tasks

#### Part 1: Basic Profiling (1 hour)

Profile a set of reference kernels and understand the basic Nsight Compute workflow.

```python
# reference_kernels.py
import torch
import torch.nn.functional as F

def profile_matmul():
    """Matrix multiplication - compute bound"""
    A = torch.randn(2048, 2048, device='cuda')
    B = torch.randn(2048, 2048, device='cuda')

    for _ in range(10):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

def profile_elementwise():
    """Elementwise operations - memory bound"""
    x = torch.randn(10_000_000, device='cuda')

    for _ in range(10):
        y = torch.relu(x)
    torch.cuda.synchronize()

def profile_reduction():
    """Reduction operations - mixed"""
    x = torch.randn(10_000_000, device='cuda')

    for _ in range(10):
        s = x.sum()
    torch.cuda.synchronize()

if __name__ == '__main__':
    import sys

    if sys.argv[1] == 'matmul':
        profile_matmul()
    elif sys.argv[1] == 'elementwise':
        profile_elementwise()
    elif sys.argv[1] == 'reduction':
        profile_reduction()
```

**Profile each kernel:**

```bash
# Profile matmul (compute bound)
ncu --set full -o matmul_profile python reference_kernels.py matmul

# Profile elementwise (memory bound)
ncu --set full -o elementwise_profile python reference_kernels.py elementwise

# Profile reduction (mixed)
ncu --set full -o reduction_profile python reference_kernels.py reduction

# Open in UI
ncu-ui matmul_profile.ncu-rep
```

**Analyze:**
- Identify compute-bound vs memory-bound patterns
- Find Speed of Light (SOL) percentages for compute and memory
- Examine occupancy and warp execution efficiency
- Compare actual vs theoretical bandwidth/throughput

#### Part 2: Custom Kernel Analysis (1.5 hours)

Implement and profile three versions of a GELU activation kernel with increasing optimization.

```cuda
// gelu_kernels.cu
#include <cuda_runtime.h>
#include <math.h>

// Version 1: Naive implementation
__global__ void gelu_naive(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x3)));
    }
}

// Version 2: Fast math
__global__ void gelu_fast(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        output[idx] = 0.5f * x * (1.0f + __tanf(0.7978845608f * (x + 0.044715f * x3)));
    }
}

// Version 3: Vectorized with fast math
__global__ void gelu_vectorized(const float4* input, float4* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 x = input[idx];

        float x1_3 = x.x * x.x * x.x;
        float x2_3 = x.y * x.y * x.y;
        float x3_3 = x.z * x.z * x.z;
        float x4_3 = x.w * x.w * x.w;

        output[idx] = make_float4(
            0.5f * x.x * (1.0f + __tanf(0.7978845608f * (x.x + 0.044715f * x1_3))),
            0.5f * x.y * (1.0f + __tanf(0.7978845608f * (x.y + 0.044715f * x2_3))),
            0.5f * x.z * (1.0f + __tanf(0.7978845608f * (x.z + 0.044715f * x3_3))),
            0.5f * x.w * (1.0f + __tanf(0.7978845608f * (x.w + 0.044715f * x4_3)))
        );
    }
}
```

**Profile and compare:**

```bash
# Profile each version
ncu --set full -o gelu_naive_profile python gelu_test.py naive
ncu --set full -o gelu_fast_profile python gelu_test.py fast
ncu --set full -o gelu_vectorized_profile python gelu_test.py vectorized

# Compare side-by-side in UI
ncu-ui gelu_naive_profile.ncu-rep gelu_fast_profile.ncu-rep gelu_vectorized_profile.ncu-rep
```

**Analysis questions:**
1. What is the compute throughput (FLOP/s) for each version?
2. What percentage of peak compute and memory bandwidth is achieved?
3. How does instruction throughput change with fast math?
4. What is the memory coalescing efficiency?
5. How does vectorization affect memory transactions?
6. What are the warp execution efficiency differences?

#### Part 3: Metrics Deep Dive (1.5 hours)

Analyze specific performance metrics for a custom LayerNorm kernel.

```cuda
// layernorm_kernel.cu
__global__ void layernorm_fused(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int N, int D, float eps = 1e-5f
) {
    int row = blockIdx.x;
    const float* x = input + row * D;
    float* y = output + row * D;

    // Welford's algorithm for mean and variance
    float mean = 0.0f, m2 = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = x[i];
        float delta = val - mean;
        mean += delta / (i + 1);
        float delta2 = val - mean;
        m2 += delta * delta2;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        mean += __shfl_down_sync(0xffffffff, mean, offset);
        m2 += __shfl_down_sync(0xffffffff, m2, offset);
    }

    // Broadcast mean and variance
    mean = __shfl_sync(0xffffffff, mean, 0);
    m2 = __shfl_sync(0xffffffff, m2, 0);
    float var = m2 / D;
    float inv_std = rsqrtf(var + eps);

    // Normalize
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float normalized = (x[i] - mean) * inv_std;
        y[i] = normalized * gamma[i] + beta[i];
    }
}
```

**Focus metrics to analyze:**

1. **Memory Metrics:**
   - Global Memory Load/Store Efficiency
   - L2 Cache Hit Rate
   - Memory Throughput vs Peak
   - Sector Efficiency (coalescing)

2. **Compute Metrics:**
   - SM Efficiency
   - Instruction Mix (int, fp32, fp64, special)
   - Warp Execution Efficiency
   - Achieved Occupancy

3. **Warp Metrics:**
   - Warp Stall Reasons (breakdown %)
   - Branch Efficiency
   - Predicated Instructions
   - Divergent Branches

4. **Launch Configuration:**
   - Blocks per SM
   - Threads per Block
   - Registers per Thread
   - Shared Memory per Block

**Deliverables:**

1. Report document (`nsight_compute_analysis.md`) containing:
   - Performance comparison table for all kernels
   - Detailed metrics analysis with screenshots
   - Identification of bottlenecks for each kernel
   - Specific optimization recommendations

2. Metrics extraction script:
```python
# extract_metrics.py
import pandas as pd
import json

def extract_ncu_metrics(ncu_rep_file):
    """Extract key metrics from Nsight Compute report"""
    # Use ncu CLI to extract metrics
    import subprocess

    result = subprocess.run([
        'ncu', '--import', ncu_rep_file,
        '--csv'
    ], capture_output=True, text=True)

    # Parse CSV and extract key metrics
    metrics = {
        'kernel_name': '',
        'duration_ms': 0.0,
        'compute_sol': 0.0,
        'memory_sol': 0.0,
        'occupancy': 0.0,
        'l2_hit_rate': 0.0,
        'warp_efficiency': 0.0
    }

    return metrics
```

3. Comparison visualizations showing SOL percentages, throughput, and efficiency metrics

**Expected Results:**
- MatMul: >80% Compute SOL, <30% Memory SOL (compute-bound)
- Elementwise: <20% Compute SOL, >70% Memory SOL (memory-bound)
- GELU Fast: 5-7x faster than naive, >50% special function throughput
- GELU Vectorized: 2x fewer memory transactions, >90% coalescing efficiency
- LayerNorm: >75% occupancy, <10% warp stalls on memory

---

## Exercise 2: Nsight Systems Timeline Analysis (4 hours)

### Objective

Use NVIDIA Nsight Systems to analyze system-wide performance, identify CPU/GPU synchronization issues, and optimize end-to-end training pipelines.

### Background

Nsight Systems provides a timeline view of CPU, GPU, and framework activity with low overhead (~5%), making it suitable for profiling entire training runs.

### Tasks

#### Part 1: Basic Timeline Profiling (1 hour)

Profile a simple training loop and understand the timeline visualization.

```python
# train_profile.py
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024)
        )

    def forward(self, x):
        return self.layers(x)

def train_step(model, x, y, optimizer):
    optimizer.zero_grad()
    output = model(x)
    loss = nn.functional.mse_loss(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_loop_baseline():
    """Baseline: no optimization"""
    model = SimpleModel().cuda()
    optimizer = optim.Adam(model.parameters())

    for i in range(100):
        x = torch.randn(256, 1024, device='cuda')
        y = torch.randn(256, 1024, device='cuda')
        loss = train_step(model, x, y, optimizer)

        # Synchronous loss logging
        print(f"Step {i}, Loss: {loss}")

def train_loop_optimized():
    """Optimized: async operations, batching"""
    model = SimpleModel().cuda()
    optimizer = optim.Adam(model.parameters())

    # Pre-allocate tensors
    x = torch.randn(256, 1024, device='cuda')
    y = torch.randn(256, 1024, device='cuda')

    losses = []

    for i in range(100):
        # Reuse tensors
        x.normal_()
        y.normal_()

        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()

        # Async loss collection
        losses.append(loss.detach())

    # Batch logging at end
    torch.cuda.synchronize()
    for i, loss in enumerate(losses):
        print(f"Step {i}, Loss: {loss.item()}")

if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'baseline':
        train_loop_baseline()
    elif sys.argv[1] == 'optimized':
        train_loop_optimized()
```

**Profile both versions:**

```bash
# Baseline profile
nsys profile --stats=true -o baseline_timeline \
    --capture-range=cudaProfilerApi \
    python train_profile.py baseline

# Optimized profile
nsys profile --stats=true -o optimized_timeline \
    --capture-range=cudaProfilerApi \
    python train_profile.py optimized

# Open in UI
nsys-ui baseline_timeline.qdrep
nsys-ui optimized_timeline.qdrep
```

**Analyze:**
- Identify GPU idle time (white gaps in CUDA timeline)
- Find CPU-GPU synchronization points
- Measure kernel launch overhead
- Compare iteration time (baseline vs optimized)

#### Part 2: DataLoader Optimization (1.5 hours)

Profile and optimize data loading bottlenecks in a training pipeline.

```python
# dataloader_profile.py
import torch
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np

class SlowDataset(Dataset):
    """Simulates slow data loading"""
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate expensive preprocessing
        time.sleep(0.01)
        x = torch.randn(3, 224, 224)
        y = torch.randint(0, 1000, (1,))
        return x, y

class SimpleModelCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 1000)
        )

    def forward(self, x):
        return self.conv(x)

def train_with_dataloader(num_workers=0, pin_memory=False, prefetch_factor=2):
    """Train with configurable dataloader settings"""
    dataset = SlowDataset(size=100)
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    model = SimpleModelCNN().cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(2):
        for batch_idx, (x, y) in enumerate(loader):
            x = x.cuda(non_blocking=pin_memory)
            y = y.cuda(non_blocking=pin_memory)

            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y.squeeze())
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    import sys
    config = sys.argv[1]

    if config == 'slow':
        train_with_dataloader(num_workers=0)
    elif config == 'fast':
        train_with_dataloader(num_workers=4, pin_memory=True, prefetch_factor=4)
```

**Profile configurations:**

```bash
# Slow configuration
nsys profile --stats=true -o dataloader_slow \
    --trace=cuda,nvtx,osrt \
    python dataloader_profile.py slow

# Fast configuration
nsys profile --stats=true -o dataloader_fast \
    --trace=cuda,nvtx,osrt \
    python dataloader_profile.py fast
```

**Analysis tasks:**
1. Identify data loading as bottleneck in slow version
2. Measure GPU utilization (% time GPU busy)
3. Visualize overlapping data loading and compute in fast version
4. Quantify speedup from num_workers and pin_memory

#### Part 3: Advanced Analysis (1.5 hours)

Analyze a complete transformer training step with attention to:
- Kernel fusion opportunities
- Memory copy overhead
- Optimizer step analysis
- Gradient synchronization (if using DDP)

```python
# transformer_profile.py
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # Attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x

def profile_transformer_step():
    """Profile a single transformer training step"""
    model = nn.Sequential(*[
        TransformerBlock() for _ in range(6)
    ]).cuda()

    optimizer = torch.optim.Adam(model.parameters())

    # Input
    x = torch.randn(32, 128, 512, device='cuda')
    target = torch.randn(32, 128, 512, device='cuda')

    # Training step
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

if __name__ == '__main__':
    profile_transformer_step()
```

**Profile with detailed tracing:**

```bash
nsys profile --stats=true -o transformer_detailed \
    --trace=cuda,nvtx,cublas,cudnn \
    --cuda-memory-usage=true \
    --gpu-metrics-device=0 \
    python transformer_profile.py

# Generate detailed report
nsys stats transformer_detailed.qdrep \
    --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum
```

**Deliverables:**

1. Timeline analysis report (`nsight_systems_analysis.md`) with:
   - Before/after timeline screenshots
   - GPU utilization graphs
   - Bottleneck identification
   - Optimization recommendations

2. Automated analysis script:
```python
# analyze_timeline.py
import json
import subprocess

def analyze_nsys_report(qdrep_file):
    """Extract statistics from Nsight Systems report"""
    result = subprocess.run([
        'nsys', 'stats', qdrep_file,
        '--report', 'cuda_gpu_kern_sum',
        '--format', 'json'
    ], capture_output=True, text=True)

    data = json.loads(result.stdout)

    metrics = {
        'total_kernel_time_ms': 0.0,
        'gpu_utilization': 0.0,
        'top_5_kernels': [],
        'num_launches': 0
    }

    return metrics
```

3. Performance comparison table showing iteration time, GPU utilization, and memory throughput

**Expected Results:**
- Baseline: 40-60% GPU utilization, frequent synchronization
- Optimized: >90% GPU utilization, overlapped data loading
- DataLoader: 3-5x speedup with proper configuration
- Transformer: Identify 10-20 optimization opportunities

---

## Exercise 3: Bottleneck Identification and Resolution (5 hours)

### Objective

Develop systematic skills for identifying performance bottlenecks and applying appropriate optimization strategies.

### Background

Real-world performance optimization requires methodically identifying the most impactful bottlenecks and applying targeted fixes. This exercise simulates a production optimization workflow.

### Tasks

#### Part 1: Bottleneck Taxonomy (1 hour)

Given a set of pre-profiled kernels, classify each by bottleneck type and recommend optimizations.

**Kernel profiles provided:**

1. **Kernel A**: Attention QK matmul
   - Compute SOL: 85%
   - Memory SOL: 30%
   - Occupancy: 92%
   - L2 Hit Rate: 45%

2. **Kernel B**: Element-wise addition + ReLU (unfused)
   - Compute SOL: 15%
   - Memory SOL: 75%
   - Occupancy: 60%
   - Memory Transactions: 2x theoretical minimum

3. **Kernel C**: Custom reduction
   - Compute SOL: 35%
   - Memory SOL: 40%
   - Occupancy: 25%
   - Warp Stalls (Memory): 45%

4. **Kernel D**: Softmax
   - Compute SOL: 40%
   - Memory SOL: 65%
   - Occupancy: 80%
   - Warp Stalls (Sync): 30%

5. **Kernel E**: Embedding lookup
   - Compute SOL: 10%
   - Memory SOL: 55%
   - Occupancy: 95%
   - L2 Hit Rate: 15%
   - Memory Access Pattern: Irregular

**For each kernel:**
1. Identify primary bottleneck
2. Explain the metrics that reveal the bottleneck
3. Propose 2-3 specific optimizations
4. Estimate potential speedup

#### Part 2: Hands-On Optimization (3 hours)

Optimize a deliberately inefficient transformer attention implementation through systematic profiling and improvements.

**Starting implementation:**

```cuda
// attention_slow.cu
__global__ void attention_qk_slow(
    const float* Q,  // [batch, heads, seq_len, head_dim]
    const float* K,
    float* scores,   // [batch, heads, seq_len, seq_len]
    int batch, int heads, int seq_len, int head_dim
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // query position

    if (i < seq_len) {
        for (int j = 0; j < seq_len; j++) {  // key position
            float score = 0.0f;

            // Compute dot product Q[i] · K[j]
            for (int d = 0; d < head_dim; d++) {
                int q_idx = ((b * heads + h) * seq_len + i) * head_dim + d;
                int k_idx = ((b * heads + h) * seq_len + j) * head_dim + d;
                score += Q[q_idx] * K[k_idx];
            }

            int out_idx = ((b * heads + h) * seq_len + i) * seq_len + j;
            scores[out_idx] = score / sqrtf((float)head_dim);
        }
    }
}

__global__ void softmax_slow(
    float* scores,  // [batch, heads, seq_len, seq_len]
    int batch, int heads, int seq_len
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // query position

    if (i < seq_len) {
        // Find max for numerical stability
        float max_val = -INFINITY;
        for (int j = 0; j < seq_len; j++) {
            int idx = ((b * heads + h) * seq_len + i) * seq_len + j;
            max_val = fmaxf(max_val, scores[idx]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            int idx = ((b * heads + h) * seq_len + i) * seq_len + j;
            scores[idx] = expf(scores[idx] - max_val);
            sum += scores[idx];
        }

        // Normalize
        for (int j = 0; j < seq_len; j++) {
            int idx = ((b * heads + h) * seq_len + i) * seq_len + j;
            scores[idx] /= sum;
        }
    }
}

__global__ void attention_v_slow(
    const float* scores,  // [batch, heads, seq_len, seq_len]
    const float* V,       // [batch, heads, seq_len, head_dim]
    float* output,        // [batch, heads, seq_len, head_dim]
    int batch, int heads, int seq_len, int head_dim
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // output position
    int d = blockIdx.y * blockDim.y + threadIdx.y;  // dimension

    if (i < seq_len && d < head_dim) {
        float result = 0.0f;

        for (int j = 0; j < seq_len; j++) {
            int score_idx = ((b * heads + h) * seq_len + i) * seq_len + j;
            int v_idx = ((b * heads + h) * seq_len + j) * head_dim + d;
            result += scores[score_idx] * V[v_idx];
        }

        int out_idx = ((b * heads + h) * seq_len + i) * head_dim + d;
        output[out_idx] = result;
    }
}
```

**Optimization workflow:**

1. **Profile baseline** (30 min):
   ```bash
   ncu --set full -o attention_v0 python attention_test.py slow
   ```
   - Document baseline performance
   - Identify primary bottlenecks in each kernel
   - Prioritize optimization targets

2. **Optimization iteration 1** (45 min): Use shared memory for data reuse
   - Profile: `ncu --set full -o attention_v1`
   - Compare metrics vs baseline
   - Measure speedup

3. **Optimization iteration 2** (45 min): Optimize memory access patterns (coalescing)
   - Profile: `ncu --set full -o attention_v2`
   - Check coalescing efficiency improvement
   - Measure speedup

4. **Optimization iteration 3** (45 min): Kernel fusion (combine QK + softmax)
   - Profile: `ncu --set full -o attention_v3`
   - Check memory transaction reduction
   - Measure speedup

5. **Optimization iteration 4** (45 min): Use fast math and vectorization
   - Profile: `ncu --set full -o attention_v4`
   - Check instruction throughput
   - Measure final speedup

**Target metrics:**
- Version 0 (baseline): ~10 ms for (batch=8, heads=12, seq_len=512, head_dim=64)
- Version 4 (optimized): <2 ms (>5x speedup)

#### Part 3: Production Debugging (1 hour)

Debug a performance regression in production code using profiling.

**Scenario**: After a recent update, transformer inference latency increased by 30%. Use profiling to identify the regression.

**Provided**:
- `model_v1.py` (baseline, fast)
- `model_v2.py` (updated, slow)
- Both models produce identical results

**Tasks:**
1. Profile both versions with Nsight Systems
2. Compare timeline and identify new bottlenecks
3. Analyze specific kernel changes with Nsight Compute
4. Root cause the regression
5. Propose fix

**Common causes to investigate:**
- Disabled kernel fusion
- Changed launch configuration
- Introduced synchronization points
- Memory allocation on critical path
- Removed caching

**Deliverables:**

1. Optimization report (`bottleneck_analysis.md`) containing:
   - Bottleneck classification for all provided kernels
   - Detailed optimization log with before/after metrics
   - Performance progression graph across 5 versions
   - Regression root cause analysis

2. Optimized attention implementation with all improvements

3. Regression debugging report with timeline comparisons

**Expected Results:**
- Bottleneck classification: >90% accuracy
- Attention optimization: >5x speedup baseline to final
- Each iteration: 1.3-2x incremental speedup
- Regression: Identified within 30 minutes

---

## Submission Guidelines

For each exercise, submit:

1. **Code**: All implementation files
2. **Profiling Data**: Nsight Compute/Systems reports (.ncu-rep, .qdrep files)
3. **Analysis Document**: Markdown report with findings and recommendations
4. **Performance Data**: Before/after metrics, speedup calculations
5. **Visualizations**: Graphs, charts, timeline screenshots

## Evaluation Criteria

Your work will be evaluated on:

1. **Profiling Skills (30%)**:
   - Correct use of profiling tools
   - Appropriate metrics selection
   - Accurate interpretation of results

2. **Optimization Quality (30%)**:
   - Achieved speedups meet targets
   - Solutions are production-ready
   - Code quality and documentation

3. **Analysis Depth (25%)**:
   - Thorough bottleneck identification
   - Clear optimization rationale
   - Systematic methodology

4. **Production Readiness (15%)**:
   - <5% profiling overhead
   - Automated testing works correctly
   - Integration is straightforward

## Additional Resources

- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [CUDA Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- Module 3 Lecture Notes: Advanced Profiling Techniques

## Tips for Success

1. **Start with high-level profiling** (Nsight Systems) to find hotspots
2. **Then use detailed profiling** (Nsight Compute) on specific kernels
3. **Always profile before and after** optimizations to measure impact
4. **Focus on the biggest bottlenecks first** - Amdahl's Law applies
5. **Validate numerical correctness** after every optimization
6. **Keep profiling overhead minimal** in production scenarios
7. **Automate regression testing** to prevent backsliding
8. **Document your findings** - profiling insights are valuable for the team

Good luck! Remember: "Measurement is the first step that leads to control and eventually to improvement." - James Harrington
