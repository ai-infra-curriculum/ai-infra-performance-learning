# Module 4 Exercises: Transformer Optimization

These exercises provide hands-on experience optimizing transformer models for production deployment, covering Flash Attention, KV caching, kernel fusion, and long-context optimization.

## Prerequisites

- Completed Modules 1-3
- Strong understanding of transformer architecture
- Experience with custom CUDA kernels
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU (A100 or H100 recommended)

## Setup

```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate flash-attn pytest pytest-benchmark

# Verify Flash Attention installation
python -c "import flash_attn; print(flash_attn.__version__)"

# Create workspace
mkdir -p mod-004-exercises
cd mod-004-exercises
```

## Learning Objectives

By completing these exercises, you will:

1. Implement KV caching for autoregressive generation
2. Build Flash Attention from scratch
3. Optimize attention with kernel fusion
4. Handle long contexts with sparse attention
5. Deploy production-optimized transformers

## Exercise Overview

| Exercise | Topic | Time | Difficulty |
|----------|-------|------|------------|
| 1 | KV Cache Implementation | 4 hours | Intermediate |
| 2 | Flash Attention Basics | 6 hours | Advanced |
| 3 | Grouped-Query Attention | 3 hours | Intermediate |
| 4 | Long Context Optimization | 5 hours | Advanced |
| 5 | Production Transformer | 8 hours | Advanced |

**Total Time**: ~26 hours

---

## Exercise 1: KV Cache Implementation (4 hours)

### Objective

Implement efficient KV caching for autoregressive generation, achieving 3-5x speedup over naive implementation.

### Background

Autoregressive generation recomputes attention for all previous tokens at each step. KV caching stores previously computed keys and values, eliminating redundant computation.

### Tasks

#### Part 1: Baseline Attention (30 min)

Implement standard attention without caching:

```python
# baseline_attention.py
import torch
import torch.nn as nn
import time

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        Returns: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        # TODO: Implement standard attention
        # 1. Project to Q, K, V
        # 2. Reshape for multi-head attention
        # 3. Compute attention scores
        # 4. Apply softmax
        # 5. Compute output
        # 6. Reshape and project

        pass

def generate_baseline(model, prompt, max_tokens=50):
    """Generate tokens without caching"""
    tokens = prompt

    for _ in range(max_tokens):
        # Forward pass through ENTIRE sequence every time
        logits = model(tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)

    return tokens

# TODO: Implement and benchmark baseline
# Expected: ~180ms for 50 tokens on GPT-2
```

#### Part 2: KV Cache Implementation (1.5 hours)

Implement attention with KV caching:

```python
# cached_attention.py
class CachedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, cache=None, use_cache=True):
        """
        x: [batch, seq_len, d_model] - can be single token for generation
        cache: Optional dict with 'k' and 'v' tensors

        Returns:
            output: [batch, seq_len, d_model]
            new_cache: Updated cache dict
        """
        batch, seq_len, d_model = x.shape

        # TODO: Implement cached attention
        # 1. Project to Q, K, V (only for new tokens)
        # 2. Concatenate with cached K, V if cache provided
        # 3. Compute attention (Q only for new tokens!)
        # 4. Return output and updated cache

        pass

def generate_cached(model, prompt, max_tokens=50):
    """Efficient generation with KV caching"""
    tokens = prompt
    cache = None

    # First pass: process prompt
    logits, cache = model(tokens, cache=None, use_cache=True)
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)

    # Subsequent passes: only process new token
    for _ in range(max_tokens - 1):
        # TODO: Forward pass with single token and cache
        # This is the key optimization!
        pass

    return tokens

# TODO: Implement and benchmark
# Target: 3-5x speedup vs baseline
```

#### Part 3: Multi-Layer Cache Management (1 hour)

Extend caching to full transformer:

```python
# full_model_cache.py
class CachedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = CachedAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, cache=None, use_cache=True):
        # TODO: Implement with residual connections
        pass

class CachedTransformer(nn.Module):
    def __init__(self, n_layers=12, d_model=768, n_heads=12, d_ff=3072):
        super().__init__()
        self.layers = nn.ModuleList([
            CachedTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        # TODO: Add embedding, output layers

    def forward(self, x, cache_list=None, use_cache=True):
        """
        cache_list: List of caches for each layer
        """
        if cache_list is None:
            cache_list = [None] * len(self.layers)

        new_cache_list = []

        # TODO: Pass through all layers with per-layer caching

        return output, new_cache_list
```

#### Part 4: Optimization and Profiling (1 hour)

Profile and optimize your implementation:

```python
# profile_cache.py
import torch.profiler as profiler

def profile_generation(model, prompt_len=100, gen_len=50):
    """Profile baseline vs cached generation"""
    prompt = torch.randint(0, 50257, (1, prompt_len), device='cuda')

    # Profile baseline
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof_baseline:
        with torch.no_grad():
            _ = generate_baseline(model, prompt, max_tokens=gen_len)

    # Profile cached
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof_cached:
        with torch.no_grad():
            _ = generate_cached(model, prompt, max_tokens=gen_len)

    # Compare
    print("Baseline Profile:")
    print(prof_baseline.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("\nCached Profile:")
    print(prof_cached.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# TODO: Identify and fix bottlenecks
# - Unnecessary memory allocations
- Cache concatenation overhead
# - Attention computation inefficiencies
```

**Deliverables**:

1. Working KV cache implementation for single-layer and multi-layer transformers
2. Generation functions (baseline and cached)
3. Profiling report showing:
   - Speedup achieved (target: 3-5x)
   - Memory usage comparison
   - Per-operation timing breakdown
4. Test suite validating numerical correctness

**Expected Results**:
- Generation speedup: 3-5x for 50 tokens
- Memory increase: 2-3% (cache overhead)
- Accuracy: Identical outputs to baseline

---

## Exercise 2: Flash Attention Basics (6 hours)

### Objective

Implement simplified Flash Attention, understanding tiling and online softmax techniques.

### Background

Flash Attention reduces memory from O(N²) to O(N) by:
1. Tiling computations to fit in SRAM
2. Using online softmax to avoid materializing attention matrix
3. Recomputing values in backward pass

### Tasks

#### Part 1: Online Softmax (1.5 hours)

Implement online softmax algorithm:

```python
# online_softmax.py
import torch

def standard_softmax(x):
    """Standard two-pass softmax"""
    # Pass 1: Find max
    max_val = x.max(dim=-1, keepdim=True).values

    # Pass 2: Compute exp and normalize
    exp_x = torch.exp(x - max_val)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

def online_softmax_sequential(x, block_size=256):
    """
    Compute softmax in blocks without storing full result

    Args:
        x: [batch, seq_len] - attention scores
        block_size: Size of blocks to process

    Returns:
        softmax(x): [batch, seq_len]
    """
    batch, seq_len = x.shape

    # TODO: Implement online softmax
    # Key variables:
    # - m: running maximum
    # - d: running sum of exponentials
    # - output: accumulated softmax values

    # For each block:
    #   1. Update running maximum
    #   2. Rescale previous contributions
    #   3. Add new contributions
    #   4. Update statistics

    pass

def test_online_softmax():
    """Verify correctness"""
    x = torch.randn(4, 1024, device='cuda')

    standard = standard_softmax(x)
    online = online_softmax_sequential(x, block_size=256)

    assert torch.allclose(standard, online, atol=1e-5)
    print("✓ Online softmax matches standard softmax")

# TODO: Implement and test
```

#### Part 2: Tiled Attention (2 hours)

Implement tiled attention computation:

```python
# tiled_attention.py
def standard_attention(Q, K, V):
    """Baseline attention - materializes N×N matrix"""
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output

def tiled_attention(Q, K, V, block_size=64):
    """
    Tiled attention - processes in blocks to save memory

    Args:
        Q, K, V: [batch, n_heads, seq_len, head_dim]
        block_size: Tile size for SRAM

    Returns:
        output: [batch, n_heads, seq_len, head_dim]
    """
    batch, n_heads, seq_len, head_dim = Q.shape

    # Initialize output and statistics
    O = torch.zeros_like(Q)
    l = torch.zeros(batch, n_heads, seq_len, device=Q.device)  # Softmax denominator
    m = torch.full((batch, n_heads, seq_len), -float('inf'), device=Q.device)  # Running max

    # TODO: Implement tiled attention
    # Outer loop: Iterate over Q tiles
    for q_start in range(0, seq_len, block_size):
        q_end = min(q_start + block_size, seq_len)
        Q_block = Q[:, :, q_start:q_end, :]

        # Initialize block accumulators
        # ...

        # Inner loop: Iterate over K, V tiles
        for kv_start in range(0, seq_len, block_size):
            kv_end = min(kv_start + block_size, seq_len)
            K_block = K[:, :, kv_start:kv_end, :]
            V_block = V[:, :, kv_start:kv_end, :]

            # TODO:
            # 1. Compute attention scores for this tile
            # 2. Update running statistics (online softmax)
            # 3. Accumulate output
            pass

        # TODO: Final normalization for this Q block
        pass

    return O

def test_tiled_attention():
    """Verify correctness and memory savings"""
    batch, n_heads, seq_len, head_dim = 2, 12, 2048, 64

    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')

    # Compare outputs
    standard = standard_attention(Q, K, V)
    tiled = tiled_attention(Q, K, V, block_size=256)

    assert torch.allclose(standard, tiled, atol=1e-3)
    print("✓ Tiled attention matches standard attention")

    # TODO: Measure memory usage
    # - Peak memory for standard vs tiled
    # - Expected: ~4x reduction for seq_len=2048

# TODO: Implement and test
```

#### Part 3: Flash Attention Forward (2 hours)

Combine tiling and online softmax:

```python
# flash_attention_fwd.py
def flash_attention_forward(Q, K, V, block_size_m=128, block_size_n=64):
    """
    Flash Attention forward pass

    Combines:
    - Tiled computation (fits in SRAM)
    - Online softmax (no materialized attention matrix)

    Args:
        Q, K, V: [batch, n_heads, seq_len, head_dim]
        block_size_m: Query block size
        block_size_n: Key/Value block size

    Returns:
        output: [batch, n_heads, seq_len, head_dim]
        l: Softmax denominators (for backward)
        m: Running maxes (for backward)
    """
    batch, n_heads, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)

    # Initialize
    O = torch.zeros_like(Q)
    l = torch.zeros(batch, n_heads, seq_len, 1, device=Q.device)
    m = torch.full((batch, n_heads, seq_len, 1), -float('inf'), device=Q.device)

    # TODO: Implement Flash Attention algorithm
    # Pseudocode:
    # for each Q block:
    #     for each K, V block:
    #         1. Load Q, K, V blocks to SRAM
    #         2. Compute S = Q @ K^T
    #         3. Update running max: m_new = max(m_old, max(S))
    #         4. Compute correction factor: exp(m_old - m_new)
    #         5. Rescale previous O, l
    #         6. Compute P = exp(S - m_new)
    #         7. Update l += sum(P)
    #         8. Update O += P @ V
    #     9. Final normalization: O = O / l

    pass

def benchmark_flash_attention():
    """Compare Flash Attention vs standard"""
    import time

    seq_lengths = [512, 1024, 2048, 4096]

    for seq_len in seq_lengths:
        Q = torch.randn(1, 12, seq_len, 64, device='cuda')
        K = torch.randn(1, 12, seq_len, 64, device='cuda')
        V = torch.randn(1, 12, seq_len, 64, device='cuda')

        # Standard attention
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = standard_attention(Q, K, V)
        torch.cuda.synchronize()
        time_standard = (time.time() - start) / 10

        # Flash attention
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = flash_attention_forward(Q, K, V)
        torch.cuda.synchronize()
        time_flash = (time.time() - start) / 10

        speedup = time_standard / time_flash
        print(f"Seq len {seq_len}: Standard={time_standard*1000:.2f}ms, "
              f"Flash={time_flash*1000:.2f}ms, Speedup={speedup:.2f}x")

# TODO: Implement and benchmark
# Target speedup: 1.5-3x depending on sequence length
```

#### Part 4: Integration with PyTorch (30 min)

Create PyTorch-compatible module:

```python
# flash_attention_module.py
class FlashAttention(nn.Module):
    """PyTorch module wrapping Flash Attention"""

    def __init__(self, d_model, n_heads, block_size_m=128, block_size_n=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size_m = block_size_m
        self.block_size_n = block_size_n

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        Returns: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape

        # Project and reshape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Flash attention
        output, _, _ = flash_attention_forward(
            q, k, v,
            block_size_m=self.block_size_m,
            block_size_n=self.block_size_n
        )

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(output)

# TODO: Test in transformer model
```

**Deliverables**:

1. Working online softmax implementation
2. Tiled attention with memory optimization
3. Complete Flash Attention forward pass
4. PyTorch module integration
5. Benchmark report comparing:
   - Speed (standard vs Flash)
   - Memory usage
   - Numerical accuracy

**Expected Results**:
- Memory: O(N²) → O(N)
- Speed: 1.5-3x faster for seq_len > 1024
- Accuracy: <1e-3 max error vs standard attention

---

## Exercise 3: Grouped-Query Attention (3 hours)

### Objective

Implement Grouped-Query Attention (GQA) to reduce KV cache memory while maintaining quality.

### Background

GQA groups query heads to share K, V projections:
- Standard MHA: n_heads Q heads, n_heads KV heads
- GQA: n_heads Q heads, n_kv_heads KV heads (n_kv_heads < n_heads)
- MQA: n_heads Q heads, 1 KV head (extreme case)

### Tasks

#### Part 1: GQA Implementation (1.5 hours)

```python
# grouped_query_attention.py
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of query heads
            n_kv_heads: Number of key/value heads (n_heads % n_kv_heads == 0)
        """
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        # TODO: Create projections
        # Q: d_model → n_heads * head_dim
        # K, V: d_model → n_kv_heads * head_dim (smaller!)

        pass

    def forward(self, x, cache=None, use_cache=True):
        """
        x: [batch, seq_len, d_model]
        Returns: output, cache
        """
        batch, seq_len, _ = x.shape

        # TODO: Implement GQA
        # 1. Project Q (full n_heads), K, V (only n_kv_heads)
        # 2. Repeat each KV head n_groups times to match Q heads
        # 3. Standard attention computation
        # 4. Return with cache

        pass

def compare_attention_variants():
    """Compare standard MHA, GQA, and MQA"""
    d_model = 768
    n_heads = 12
    batch, seq_len = 8, 1024

    x = torch.randn(batch, seq_len, d_model, device='cuda')

    # Standard MHA
    mha = StandardAttention(d_model, n_heads).cuda()

    # GQA variants
    gqa_6 = GroupedQueryAttention(d_model, n_heads, n_kv_heads=6).cuda()
    gqa_4 = GroupedQueryAttention(d_model, n_heads, n_kv_heads=4).cuda()
    gqa_2 = GroupedQueryAttention(d_model, n_heads, n_kv_heads=2).cuda()
    mqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads=1).cuda()

    # TODO: Compare:
    # - Cache memory size
    # - Inference speed
    # - Parameter count
    # - Generate quality report

# TODO: Implement and analyze
```

#### Part 2: Quality vs Efficiency Trade-off (1 hour)

Analyze GQA trade-offs:

```python
# gqa_analysis.py
def analyze_gqa_tradeoffs(model_configs):
    """
    Analyze cache size, speed, and quality for different GQA configs

    Args:
        model_configs: List of (n_heads, n_kv_heads) tuples
    """
    results = []

    for n_heads, n_kv_heads in model_configs:
        # TODO: Measure:
        # 1. Cache memory (KB per token)
        # 2. Inference speed (ms)
        # 3. Quality (perplexity or downstream task)

        results.append({
            'config': f'{n_heads}H-{n_kv_heads}KV',
            'cache_mb_per_1k_tokens': ...,
            'inference_ms': ...,
            'relative_quality': ...,
        })

    return results

# TODO: Run analysis and create visualization
# Plot: Cache size vs speed vs quality
```

#### Part 3: Integration (30 min)

Integrate GQA into transformer:

```python
# gqa_transformer.py
class GQATransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff):
        super().__init__()
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, cache=None, use_cache=True):
        # TODO: Standard transformer block with GQA
        pass

# TODO: Test end-to-end generation
```

**Deliverables**:

1. Working GQA implementation with configurable n_kv_heads
2. Comparison report (MHA vs GQA-4 vs GQA-2 vs MQA)
3. Trade-off analysis (cache vs speed vs quality)
4. Integration into transformer model

**Expected Results**:
- GQA-4: 3x cache reduction, <1% quality loss
- GQA-2: 6x cache reduction, 1-2% quality loss
- MQA: 12x cache reduction, 2-4% quality loss

---

## Exercise 4: Long Context Optimization (5 hours)

### Objective

Implement techniques for handling 16K+ token contexts efficiently.

### Background

Standard attention scales quadratically with sequence length. Long contexts require:
- Sparse attention patterns
- Compression techniques
- Efficient memory management

### Tasks

#### Part 1: Sliding Window Attention (1.5 hours)

```python
# sliding_window_attention.py
def sliding_window_attention(Q, K, V, window_size=512):
    """
    Each query attends only to nearby keys within window

    Complexity: O(N × window_size) instead of O(N²)
    """
    batch, n_heads, seq_len, head_dim = Q.shape

    # TODO: Implement sliding window
    # Create sparse attention mask
    # Compute attention only for valid positions

    pass

def benchmark_sliding_window():
    """Compare full vs sliding window attention"""
    seq_lengths = [2048, 4096, 8192, 16384]
    window_size = 512

    for seq_len in seq_lengths:
        # TODO: Benchmark and compare:
        # - Memory usage
        # - Computation time
        # - Quality (on sample task)
        pass
```

#### Part 2: Block-Sparse Attention (1.5 hours)

Implement Longformer-style sparse attention:

```python
# block_sparse_attention.py
def block_sparse_attention(Q, K, V, block_size=64, num_global=128):
    """
    Attention pattern:
    - Local: Block-diagonal attention
    - Global: First num_global tokens attend to all
    - Dilated: Strided attention for longer range

    Args:
        Q, K, V: [batch, n_heads, seq_len, head_dim]
        block_size: Size of local blocks
        num_global: Number of global attention tokens
    """
    batch, n_heads, seq_len, head_dim = Q.shape

    # TODO: Create block-sparse mask
    # 1. Local blocks (block-diagonal)
    # 2. Global tokens (attend to/from all)
    # 3. Optional: dilated attention for medium-range

    # TODO: Efficient sparse computation
    # Don't compute masked-out attention scores

    pass

# TODO: Implement and evaluate on long-context tasks
```

#### Part 3: Streaming KV Cache (1.5 hours)

Implement cache compression for very long contexts:

```python
# streaming_cache.py
class StreamingKVCache:
    """
    Manages KV cache for streaming long contexts
    - Recent tokens: Full resolution
    - Old tokens: Compressed
    """

    def __init__(self, window_size=2048, compress_ratio=4):
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.recent_cache = {'k': [], 'v': []}
        self.compressed_cache = {'k': [], 'v': []}

    def append(self, k_new, v_new):
        """Add new K, V to cache"""
        # TODO:
        # 1. Add to recent cache
        # 2. If recent cache exceeds window_size:
        #    - Compress oldest entries
        #    - Move to compressed cache
        pass

    def get_full_cache(self):
        """Retrieve full cache (recent + decompressed)"""
        # TODO: Concatenate recent and compressed caches
        pass

    def compress(self, k, v):
        """Compress KV cache entries"""
        # TODO: Implement compression
        # Options:
        # - Average pooling
        # - Learned compression
        # - Select representative tokens
        pass

# TODO: Implement and test on 32K+ context
```

#### Part 4: End-to-End Long Context (30 min)

Combine techniques:

```python
# long_context_model.py
class LongContextTransformer(nn.Module):
    """
    Transformer optimized for long contexts
    - Sliding window for recent context
    - Sparse attention for global context
    - Streaming cache for efficiency
    """

    def __init__(
        self,
        d_model=768,
        n_layers=12,
        n_heads=12,
        window_size=2048,
        max_context=32768
    ):
        super().__init__()
        # TODO: Initialize with appropriate attention mechanisms
        pass

    def forward(self, input_ids, cache=None):
        # TODO: Process long context efficiently
        pass

# TODO: Benchmark on long-context tasks
# - Quality vs standard attention
# - Memory usage
# - Speed
```

**Deliverables**:

1. Sliding window attention implementation
2. Block-sparse attention with global tokens
3. Streaming KV cache with compression
4. Integrated long-context transformer
5. Benchmark report on 16K-32K contexts

**Expected Results**:
- Memory: 8-16x reduction vs standard attention
- Speed: 5-10x faster for 16K+ contexts
- Quality: >95% of full attention on long-context tasks

---

## Exercise 5: Production Transformer (8 hours)

### Objective

Build production-ready transformer with all optimizations integrated.

### Background

Combine all techniques into deployable system:
- Flash Attention
- KV caching
- GQA
- Fused operations
- Batched inference

### Tasks

#### Part 1: Optimized Transformer (3 hours)

```python
# production_transformer.py
class ProductionTransformer(nn.Module):
    """
    Production-optimized transformer with:
    - Flash Attention v2
    - KV caching with PagedAttention
    - Grouped-query attention
    - Fused LayerNorm + residual
    - Fused GELU
    """

    def __init__(
        self,
        vocab_size=50257,
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        d_ff=3072,
        max_seq_len=8192,
        use_flash=True,
        use_paged_cache=True
    ):
        super().__init__()
        # TODO: Initialize all components with optimizations
        pass

    def forward(self, input_ids, cache=None, use_cache=True):
        # TODO: Forward pass with all optimizations
        pass

# TODO: Implement complete model
```

#### Part 2: Batched Serving (2 hours)

Implement efficient batched inference:

```python
# batched_serving.py
class ContinuousBatcher:
    """
    Continuous batching for high throughput serving
    """

    def __init__(self, model, max_batch_size=32, max_seq_len=2048):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.pending_requests = []
        self.active_requests = {}

    def add_request(self, request_id, prompt, max_tokens=50):
        """Add new generation request"""
        # TODO: Add to pending queue
        pass

    def step(self):
        """Process one generation step for all active requests"""
        # TODO:
        # 1. Form batch from active requests
        # 2. Run model forward pass
        # 3. Update each request
        # 4. Remove completed requests
        # 5. Add new requests from pending
        pass

    async def generate(self, prompt, max_tokens=50):
        """Async generation interface"""
        # TODO: Implement async request handling
        pass

# TODO: Implement and benchmark throughput
```

#### Part 3: Performance Optimization (2 hours)

Profile and optimize the complete system:

```python
# optimize_system.py
def profile_production_model():
    """Profile complete system"""
    model = ProductionTransformer(...).cuda()

    # TODO: Profile with various workloads
    # - Single request latency
    # - Batched throughput
    # - Memory usage
    # - GPU utilization

    # Identify bottlenecks and optimize
    pass

def benchmark_production():
    """Comprehensive benchmark suite"""
    configurations = [
        ('baseline', StandardTransformer),
        ('flash', FlashTransformer),
        ('flash+gqa', FlashGQATransformer),
        ('full', ProductionTransformer),
    ]

    # TODO: Benchmark each configuration
    # Metrics:
    # - Latency (p50, p95, p99)
    # - Throughput (req/s)
    # - Memory (MB)
    # - Quality (perplexity)

    pass
```

#### Part 4: Deployment (1 hour)

Package for deployment:

```python
# serve.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 1.0

class GenerationResponse(BaseModel):
    text: str
    tokens: int
    latency_ms: float

# Load model
model = ProductionTransformer.from_pretrained('gpt2')
model = model.cuda().eval()
batcher = ContinuousBatcher(model, max_batch_size=32)

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text from prompt"""
    # TODO: Implement with batching support
    pass

@app.get("/health")
def health():
    return {"status": "healthy", "gpu_memory": torch.cuda.memory_allocated()}

# TODO: Add monitoring, metrics, logging
```

**Deliverables**:

1. Complete production-optimized transformer
2. Continuous batching implementation
3. Comprehensive benchmark report comparing:
   - Baseline
   - Individual optimizations
   - Full optimization stack
4. Deployment-ready serving code
5. Performance profiling analysis

**Expected Results**:
- Latency: 5-10x faster than baseline
- Throughput: 10-20x higher with batching
- Memory: 3-5x more efficient
- Quality: <1% degradation

---

## Submission Guidelines

For each exercise, submit:

1. **Code**: Complete, documented implementations
2. **Tests**: Unit tests and integration tests
3. **Benchmarks**: Performance comparison data
4. **Analysis**: Written analysis of results
5. **Documentation**: Usage examples and API docs

## Evaluation Criteria

1. **Correctness (30%)**:
   - Numerical accuracy vs reference implementations
   - Passes all test cases

2. **Performance (30%)**:
   - Meets speedup targets
   - Efficient memory usage
   - Proper GPU utilization

3. **Code Quality (20%)**:
   - Clean, readable code
   - Proper documentation
   - Good software engineering practices

4. **Analysis (20%)**:
   - Thorough benchmarking
   - Insightful performance analysis
   - Clear trade-off discussions

## Resources

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Flash Attention v2](https://arxiv.org/abs/2307.08691)
- [GQA Paper](https://arxiv.org/abs/2305.13245)
- [Longformer](https://arxiv.org/abs/2004.05150)
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)

## Tips for Success

1. **Start simple**: Implement basic versions before optimizing
2. **Test incrementally**: Verify correctness at each step
3. **Profile early**: Use profiling to guide optimization
4. **Compare carefully**: Always benchmark against baselines
5. **Document trade-offs**: Understand speed vs memory vs quality

Good luck! These optimizations are critical for production ML systems and represent state-of-the-art techniques used by leading AI companies.
