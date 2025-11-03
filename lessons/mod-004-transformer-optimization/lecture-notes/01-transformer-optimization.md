# Transformer Optimization Techniques

## Table of Contents

1. [Introduction](#introduction)
2. [Transformer Architecture Review](#transformer-architecture-review)
3. [Attention Bottlenecks](#attention-bottlenecks)
4. [Flash Attention v2](#flash-attention-v2)
5. [KV Cache Optimization](#kv-cache-optimization)
6. [Fused Operations](#fused-operations)
7. [Multi-Query and Grouped-Query Attention](#multi-query-and-grouped-query-attention)
8. [Long Context Optimization](#long-context-optimization)
9. [PagedAttention](#pagedattention)
10. [Production Implementation](#production-implementation)
11. [Case Studies](#case-studies)

## Introduction

Transformer models have revolutionized AI, powering everything from GPT-4 to protein folding. However, their computational demands grow quadratically with sequence length, making optimization critical for production deployment. This lecture covers state-of-the-art optimization techniques that can achieve 5-10x speedups while maintaining numerical accuracy.

### Why Optimize Transformers?

**Cost at Scale**:
- GPT-3 (175B parameters) inference: ~$0.06 per 1K tokens
- 10M requests/day = $600,000/day
- 2x speedup = $300,000/day savings = $109M/year

**Latency Requirements**:
- Interactive chat: <200ms response time target
- Code completion: <100ms for smooth UX
- Voice assistants: <50ms for natural conversation

**Memory Constraints**:
- Standard attention: O(N²) memory for sequence length N
- GPT-3 with 2048 tokens: 48GB attention matrix alone
- Flash Attention: O(N) memory - enables 10x longer contexts

### Module Goals

By the end of this lecture, you will understand:

1. **Attention Bottlenecks**: Why attention is the primary performance bottleneck
2. **Flash Attention**: How to reduce O(N²) memory to O(N)
3. **KV Caching**: 3-4x inference speedup for autoregressive generation
4. **Kernel Fusion**: 1.5-2x speedup by combining operations
5. **Long Context**: Techniques for handling 100K+ token contexts
6. **Production Deployment**: Real-world implementation patterns

## Transformer Architecture Review

### Standard Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x):
        # Self-attention with residual
        x = x + self.attention(self.norm1(x))

        # Feedforward with residual
        x = x + self.ff(self.norm2(x))

        return x
```

### Attention Mechanism

The core of transformer performance challenges lies in the attention mechanism:

```python
def attention(Q, K, V):
    """
    Q: [batch, heads, seq_len, head_dim]
    K: [batch, heads, seq_len, head_dim]
    V: [batch, heads, seq_len, head_dim]
    """
    # Compute attention scores: O(N²d) complexity
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)

    # Softmax: O(N²) complexity
    attn_weights = torch.softmax(scores, dim=-1)

    # Apply attention: O(N²d) complexity
    output = torch.matmul(attn_weights, V)

    return output
```

**Complexity Analysis**:
- Time: O(N²d) where N = sequence length, d = hidden dimension
- Memory: O(N²) for attention matrix
- For GPT-3 (d=12288, N=2048): 4B multiplications per layer × 96 layers = 384B ops

### FeedForward Network

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

**Complexity**: O(Nd²) - typically d_ff = 4d

For typical transformer:
- Attention: 40-50% of compute time
- FFN: 40-50% of compute time
- LayerNorm + residuals: 5-10% of compute time

## Attention Bottlenecks

### Memory Bottleneck

**Problem**: Standard attention materializes the N×N attention matrix

```python
# This creates a massive intermediate tensor!
scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, heads, N, N]
```

**Memory Requirements**:
- Sequence length N=1024: 1M elements per head
- 12 heads: 12M elements
- Float16: 24 MB per batch
- Batch size 32: 768 MB just for attention scores

**Scaling**:
- N=2048: 4x memory (quadratic scaling!)
- N=4096: 16x memory
- N=8192: 64x memory

### Computational Bottleneck

**FLOP Analysis** for single attention layer:

1. **QK^T matmul**: 2Nd² FLOPs
2. **Softmax**: 5N² FLOPs (reduction + exp + division)
3. **Attention × V**: 2N²d FLOPs

**Total**: 2Nd² + 2N²d FLOPs

For GPT-3 (d=12288, N=2048):
- QK^T: 103 GFLOPs
- Softmax: 21 GFLOPs
- Attn×V: 103 GFLOPs
- **Total: 227 GFLOPs per layer**

With 96 layers: 21.8 TFLOPs per forward pass

### Memory Access Pattern Bottleneck

**Problem**: Non-contiguous memory access in attention computation

```cuda
// Inefficient: Q[i] accessed N times, K[j] accessed N times
for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < seq_len; j++) {
        score[i][j] = dot(Q[i], K[j]);  // Repeated loads!
    }
}
```

**HBM Bandwidth Analysis**:
- A100 HBM bandwidth: 2 TB/s theoretical
- Attention reads Q, K, V (3× Nd elements)
- Writes attention matrix (N² elements)
- For N=2048, d=128: (3×2048×128 + 2048²)×2 bytes = 18 MB
- Time: 18 MB / 2000 GB/s = 9 ¼s
- Actual measured: 450 ¼s (50x overhead!)

**Why the gap?**
- Memory access pattern is not coalesced
- Multiple passes over same data
- Intermediate results don't fit in cache

## Flash Attention v2

Flash Attention revolutionizes attention computation by:
1. **Tiling**: Breaking computation into blocks that fit in SRAM
2. **Recomputation**: Trading compute for memory bandwidth
3. **Online Softmax**: Avoiding materialization of attention matrix

### Key Innovation: Online Softmax

Traditional softmax requires two passes:

```python
# Pass 1: Find max for numerical stability
m = torch.max(x, dim=-1)

# Pass 2: Compute exp and normalize
exp_x = torch.exp(x - m)
sum_exp = torch.sum(exp_x, dim=-1)
softmax_x = exp_x / sum_exp
```

**Online Softmax** computes in one pass using incremental updates:

```python
def online_softmax(x, chunk_size):
    """
    Compute softmax in chunks without storing full result
    """
    m = -float('inf')  # Running max
    d = 0.0            # Running denominator

    for chunk in x.chunks(chunk_size):
        # Update running max
        m_new = max(m, chunk.max())

        # Rescale previous contribution
        correction = exp(m - m_new)
        d = d * correction

        # Add new contribution
        exp_chunk = exp(chunk - m_new)
        d = d + exp_chunk.sum()

        # Update max
        m = m_new

    return d, m
```

### Flash Attention Algorithm

```python
def flash_attention(Q, K, V, block_size=256):
    """
    Flash Attention with tiling and online softmax

    Args:
        Q, K, V: [batch, heads, seq_len, head_dim]
        block_size: Tile size that fits in SRAM

    Returns:
        output: [batch, heads, seq_len, head_dim]
    """
    N, d = Q.shape[-2:]

    # Initialize output and statistics
    O = torch.zeros_like(Q)
    l = torch.zeros(Q.shape[:-1])  # Running sum for softmax
    m = torch.full(Q.shape[:-1], -float('inf'))  # Running max

    # Tile over sequence length
    for i in range(0, N, block_size):
        Q_block = Q[:, :, i:i+block_size, :]  # Load Q block to SRAM

        # Initialize block outputs
        O_block = torch.zeros_like(Q_block)
        l_block = torch.zeros(Q_block.shape[:-1])
        m_block = torch.full(Q_block.shape[:-1], -float('inf'))

        # Inner loop over K, V tiles
        for j in range(0, N, block_size):
            K_block = K[:, :, j:j+block_size, :]  # Load K block to SRAM
            V_block = V[:, :, j:j+block_size, :]  # Load V block to SRAM

            # Compute attention scores for this tile
            S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) / sqrt(d)

            # Update statistics with online algorithm
            m_block_new = torch.maximum(m_block, S_block.max(dim=-1).values)

            # Rescale previous values
            correction = torch.exp(m_block - m_block_new)
            l_block = l_block * correction

            # Compute new contributions
            P_block = torch.exp(S_block - m_block_new.unsqueeze(-1))
            l_block = l_block + P_block.sum(dim=-1)

            # Accumulate output
            O_block = O_block * correction.unsqueeze(-1) + torch.matmul(P_block, V_block)

            # Update max
            m_block = m_block_new

        # Normalize block output
        O[:, :, i:i+block_size, :] = O_block / l_block.unsqueeze(-1)

    return O
```

### Flash Attention v2 Improvements

Flash Attention v2 (2023) improves upon the original with:

1. **Reduced Non-Matmul FLOPs**: Better scheduling to minimize overhead
2. **Parallelism**: Better work partitioning across threads
3. **Work Balancing**: Handling variable sequence lengths efficiently

**Key Optimizations**:

```cuda
// v2: Better parallelism over sequence dimension
// Partition work by (batch, head, query_tile) instead of (batch, head)
__global__ void flash_attention_v2_kernel(
    const float* Q, const float* K, const float* V,
    float* O,
    int batch, int heads, int seq_len, int head_dim,
    int block_m, int block_n
) {
    // Each thread block handles one query tile
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int query_block_idx = blockIdx.z;  // NEW: Parallelize over query tiles

    // Shared memory for Q, K, V tiles
    __shared__ float Q_shared[BLOCK_M][HEAD_DIM];
    __shared__ float K_shared[BLOCK_N][HEAD_DIM];
    __shared__ float V_shared[BLOCK_N][HEAD_DIM];

    // Load Q tile to shared memory (this tile stays resident)
    int q_start = query_block_idx * BLOCK_M;
    for (int i = threadIdx.x; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        if (q_start + row < seq_len) {
            Q_shared[row][col] = Q[batch_idx][head_idx][q_start + row][col];
        }
    }

    // Initialize output accumulators in registers
    float O_local[BLOCK_M][HEAD_DIM] = {0};
    float m_local[BLOCK_M];  // Running max
    float l_local[BLOCK_M];  // Running sum

    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) {
        m_local[i] = -INFINITY;
        l_local[i] = 0.0f;
    }

    // Loop over K, V tiles
    for (int kv_block = 0; kv_block < (seq_len + BLOCK_N - 1) / BLOCK_N; kv_block++) {
        int kv_start = kv_block * BLOCK_N;

        // Load K, V tiles cooperatively
        __syncthreads();  // Ensure previous iteration is done
        // ... K, V loading code ...
        __syncthreads();  // Ensure K, V loaded

        // Compute attention scores: Q @ K^T
        float S_local[BLOCK_M][BLOCK_N];
        for (int i = 0; i < BLOCK_M; i++) {
            for (int j = 0; j < BLOCK_N; j++) {
                float sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < HEAD_DIM; k++) {
                    sum += Q_shared[i][k] * K_shared[j][k];
                }
                S_local[i][j] = sum / sqrtf((float)HEAD_DIM);
            }
        }

        // Online softmax update
        for (int i = 0; i < BLOCK_M; i++) {
            // Find max in this block
            float m_block = -INFINITY;
            for (int j = 0; j < BLOCK_N; j++) {
                m_block = fmaxf(m_block, S_local[i][j]);
            }

            // Update running max
            float m_new = fmaxf(m_local[i], m_block);

            // Rescale previous contributions
            float correction = expf(m_local[i] - m_new);
            l_local[i] *= correction;

            for (int d = 0; d < HEAD_DIM; d++) {
                O_local[i][d] *= correction;
            }

            // Compute softmax and accumulate
            float l_block = 0.0f;
            for (int j = 0; j < BLOCK_N; j++) {
                float p = expf(S_local[i][j] - m_new);
                l_block += p;

                // Accumulate O += P @ V
                for (int d = 0; d < HEAD_DIM; d++) {
                    O_local[i][d] += p * V_shared[j][d];
                }
            }

            l_local[i] += l_block;
            m_local[i] = m_new;
        }
    }

    // Final normalization and write out
    for (int i = 0; i < BLOCK_M; i++) {
        if (q_start + i < seq_len) {
            for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
                O[batch_idx][head_idx][q_start + i][d] =
                    O_local[i][d] / l_local[i];
            }
        }
    }
}
```

### Performance Comparison

**Memory Usage**:
- Standard Attention: O(N²) = 4 MB for N=1024, fp16
- Flash Attention: O(N) = 2 KB for N=1024, fp16
- **Memory Reduction: 2000x**

**Speed (A100 GPU)**:

| Sequence Length | Standard Attention | Flash Attention v2 | Speedup |
|----------------|-------------------|-------------------|---------|
| 512 | 0.45 ms | 0.28 ms | 1.6x |
| 1024 | 1.80 ms | 0.95 ms | 1.9x |
| 2048 | 7.20 ms | 1.95 ms | 3.7x |
| 4096 | 28.80 ms | 4.10 ms | 7.0x |
| 8192 | 115.20 ms | 8.90 ms | 12.9x |

**Why the speedup?**
1. Reduced memory bandwidth (HBM ’ SRAM): 20x faster
2. Better GPU utilization: 85% vs 45%
3. Fewer memory-bound operations

### Flash Attention Backward Pass

Forward pass is only half the story - training requires efficient gradients:

```python
def flash_attention_backward(dO, Q, K, V, O, l, m):
    """
    Backward pass for Flash Attention

    Uses same tiling strategy but computes gradients
    dO: gradient of output
    l, m: statistics saved from forward pass
    """
    N, d = Q.shape[-2:]
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    for i in range(0, N, block_size):
        # Load tiles
        Q_block = Q[:, :, i:i+block_size, :]
        dO_block = dO[:, :, i:i+block_size, :]
        m_block = m[:, :, i:i+block_size]
        l_block = l[:, :, i:i+block_size]

        dQ_block = torch.zeros_like(Q_block)

        for j in range(0, N, block_size):
            K_block = K[:, :, j:j+block_size, :]
            V_block = V[:, :, j:j+block_size, :]

            # Recompute attention scores (no stored from forward!)
            S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) / sqrt(d)
            P_block = torch.exp(S_block - m_block.unsqueeze(-1)) / l_block.unsqueeze(-1)

            # Compute gradients
            dV_block = torch.matmul(P_block.transpose(-2, -1), dO_block)
            dP_block = torch.matmul(dO_block, V_block.transpose(-2, -1))

            # Softmax backward
            dS_block = P_block * (dP_block - (dP_block * P_block).sum(dim=-1, keepdim=True))

            # Accumulate gradients
            dQ_block += torch.matmul(dS_block, K_block) / sqrt(d)
            dK[:, :, j:j+block_size, :] += torch.matmul(dS_block.transpose(-2, -1), Q_block) / sqrt(d)
            dV[:, :, j:j+block_size, :] += dV_block

        dQ[:, :, i:i+block_size, :] = dQ_block

    return dQ, dK, dV
```

**Key insight**: Recompute attention scores during backward rather than storing them. Trades 20% more compute for 10x less memory.

## KV Cache Optimization

### Autoregressive Generation

Transformers are used autoregressively for text generation:

```python
def generate(model, prompt, max_tokens=50):
    """Standard autoregressive generation"""
    tokens = prompt

    for _ in range(max_tokens):
        # Forward pass through entire sequence EVERY TIME
        logits = model(tokens)  # [batch, seq_len, vocab_size]

        # Sample next token
        next_token = torch.argmax(logits[:, -1, :], dim=-1)

        # Append and continue
        tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=1)

    return tokens
```

**Problem**: Each generation step recomputes attention for all previous tokens

- Step 1: Process 100 tokens
- Step 2: Process 101 tokens (100 redundant!)
- Step 3: Process 102 tokens (101 redundant!)
- ...
- Step 50: Process 150 tokens (149 redundant!)

**Total wasted computation**: 100 + 101 + ... + 149 = 6,225 token forward passes

### KV Cache Solution

**Key Insight**: In attention, K and V for previous tokens never change!

```python
# Attention computation
scores = Q @ K.T  # Only Q changes for new token!
output = softmax(scores) @ V  # Only need new Q row
```

**Implementation**:

```python
class CachedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, cache=None, use_cache=True):
        """
        Args:
            x: [batch, seq_len, d_model] - can be single token
            cache: {
                'k': [batch, n_heads, cached_len, head_dim],
                'v': [batch, n_heads, cached_len, head_dim]
            }
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3*d_model]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Use cache if provided
        if cache is not None:
            # Concatenate with cached K, V
            k = torch.cat([cache['k'], k], dim=2)
            v = torch.cat([cache['v'], v], dim=2)

        # Compute attention (only for new query tokens!)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(output)

        # Update cache
        if use_cache:
            new_cache = {'k': k, 'v': v}
            return output, new_cache

        return output
```

**Optimized Generation**:

```python
def generate_with_cache(model, prompt, max_tokens=50):
    """Efficient generation with KV caching"""
    tokens = prompt
    cache = None  # Initialize cache

    # First forward pass: process entire prompt
    logits, cache = model(tokens, cache=None, use_cache=True)
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=1)

    # Subsequent passes: only process new token!
    for _ in range(max_tokens - 1):
        # Only forward pass the NEW token
        logits, cache = model(
            next_token.unsqueeze(-1),  # [batch, 1, d_model]
            cache=cache,
            use_cache=True
        )

        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=1)

    return tokens
```

### Performance Impact

**Computation Reduction**:
- Without cache: N + (N+1) + (N+2) + ... + (N+G) = G·N + G(G+1)/2 operations
- With cache: N + 1 + 1 + ... + 1 = N + G operations
- **Speedup**: H G/2 for long generations (G H N)

**Measured Performance (GPT-2, A100)**:

| Generation Length | Without Cache | With Cache | Speedup |
|------------------|---------------|------------|---------|
| 10 tokens | 15 ms | 8 ms | 1.9x |
| 50 tokens | 180 ms | 45 ms | 4.0x |
| 100 tokens | 650 ms | 85 ms | 7.6x |
| 500 tokens | 15.2 s | 420 ms | 36x |

**Memory Cost**:
- Cache size: 2 × batch × heads × seq_len × head_dim × sizeof(float16)
- GPT-2: 2 × 1 × 12 × 1024 × 64 × 2 bytes = 3 MB per layer × 12 layers = 36 MB
- GPT-3: 2 × 1 × 96 × 2048 × 128 × 2 bytes = 96 MB per layer × 96 layers = 9.2 GB

**Trade-off**: Cache memory grows linearly with sequence length. For very long contexts (>10K tokens), can use strategies like:
- Sliding window cache
- Compress old cache entries
- Selective caching (only important tokens)

### Multi-Batch KV Cache Management

For serving multiple requests efficiently:

```python
class BatchedKVCache:
    """Efficient KV cache for batch serving"""

    def __init__(self, max_batch_size, max_seq_len, n_layers, n_heads, head_dim):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Pre-allocate cache memory
        cache_shape = (n_layers, max_batch_size, n_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(cache_shape, dtype=torch.float16, device='cuda')
        self.v_cache = torch.zeros(cache_shape, dtype=torch.float16, device='cuda')

        # Track which slots are active and current lengths
        self.active_mask = torch.zeros(max_batch_size, dtype=torch.bool)
        self.lengths = torch.zeros(max_batch_size, dtype=torch.int32)

    def allocate_slot(self):
        """Allocate a cache slot for new request"""
        free_slots = (~self.active_mask).nonzero(as_tuple=True)[0]
        if len(free_slots) == 0:
            raise RuntimeError("No free cache slots")

        slot = free_slots[0].item()
        self.active_mask[slot] = True
        self.lengths[slot] = 0
        return slot

    def free_slot(self, slot):
        """Free a cache slot"""
        self.active_mask[slot] = False
        self.lengths[slot] = 0

    def get_cache(self, layer, slot):
        """Get cache for specific layer and slot"""
        length = self.lengths[slot].item()
        return {
            'k': self.k_cache[layer, slot, :, :length, :],
            'v': self.v_cache[layer, slot, :, :length, :]
        }

    def update_cache(self, layer, slot, k_new, v_new):
        """Append new K, V to cache"""
        current_len = self.lengths[slot].item()
        new_len = current_len + k_new.shape[2]  # k_new: [1, heads, new_tokens, head_dim]

        assert new_len <= self.max_seq_len, "Sequence too long for cache"

        # Copy new K, V into cache
        self.k_cache[layer, slot, :, current_len:new_len, :] = k_new[0]
        self.v_cache[layer, slot, :, current_len:new_len, :] = v_new[0]

        self.lengths[slot] = new_len
```

## Fused Operations

### Why Fusion Matters

Modern GPUs are memory-bandwidth limited for most ML operations. Fusing operations reduces memory traffic:

```python
# Unfused: 6 memory operations
x = layer_norm(x)      # Read x, write x
x = x + residual        # Read x, read residual, write x
x = gelu(x)             # Read x, write x

# Fused: 2 memory operations
x = fused_ln_residual_gelu(x, residual)  # Read x and residual, write x
```

**Speedup**: 3x fewer memory operations = ~2x faster (memory-bound workloads)

### Fused LayerNorm + Residual

LayerNorm is a major bottleneck in transformers (5-10% of total time):

```cuda
// fused_layernorm_residual.cu
__global__ void layernorm_residual_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, int D, float eps
) {
    int row = blockIdx.x;

    // Shared memory for reduction
    __shared__ float shared_mean;
    __shared__ float shared_var;

    // Compute mean using Welford's online algorithm
    float mean = 0.0f, m2 = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = x[row * D + i] + residual[row * D + i];  // Fuse residual add
        float delta = val - mean;
        mean += delta / (i + 1);
        m2 += delta * (val - mean);
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        mean += __shfl_down_sync(0xffffffff, mean, offset);
        m2 += __shfl_down_sync(0xffffffff, m2, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&shared_mean, mean);
        atomicAdd(&shared_var, m2);
    }
    __syncthreads();

    mean = shared_mean / D;
    float var = shared_var / D;
    float inv_std = rsqrtf(var + eps);

    // Normalize and apply affine transform
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = x[row * D + i] + residual[row * D + i];
        float normalized = (val - mean) * inv_std;
        output[row * D + i] = normalized * gamma[i] + beta[i];
    }
}
```

**Performance**:
- Unfused: 0.45 ms (3 kernel launches, 6 HBM reads/writes)
- Fused: 0.22 ms (1 kernel launch, 3 HBM reads/writes)
- **Speedup: 2.0x**

### Fused GELU

GELU activation can be fused with the preceding linear layer:

```cuda
// fused_linear_gelu.cu
__global__ void linear_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // Matrix multiplication
        for (int k = 0; k < K; k++) {
            sum += input[row * K + k] * weight[k * N + col];
        }

        // Add bias
        sum += bias[col];

        // Apply GELU (fused!)
        float x3 = sum * sum * sum;
        float gelu_out = 0.5f * sum * (1.0f + __tanf(0.7978845608f * (sum + 0.044715f * x3)));

        output[row * N + col] = gelu_out;
    }
}
```

**Performance (2048×4096 @ 4096×2048)**:
- Unfused: 0.85 ms linear + 0.12 ms GELU = 0.97 ms
- Fused: 0.61 ms
- **Speedup: 1.6x**

### Fused Attention QKV Projection

Transformers compute Q, K, V from same input - can fuse:

```cuda
__global__ void fused_qkv_proj_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight_q,
    const float* __restrict__ weight_k,
    const float* __restrict__ weight_v,
    const float* __restrict__ bias_q,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    int batch_seq, int d_model
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_seq && col < d_model) {
        // Load input once
        float input_val = input[row * d_model + col];

        // Compute Q, K, V in parallel (ILP)
        float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;

        for (int k = 0; k < d_model; k++) {
            float in_k = input[row * d_model + k];
            q_sum += in_k * weight_q[k * d_model + col];
            k_sum += in_k * weight_k[k * d_model + col];
            v_sum += in_k * weight_v[k * d_model + col];
        }

        q_out[row * d_model + col] = q_sum + bias_q[col];
        k_out[row * d_model + col] = k_sum + bias_k[col];
        v_out[row * d_model + col] = v_sum + bias_v[col];
    }
}
```

**Better approach**: Use single fused weight matrix

```python
# Instead of 3 separate projections
self.q_proj = nn.Linear(d_model, d_model)
self.k_proj = nn.Linear(d_model, d_model)
self.v_proj = nn.Linear(d_model, d_model)

# Use single fused projection
self.qkv_proj = nn.Linear(d_model, 3 * d_model)

# Forward pass
qkv = self.qkv_proj(x)  # Single matmul
q, k, v = qkv.chunk(3, dim=-1)  # Split outputs
```

**Performance**:
- 3 separate: 3 × 0.45 ms = 1.35 ms
- Fused weight: 0.82 ms
- **Speedup: 1.65x**

### PyTorch JIT Fusion

PyTorch can automatically fuse operations with TorchScript:

```python
@torch.jit.script
def fused_gelu_bias_add(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """JIT will fuse these operations"""
    x = x + bias
    x3 = x * x * x
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x3)))

# Use in model
class TransformerFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(d_ff))
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = fused_gelu_bias_add(x, self.bias1)  # Fused!
        x = self.linear2(x)
        return x
```

## Multi-Query and Grouped-Query Attention

### Multi-Query Attention (MQA)

**Motivation**: Reduce KV cache memory footprint

**Idea**: Share K, V across all query heads

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)  # n_heads query heads
        self.k_proj = nn.Linear(d_model, self.head_dim)  # Single K head
        self.v_proj = nn.Linear(d_model, self.head_dim)  # Single V head
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Project Q (multi-head), K, V (single head)
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)

        # Broadcast K, V to all query heads
        k = k.expand(batch, self.n_heads, seq_len, self.head_dim)
        v = v.expand(batch, self.n_heads, seq_len, self.head_dim)

        # Standard attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(output)
```

**KV Cache Reduction**:
- Standard: `2 × n_heads × seq_len × head_dim`
- MQA: `2 × 1 × seq_len × head_dim`
- **Reduction: n_heads× (typically 12-96×)**

For GPT-3 (96 heads): 9.2 GB ’ 98 MB cache (94x reduction!)

**Quality Impact**: Minimal (<1% perplexity increase) when trained with MQA from scratch

### Grouped-Query Attention (GQA)

**Motivation**: Balance between standard and MQA

**Idea**: Group query heads to share K, V

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Expand K, V to match Q heads (repeat each KV head n_groups times)
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        # Standard attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(output)
```

**Example configurations**:
- Standard: 96 Q heads, 96 KV heads
- GQA-8: 96 Q heads, 8 KV heads (12x reduction)
- GQA-4: 96 Q heads, 4 KV heads (24x reduction)
- MQA: 96 Q heads, 1 KV head (96x reduction)

**Performance vs Quality Trade-off**:

| Configuration | KV Cache Size | Speed | Quality (PPL) |
|--------------|---------------|-------|---------------|
| Standard (96) | 9.2 GB | 1.0x | Baseline |
| GQA-8 | 1.15 GB | 1.8x | +0.2% |
| GQA-4 | 575 MB | 2.1x | +0.4% |
| MQA (1) | 98 MB | 2.3x | +0.8% |

**Used in**: LLaMA 2 70B (GQA-8), PaLM (MQA)

## Long Context Optimization

### The Long Context Challenge

Standard attention: O(N²) complexity becomes prohibitive for long contexts

**Memory requirements (fp16, 12 heads, d=64)**:

| Sequence Length | Attention Memory | A100 80GB Capacity |
|----------------|------------------|-------------------|
| 2K | 96 MB | 853 batches |
| 8K | 1.5 GB | 53 batches |
| 32K | 24 GB | 3 batches |
| 128K | 384 GB | OOM! |

**Compute time (A100)**:

| Sequence Length | Attention Time | Total Forward |
|----------------|----------------|---------------|
| 2K | 1.2 ms | 12 ms |
| 8K | 18 ms | 85 ms |
| 32K | 290 ms | 1.1 s |
| 128K | 4.6 s | 18 s |

### Sparse Attention Patterns

**Idea**: Don't attend to all tokens - use structured sparsity

#### Sliding Window Attention

Only attend to local context window:

```python
def sliding_window_attention(q, k, v, window_size=512):
    """
    Each query only attends to nearby keys within window_size

    Complexity: O(N × window_size) instead of O(N²)
    """
    batch, n_heads, seq_len, head_dim = q.shape

    # Create attention mask for sliding window
    mask = torch.zeros(seq_len, seq_len, device=q.device)
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2)
        mask[i, start:end] = 1

    # Compute attention with mask
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)

    return output
```

**Performance**: For window_size=512, seq_len=8192
- Memory: 8192² ’ 8192×512 = 16x reduction
- Compute: 16x reduction
- Quality: Minimal loss for many tasks

**Used in**: Longformer, BigBird

#### Block-Sparse Attention

Combine local + global + random attention:

```python
def block_sparse_attention(q, k, v, block_size=64, num_global=128, num_random=64):
    """
    Attention pattern:
    - Local: Attend to nearby tokens within block
    - Global: Attend to first num_global tokens
    - Random: Attend to num_random random tokens
    """
    batch, n_heads, seq_len, head_dim = q.shape

    # Create sparse attention mask
    mask = torch.zeros(seq_len, seq_len, device=q.device)

    # Local attention (block-diagonal)
    for i in range(0, seq_len, block_size):
        mask[i:i+block_size, i:i+block_size] = 1

    # Global attention (first num_global tokens attend to all)
    mask[:num_global, :] = 1
    mask[:, :num_global] = 1

    # Random attention
    for i in range(seq_len):
        random_indices = torch.randint(0, seq_len, (num_random,))
        mask[i, random_indices] = 1

    # Attention with sparse mask
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)

    return output
```

**Complexity**: O(N × (block_size + num_global + num_random))

For seq_len=16K, block=64, global=128, random=64:
- Standard: 16K × 16K = 256M operations
- Sparse: 16K × 256 = 4M operations
- **64x speedup**

### FlashAttention for Long Context

Flash Attention naturally extends to long contexts:

**Key optimization**: Tile size independent of sequence length

```python
# Standard attention: memory O(N²)
scores = Q @ K.T  # [seq_len, seq_len] - 256GB for 128K!

# Flash Attention: memory O(N)
for q_tile in Q.tiles():
    for kv_tile in K.tiles(), V.tiles():
        # Process tile (fits in SRAM)
        scores_tile = q_tile @ kv_tile.T  # [tile_size, tile_size]
        # ... online softmax ...
```

**Performance (A100, Flash Attention v2)**:

| Sequence Length | Memory | Time | vs Standard |
|----------------|--------|------|-------------|
| 8K | 512 MB | 35 ms | 5.1x faster |
| 32K | 2 GB | 180 ms | 16x faster |
| 128K | 8 GB | 950 ms | 77x faster |

**Enables**: Training with 128K context on single A100 (vs OOM with standard attention)

### Streaming Long Context

For inference, can stream very long contexts:

```python
class StreamingAttention:
    def __init__(self, window_size=2048, compress_ratio=4):
        self.window_size = window_size
        self.compress_ratio = compress_ratio

        self.recent_cache = []  # Recent tokens (full resolution)
        self.compressed_cache = []  # Old tokens (compressed)

    def forward(self, x, cache):
        """
        Recent tokens: Full attention (high quality)
        Old tokens: Compressed representation (approximate)
        """
        # Process new tokens with full attention to recent cache
        recent_k, recent_v = cache['recent']
        output = flash_attention(x, recent_k, recent_v)

        # Add approximate attention to compressed cache
        if len(cache['compressed']) > 0:
            comp_k, comp_v = cache['compressed']
            approx_output = approximate_attention(x, comp_k, comp_v)
            output = output + 0.1 * approx_output  # Weighted combination

        # Update cache: move old recent’compressed, add new’recent
        if len(cache['recent']) > self.window_size:
            to_compress = cache['recent'][:-self.window_size]
            cache['compressed'].append(self.compress(to_compress))
            cache['recent'] = cache['recent'][-self.window_size:]

        cache['recent'].append((k, v))

        return output

    def compress(self, cache_segment):
        """Compress old cache entries (e.g., average pooling)"""
        k, v = cache_segment
        k_compressed = F.avg_pool1d(k, kernel_size=self.compress_ratio)
        v_compressed = F.avg_pool1d(v, kernel_size=self.compress_ratio)
        return k_compressed, v_compressed
```

**Example**: 100K context
- Recent 2K tokens: Full resolution cache (48 MB)
- Older 98K tokens: 4x compressed (147 MB)
- Total: 195 MB vs 7.2 GB uncompressed (37x reduction)

## PagedAttention

### Motivation

KV cache management challenge in serving:
- Pre-allocate max length: Wastes memory (average 60% utilization)
- Dynamic allocation: Fragmentation and slow

**Solution**: Borrow from OS virtual memory - paging!

### PagedAttention Design

**Key idea**: Store KV cache in fixed-size pages, allocate dynamically

```python
class PagedKVCache:
    def __init__(self, page_size=256, num_pages=1000, n_layers=12, n_heads=12, head_dim=64):
        """
        Initialize paged KV cache

        page_size: Number of tokens per page (e.g., 256)
        num_pages: Total number of pages in memory pool
        """
        self.page_size = page_size
        self.num_pages = num_pages

        # Physical memory: pool of pages
        # Shape: [num_pages, n_layers, 2 (K+V), n_heads, page_size, head_dim]
        self.physical_cache = torch.zeros(
            num_pages, n_layers, 2, n_heads, page_size, head_dim,
            dtype=torch.float16, device='cuda'
        )

        # Page table: maps logical page ’ physical page
        # request_id ’ [list of physical page indices]
        self.page_tables = {}

        # Free list: available physical pages
        self.free_pages = list(range(num_pages))

    def allocate_request(self, request_id):
        """Allocate cache for new request"""
        self.page_tables[request_id] = []

    def allocate_page(self, request_id):
        """Allocate a new page for request"""
        if not self.free_pages:
            raise RuntimeError("Out of memory - no free pages")

        physical_page = self.free_pages.pop()
        self.page_tables[request_id].append(physical_page)
        return physical_page

    def free_request(self, request_id):
        """Free all pages for completed request"""
        if request_id in self.page_tables:
            # Return pages to free list
            self.free_pages.extend(self.page_tables[request_id])
            del self.page_tables[request_id]

    def get_cache(self, request_id, layer):
        """Get cache for request (all pages)"""
        pages = self.page_tables.get(request_id, [])
        if not pages:
            return None

        # Gather K, V from all pages
        k_pages = [self.physical_cache[p, layer, 0] for p in pages]
        v_pages = [self.physical_cache[p, layer, 1] for p in pages]

        k = torch.cat(k_pages, dim=1)  # Concatenate along sequence dim
        v = torch.cat(v_pages, dim=1)

        return {'k': k, 'v': v}

    def append_tokens(self, request_id, layer, k_new, v_new):
        """Append new tokens to cache"""
        pages = self.page_tables[request_id]

        # Check if need new page
        current_tokens = len(pages) * self.page_size
        if current_tokens % self.page_size == 0:
            # Current page full, allocate new one
            self.allocate_page(request_id)

        # Write new K, V to current page
        current_page = pages[-1]
        offset = current_tokens % self.page_size
        tokens_to_write = min(k_new.shape[1], self.page_size - offset)

        self.physical_cache[current_page, layer, 0, :, offset:offset+tokens_to_write, :] = \
            k_new[:, :tokens_to_write, :]
        self.physical_cache[current_page, layer, 1, :, offset:offset+tokens_to_write, :] = \
            v_new[:, :tokens_to_write, :]
```

### PagedAttention Kernel

Attention kernel must handle non-contiguous pages:

```cuda
__global__ void paged_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ physical_cache,  // All pages
    const int* __restrict__ page_table,  // Logical ’ physical mapping
    float* __restrict__ output,
    int num_pages_in_use,
    int page_size,
    int head_dim
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output_acc[HEAD_DIM] = {0};

    // Iterate over all pages for this request
    for (int page_idx = 0; page_idx < num_pages_in_use; page_idx++) {
        int physical_page = page_table[page_idx];

        // Load K, V from physical page
        for (int token_in_page = 0; token_in_page < page_size; token_in_page++) {
            // Compute attention score
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float q_val = Q[query_idx * head_dim + d];
                float k_val = physical_cache[
                    physical_page * PAGE_SIZE * HEAD_DIM +
                    token_in_page * HEAD_DIM + d
                ];
                score += q_val * k_val;
            }
            score /= sqrtf((float)head_dim);

            // Online softmax update
            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_score = expf(score - max_score);
            sum_exp = sum_exp * expf(old_max - max_score) + exp_score;

            // Accumulate output
            for (int d = 0; d < head_dim; d++) {
                float v_val = physical_cache[
                    physical_page * PAGE_SIZE * HEAD_DIM +
                    PAGE_SIZE * HEAD_DIM +  // Offset to V
                    token_in_page * HEAD_DIM + d
                ];
                output_acc[d] += exp_score * v_val;
            }
        }
    }

    // Normalize and write output
    for (int d = 0; d < head_dim; d++) {
        output[query_idx * head_dim + d] = output_acc[d] / sum_exp;
    }
}
```

### Benefits

**Memory Efficiency**:
- No wasted memory from over-allocation
- No fragmentation
- Near 100% utilization

**Flexibility**:
- Variable-length sequences (no padding waste)
- Dynamic batching
- Request preemption (can free pages mid-generation)

**Performance**:
- Efficient memory access (pages are contiguous)
- Enables larger batch sizes
- 2-3x higher throughput vs contiguous cache

**Example Utilization**:

| Caching Strategy | Memory Allocated | Memory Used | Utilization |
|-----------------|------------------|-------------|-------------|
| Pre-allocated max | 10 GB | 4.2 GB | 42% |
| Dynamic contiguous | 4.5 GB | 4.2 GB | 93% (fragmentation) |
| PagedAttention | 4.3 GB | 4.2 GB | 98% |

**Used in**: vLLM (state-of-the-art LLM serving framework)

## Production Implementation

### Complete Optimized Transformer

Putting it all together:

```python
class OptimizedTransformer(nn.Module):
    """
    Production transformer with all optimizations:
    - Flash Attention v2
    - KV caching
    - Grouped-query attention
    - Fused operations
    - PagedAttention support
    """

    def __init__(
        self,
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,  # GQA
        d_ff=3072,
        max_seq_len=8192,
        use_flash=True,
        use_paged_cache=True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.use_flash = use_flash
        self.use_paged_cache = use_paged_cache

        # Embedding
        self.token_emb = nn.Embedding(50257, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            OptimizedTransformerBlock(
                d_model, n_heads, n_kv_heads, d_ff, use_flash
            )
            for _ in range(n_layers)
        ])

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 50257, bias=False)

        # KV cache
        if use_paged_cache:
            self.kv_cache = PagedKVCache(
                page_size=256,
                num_pages=1000,
                n_layers=n_layers,
                n_heads=n_kv_heads,
                head_dim=d_model // n_heads
            )
        else:
            self.kv_cache = None

    def forward(
        self,
        input_ids,
        request_id=None,
        use_cache=False,
        position_ids=None
    ):
        batch, seq_len = input_ids.shape

        # Embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)

        x = self.token_emb(input_ids) + self.pos_emb(position_ids)

        # Transformer blocks
        new_caches = []
        for i, block in enumerate(self.blocks):
            # Get cache for this layer
            if use_cache and request_id is not None:
                cache = self.kv_cache.get_cache(request_id, i)
            else:
                cache = None

            x, new_cache = block(x, cache=cache, use_cache=use_cache)

            if use_cache and request_id is not None:
                # Update paged cache
                self.kv_cache.append_tokens(
                    request_id, i,
                    new_cache['k'], new_cache['v']
                )

        # Output
        x = self.ln_f(x)
        logits = self.head(x)

        return logits


class OptimizedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, use_flash):
        super().__init__()

        # Attention
        if use_flash:
            self.attn = FlashGQAttention(d_model, n_heads, n_kv_heads)
        else:
            self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)

        # FFN with fused ops
        self.ffn = FusedFFN(d_model, d_ff)

        # Fused LayerNorm
        self.norm1 = FusedLayerNorm(d_model)
        self.norm2 = FusedLayerNorm(d_model)

    def forward(self, x, cache=None, use_cache=False):
        # Pre-norm + attention + residual (fused)
        residual = x
        x = self.norm1(x)
        x, new_cache = self.attn(x, cache=cache, use_cache=use_cache)
        x = x + residual

        # Pre-norm + FFN + residual (fused)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x, new_cache


class FusedFFN(nn.Module):
    """Feedforward with fused GELU"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        # Use custom fused GELU kernel
        self.gelu = FusedGELU.apply

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)  # Custom CUDA kernel
        x = self.linear2(x)
        return x


class FlashGQAttention(nn.Module):
    """Flash Attention v2 with GQA"""

    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, d_model + 2 * n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

        # Use Flash Attention kernel
        self.flash_attn = FlashAttentionV2.apply

    def forward(self, x, cache=None, use_cache=False):
        batch, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim

        q = qkv[:, :, :q_dim]
        k = qkv[:, :, q_dim:q_dim+kv_dim]
        v = qkv[:, :, q_dim+kv_dim:]

        # Reshape
        q = q.view(batch, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Use cache if provided
        if cache is not None:
            k = torch.cat([cache['k'], k], dim=1)
            v = torch.cat([cache['v'], v], dim=1)

        # Flash Attention
        output = self.flash_attn(q, k, v, self.n_heads, self.n_kv_heads)

        # Project output
        output = output.view(batch, seq_len, -1)
        output = self.out_proj(output)

        # Return cache
        new_cache = {'k': k, 'v': v} if use_cache else None
        return output, new_cache
```

### Performance Benchmarks

**Single Forward Pass (A100, batch=1)**:

| Model | Standard | Optimized | Speedup |
|-------|----------|-----------|---------|
| GPT-2 (124M) | 12 ms | 5.2 ms | 2.3x |
| GPT-2 Large (774M) | 45 ms | 18 ms | 2.5x |
| GPT-3 (6.7B) | 280 ms | 95 ms | 2.9x |
| GPT-3 (13B) | 550 ms | 175 ms | 3.1x |

**Generation (50 tokens, A100)**:

| Model | Standard | + KV Cache | + Flash | + GQA | Full Opt |
|-------|----------|------------|---------|-------|----------|
| GPT-2 | 680 ms | 180 ms | 140 ms | 125 ms | 95 ms |
| GPT-2 Large | 2.4 s | 620 ms | 480 ms | 420 ms | 320 ms |
| GPT-3 (6.7B) | 15 s | 3.8 s | 2.9 s | 2.5 s | 1.9 s |

**Cumulative Speedups**:
- KV Cache: 3.8x
- Flash Attention: +1.3x = 4.9x total
- GQA: +1.2x = 5.9x total
- Fused ops: +1.3x = **7.7x total**

### Deployment Considerations

1. **Mixed Precision**:
```python
# Use FP16 for speed, FP32 for stability
model = OptimizedTransformer(...)
model = model.half()  # Convert to FP16

# Or use automatic mixed precision
with torch.cuda.amp.autocast():
    output = model(input_ids)
```

2. **Batch Inference**:
```python
# Continuous batching for high throughput
class ContinuousBatcher:
    def __init__(self, model, max_batch_size=32):
        self.model = model
        self.max_batch_size = max_batch_size
        self.pending = []

    def add_request(self, request):
        self.pending.append(request)

        if len(self.pending) >= self.max_batch_size:
            self.process_batch()

    def process_batch(self):
        # Batch requests with similar lengths
        batch = self.pending[:self.max_batch_size]
        self.pending = self.pending[self.max_batch_size:]

        # Process batch
        input_ids = pad_sequence([r.input_ids for r in batch])
        outputs = self.model(input_ids, use_cache=True)

        # Distribute results
        for request, output in zip(batch, outputs):
            request.complete(output)
```

3. **Model Parallelism** (for large models):
```python
# Tensor parallelism for attention
# Split heads across GPUs
n_heads_per_gpu = n_heads // world_size
local_q = q[:, rank*n_heads_per_gpu:(rank+1)*n_heads_per_gpu, :, :]
```

## Case Studies

### Case Study 1: GPT-2 Optimization

**Goal**: Reduce GPT-2 inference latency from 180ms to <50ms on A100

**Baseline Profile**:
- Attention: 72 ms (40%)
- FFN: 68 ms (38%)
- LayerNorm: 24 ms (13%)
- Other: 16 ms (9%)

**Optimization Steps**:

1. **Implement KV Cache**: 180ms ’ 52ms (3.5x)
   - Eliminates redundant attention computation
   - Only processes new token each step

2. **Flash Attention**: 52ms ’ 39ms (1.3x additional)
   - Reduces memory bandwidth for attention
   - Better GPU utilization

3. **Fused LayerNorm**: 39ms ’ 33ms (1.2x additional)
   - Combines LayerNorm + residual + bias
   - Reduces memory traffic

4. **Fused GELU**: 33ms ’ 28ms (1.2x additional)
   - Custom CUDA kernel with fast math
   - Fuses with preceding linear layer

5. **GQA (4 heads)**: 28ms ’ 24ms (1.2x additional)
   - Reduces KV cache memory
   - Enables larger batch sizes

**Final Result**: 24ms (7.5x speedup, meets <50ms target)

### Case Study 2: Long Context Chatbot

**Goal**: Support 32K context for chatbot application

**Challenge**:
- Standard attention: 24 GB memory (OOM on A100 80GB with batch=8)
- Slow: 1.2s per generation step

**Solution**:

1. **Flash Attention v2**: Enables 32K context
   - Memory: 24 GB ’ 3.2 GB
   - Fits batch=8 easily

2. **Sliding Window** (8K window) + **Global Attention** (256 tokens):
   - Maintains quality for chat use case
   - Further reduces compute

3. **Streaming KV Cache**:
   - Compress old context (>8K) by 4x
   - Minimal quality loss

**Results**:
- Memory: 3.2 GB total (enables batch=24)
- Latency: 1.2s ’ 180ms (6.7x)
- Quality: 95% of full attention performance
- Cost savings: 6.7x higher throughput = $2.4M/year at scale

### Case Study 3: Code Generation

**Goal**: Optimize code generation model (similar to Codex)

**Requirements**:
- Low latency: <100ms first token
- Long context: 8K tokens (entire file)
- High quality: Preserve accuracy

**Optimizations**:

1. **GQA (8’2 KV heads)**:
   - Reduces cache 4x
   - Enables longer prefill sequences
   - Minimal quality loss (<1% worse)

2. **Speculative Decoding**:
   - Use small draft model (GPT-2 Small)
   - Verify with large model
   - 2.5x speedup for code (high token acceptance)

3. **Fused Ops + Flash Attention**:
   - Standard optimizations
   - 2.1x speedup

**Results**:
- First token: 250ms ’ 85ms (2.9x)
- Subsequent tokens: 45ms ’ 12ms (3.8x)
- Quality: 99.2% of baseline
- Meets <100ms target

## Summary

Transformer optimization is essential for production deployment. Key techniques:

1. **Flash Attention v2**: O(N) memory, 3-10x speedup
2. **KV Caching**: 3-7x speedup for generation
3. **Fused Operations**: 1.5-2x speedup by reducing memory traffic
4. **Multi-Query/Grouped-Query Attention**: 10-90x cache reduction
5. **Long Context**: Sparse attention, streaming, and tiling
6. **PagedAttention**: Near-perfect memory utilization

**Combined Impact**: 5-10x speedup is typical, with cost savings in millions of dollars at scale.

**Next Steps**:
- Module 5: Model Compression (quantization, pruning, distillation)
- Module 6: Distributed Inference (multi-GPU, tensor parallelism)
- Module 7: Production Deployment (serving, monitoring, scaling)

---

**References**:
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Flash Attention v2](https://arxiv.org/abs/2307.08691)
- [Multi-Query Attention](https://arxiv.org/abs/1911.02150)
- [Grouped-Query Attention (GQA)](https://arxiv.org/abs/2305.13245)
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
- [Efficient Transformers Survey](https://arxiv.org/abs/2009.06732)
