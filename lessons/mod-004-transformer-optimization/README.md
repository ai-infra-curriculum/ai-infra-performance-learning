# Module 4: Transformer Optimization

## Overview

This module focuses on state-of-the-art optimization techniques for transformer models, covering Flash Attention, KV caching, kernel fusion, and production deployment strategies. You'll learn to achieve 5-10x speedups while maintaining numerical accuracy.

**Duration**: 5-6 weeks (25-30 hours)

## Prerequisites

- Completed Modules 1-3
- Strong understanding of transformer architecture
- Experience with custom CUDA kernels
- Knowledge of attention mechanisms
- Proficiency in PyTorch

## Learning Objectives

By completing this module, you will:

1. **Understand Transformer Bottlenecks**: Identify memory and compute bottlenecks in attention
2. **Implement Flash Attention**: Build O(N) memory attention using tiling and online softmax
3. **Master KV Caching**: Achieve 3-7x speedup for autoregressive generation
4. **Apply Kernel Fusion**: Reduce memory bandwidth with fused operations
5. **Optimize Long Contexts**: Handle 16K-128K token sequences efficiently
6. **Deploy Production Systems**: Build high-throughput serving infrastructure

## Module Structure

### Lecture Notes

**01-transformer-optimization.md** (14,000+ words)
- Transformer architecture review and bottleneck analysis
- Flash Attention v2 algorithm and implementation
- KV cache optimization for generation
- Fused operations (LayerNorm, GELU, QKV projection)
- Multi-Query and Grouped-Query Attention
- Long context techniques (sparse attention, streaming cache)
- PagedAttention for efficient serving
- Production case studies with performance data

### Exercises

5 comprehensive hands-on exercises (26 hours total):

1. **KV Cache Implementation** (4h) - Build efficient caching for generation
2. **Flash Attention Basics** (6h) - Implement tiling and online softmax
3. **Grouped-Query Attention** (3h) - Reduce cache memory with GQA
4. **Long Context Optimization** (5h) - Handle 16K+ token contexts
5. **Production Transformer** (8h) - End-to-end optimized system

## Key Techniques

### Flash Attention v2

Reduces attention memory from O(N²) to O(N):

```python
# Standard: Materializes full attention matrix
scores = Q @ K.T  # [N, N] - 256GB for 128K tokens!
attn = softmax(scores)
output = attn @ V

# Flash Attention: Tiled computation in SRAM
# Never materializes N×N matrix
# Uses online softmax for efficiency
```

**Impact**:
- Memory: O(N²) → O(N) (1000x reduction for N=32K)
- Speed: 3-12x faster (increases with sequence length)
- Enables: 128K context on single A100

### KV Cache Optimization

Eliminates redundant computation in autoregressive generation:

```python
# Without cache: Recompute everything
for i in range(max_tokens):
    logits = model(tokens[:i+1])  # Processes all tokens!

# With cache: Only process new token
cache = None
for i in range(max_tokens):
    logits, cache = model(tokens[i:i+1], cache=cache)  # Only 1 token!
```

**Impact**:
- Speed: 3-7x faster (50 tokens: 180ms → 45ms)
- Memory: +2-3% for cache storage
- Used by: All production LLM serving systems

### Grouped-Query Attention

Reduces KV cache memory by sharing K, V heads:

| Configuration | KV Heads | Cache Size | Speed | Quality |
|--------------|----------|------------|-------|---------|
| Standard MHA | 96 | 9.2 GB | 1.0x | Baseline |
| GQA-8 | 8 | 1.15 GB | 1.8x | -0.2% |
| GQA-4 | 4 | 575 MB | 2.1x | -0.4% |
| MQA | 1 | 98 MB | 2.3x | -0.8% |

**Used in**: LLaMA 2 70B (GQA-8), PaLM (MQA)

### Long Context Strategies

**Sliding Window**: O(N × window) complexity
- Each token attends to local window only
- 16x reduction for window=512, seq=8K

**Block-Sparse**: Combine local + global + random
- Local blocks + global tokens + optional dilated
- 64x reduction with minimal quality loss

**Streaming Cache**: Compress old context
- Recent: Full resolution (2K tokens)
- Old: 4x compressed (remaining tokens)
- 37x memory reduction for 100K context

## Performance Benchmarks

### Single Forward Pass (A100)

| Model | Standard | Optimized | Speedup |
|-------|----------|-----------|---------|
| GPT-2 (124M) | 12 ms | 5.2 ms | 2.3x |
| GPT-2 Large | 45 ms | 18 ms | 2.5x |
| GPT-3 (6.7B) | 280 ms | 95 ms | 2.9x |

### Generation (50 tokens, A100)

| Model | No Cache | + Cache | + Flash | + GQA | Full Opt |
|-------|----------|---------|---------|-------|----------|
| GPT-2 | 680 ms | 180 ms | 140 ms | 125 ms | 95 ms |
| GPT-3 6.7B | 15 s | 3.8 s | 2.9 s | 2.5 s | 1.9 s |

**Cumulative**: 7.7x speedup with all optimizations

### Long Context (Flash Attention v2)

| Sequence Length | Memory | Time | vs Standard |
|----------------|--------|------|-------------|
| 8K | 512 MB | 35 ms | 5.1x faster |
| 32K | 2 GB | 180 ms | 16x faster |
| 128K | 8 GB | 950 ms | 77x faster |

## Case Studies

### Case Study 1: GPT-2 Latency Reduction

**Goal**: 180ms → <50ms on A100

**Approach**:
1. KV Cache: 180ms → 52ms (3.5x)
2. Flash Attention: 52ms → 39ms (+1.3x)
3. Fused LayerNorm: 39ms → 33ms (+1.2x)
4. Fused GELU: 33ms → 28ms (+1.2x)
5. GQA-4: 28ms → 24ms (+1.2x)

**Result**: 24ms (7.5x total, meets <50ms target)

### Case Study 2: 32K Context Chatbot

**Challenge**: Standard attention OOM on A100 80GB

**Solution**:
- Flash Attention v2: 24GB → 3.2GB memory
- Sliding Window (8K) + Global (256): Further compute reduction
- Streaming Cache: Compress >8K context by 4x

**Result**:
- Memory: Fits batch=24 (was OOM at batch=8)
- Latency: 1.2s → 180ms (6.7x)
- Quality: 95% of full attention
- Cost savings: $2.4M/year at scale

### Case Study 3: Code Generation Model

**Requirements**: <100ms first token, 8K context

**Optimizations**:
- GQA (8→2 heads): 4x cache reduction
- Speculative decoding: 2.5x speedup
- Flash Attention + fused ops: 2.1x speedup

**Result**:
- First token: 250ms → 85ms (2.9x)
- Subsequent: 45ms → 12ms (3.8x)
- Meets <100ms target

## Tools and Setup

### Required Software

```bash
# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Flash Attention (official implementation)
pip install flash-attn --no-build-isolation

# Other dependencies
pip install transformers accelerate pytest pytest-benchmark
```

### Recommended Hardware

- **Minimum**: NVIDIA A100 40GB
- **Recommended**: A100 80GB or H100 for long contexts
- **For Development**: RTX 4090 (limited memory)

## Production Considerations

### Memory Management

```python
# Use PagedAttention for efficient cache management
cache = PagedKVCache(
    page_size=256,
    num_pages=1000,
    n_layers=12,
    n_heads=4  # GQA
)

# Automatic cache eviction and reuse
request_id = cache.allocate_request()
# ... generate ...
cache.free_request(request_id)  # Return pages to pool
```

### Batched Serving

```python
# Continuous batching for high throughput
class ContinuousBatcher:
    def __init__(self, model, max_batch_size=32):
        self.model = model
        self.pending = []
        self.active = {}

    def step(self):
        # Form batch, run inference
        # Remove completed, add new requests
        pass
```

**Impact**: 10-20x throughput increase vs sequential

### Monitoring

```python
# Production metrics
metrics = {
    'p50_latency_ms': 45,
    'p95_latency_ms': 120,
    'p99_latency_ms': 250,
    'throughput_req_s': 150,
    'gpu_utilization': 0.85,
    'cache_hit_rate': 0.92
}
```

## Best Practices

1. **Start with KV Cache**: Easiest optimization, 3-5x speedup
2. **Profile Before Optimizing**: Use Nsight to find actual bottlenecks
3. **Test Numerical Accuracy**: Always validate outputs match baseline
4. **Batch When Possible**: Batching provides 10-20x throughput gains
5. **Monitor Production**: Track latency percentiles and cache efficiency

## Common Pitfalls

### Pitfall 1: Cache Memory Explosion

**Problem**: Linear memory growth with context length

**Solution**:
- Use GQA or MQA to reduce cache size
- Implement cache compression for old tokens
- Set max context length limits

### Pitfall 2: Flash Attention Overhead

**Problem**: Flash Attention slower for short sequences (<512 tokens)

**Solution**:
- Use standard attention for short sequences
- Switch to Flash Attention above threshold
- Profile to find optimal switching point

### Pitfall 3: Quality Degradation

**Problem**: Optimizations hurt model quality

**Solution**:
```python
# Always validate
baseline_perplexity = 15.2
optimized_perplexity = 15.4
assert optimized_perplexity < baseline_perplexity * 1.02, "Quality regression!"
```

## Additional Resources

### Papers

- [Flash Attention](https://arxiv.org/abs/2205.14135) - Original Flash Attention paper
- [Flash Attention v2](https://arxiv.org/abs/2307.08691) - Improved algorithm
- [GQA](https://arxiv.org/abs/2305.13245) - Grouped-Query Attention
- [PagedAttention](https://arxiv.org/abs/2309.06180) - vLLM serving system
- [Efficient Transformers Survey](https://arxiv.org/abs/2009.06732) - Comprehensive overview

### Implementations

- [flash-attn](https://github.com/Dao-AILab/flash-attention) - Official Flash Attention
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
- [TGI](https://github.com/huggingface/text-generation-inference) - HuggingFace serving
- [xFormers](https://github.com/facebookresearch/xformers) - Memory-efficient operations

### Documentation

- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [LLM.int8()](https://arxiv.org/abs/2208.07339) - Quantization for transformers

## Assessment

### Knowledge Check

1. Why is attention memory O(N²)? How does Flash Attention reduce it to O(N)?
2. Explain KV caching and why it provides speedup for generation
3. What is the trade-off between GQA-4 and GQA-2?
4. When should you use sparse attention vs full attention?
5. How does PagedAttention improve memory utilization?

### Practical Assessment

1. Implement KV caching from scratch (3-5x speedup)
2. Build Flash Attention with online softmax
3. Create GQA module with configurable n_kv_heads
4. Optimize transformer for 32K context
5. Deploy production serving system with batching

## Next Steps

After completing this module:

1. **Module 5: Model Compression** - Quantization, pruning, distillation
2. **Module 6: Distributed Inference** - Multi-GPU serving, tensor parallelism
3. **Module 7: Production Deployment** - Monitoring, auto-scaling, cost optimization
4. **Module 8: Advanced Topics** - Custom hardware, research frontiers

## Summary

Transformer optimization is critical for production LLM deployment. This module covers:

- **Flash Attention**: O(N) memory, 3-12x faster
- **KV Caching**: 3-7x generation speedup
- **GQA**: 3-12x cache reduction
- **Long Context**: Techniques for 128K+ tokens
- **Production**: High-throughput serving systems

**Combined Impact**: 5-10x speedup typical, with millions in cost savings at scale.

**Time Investment**: 25-30 hours
**Expected Outcome**: Production-ready transformer optimization skills
**Industry Relevance**: Critical for all LLM serving systems

---

**Module Maintainer**: AI Infrastructure Curriculum Team
**Last Updated**: 2025
**Feedback**: Please report issues via GitHub Issues
