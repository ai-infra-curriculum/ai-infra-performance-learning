# Lecture 01: Attention Optimization

## Why attention is the bottleneck

Standard attention: `softmax(QK^T / sqrt(d_k)) V` — compute and memory are
both O(N²) in sequence length. At N=8192 this dominates training cost.

## FlashAttention

Re-formulates attention to:
- Tile-based: compute attention block-by-block in SRAM
- Memory: O(N) instead of O(N²); intermediate matrices never materialized
- Speed: 2-4× faster on A100/H100 for long sequences

Available in PyTorch as `torch.nn.functional.scaled_dot_product_attention`.

```python
# Drop-in replacement
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Use this everywhere; don't hand-roll attention.

## KV cache

During autoregressive generation, K and V from prior tokens are reused. Cache
them; only compute new tokens' Q.

Memory cost: `batch × seq_len × layers × heads × head_dim × 2 (K+V) × precision_bytes`.

At seq_len=4096, layers=32, heads=32, head_dim=128, bf16:
4096 × 32 × 32 × 128 × 2 × 2 = ~2.1 GB per request.

## Paged attention (vLLM)

KV cache is large + variable per request. Naive allocation fragments memory.
PagedAttention uses fixed-size pages (like virtual memory) — each request's
KV cache is a list of page indices.

Result: 2-4× more concurrent requests on the same GPU.
