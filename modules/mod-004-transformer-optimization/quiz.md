# Module 04 — Quiz

10 questions. 70% pass.

### 1. FlashAttention reduces attention memory from:
- [x] a) O(N²) to O(N)
- [ ] b) O(N) to O(log N)
- [ ] c) O(N²) to O(N²) — only speed improved
- [ ] d) O(1) regardless of sequence length

### 2. PagedAttention's primary value is:
- [ ] a) Smaller model size
- [ ] b) Faster training
- [ ] c) Better generation accuracy
- [x] d) Eliminates KV-cache fragmentation → more concurrent requests per GPU

### 3. Speculative decoding requires:
- [ ] a) Two equal-size models for verification
- [ ] b) A pre-trained vocab swap step
- [ ] c) That it always be slower than greedy decoding
- [x] d) A small draft model that shares a tokenizer with the target model

### 4. Continuous batching speeds LLM serving by:
- [ ] a) Padding all requests in a batch to the longest sequence length
- [ ] b) Batching by time-of-day buckets
- [ ] c) Disabling batching entirely
- [x] d) Dropping finished requests + adding new ones at every step (no fixed batches)

### 5. Prefix caching benefits workloads where:
- [x] a) Many requests share a common system prompt or context
- [ ] b) Every prompt is uniquely random
- [ ] c) Only during training
- [ ] d) Only for embedding generation

### 6. `torch.compile(model, mode="reduce-overhead")`:
- [ ] a) Disables most optimizations
- [ ] b) Reduces memory only, not latency
- [ ] c) Is required for any inference workload
- [x] d) Enables CUDA graphs + aggressive fusion to cut per-launch overhead

### 7. KV cache size scales as:
- [x] a) batch × seq_len × num_layers × num_heads × head_dim × 2 (K + V) × precision_bytes
- [ ] b) Just batch size
- [ ] c) Only sequence length
- [ ] d) A constant regardless of inputs

### 8. Triton is:
- [ ] a) An LLM developed by Anthropic
- [ ] b) A vendored inference server (NVIDIA)
- [ ] c) A profiler for GPU kernels
- [x] d) A Python-style language + JIT compiler for authoring GPU kernels

### 9. The combined speedup from FlashAttn + continuous batching + prefix cache + spec decode is typically:
- [ ] a) Strictly additive; rarely exceeds 2×
- [ ] b) Cannot be combined (mutually exclusive)
- [ ] c) Always exactly 100×
- [x] d) Roughly multiplicative; realistic 20-30× over a HuggingFace baseline

### 10. The right place to start optimizing transformer inference is:
- [x] a) Use vLLM defaults; profile + tune from there
- [ ] b) Write hand-rolled CUDA kernels from scratch
- [ ] c) Add more GPUs before tuning
- [ ] d) Quantize to int4 immediately, no profiling

---

## Answer key + rationale

1. **a** — Tiling lets attention compute over blocks that fit in SRAM; the N×N matrix is never materialized.
2. **d** — Paged allocation avoids the fragmentation that capped concurrent request count in vanilla allocators.
3. **d** — Without shared tokenizer the verification step can't compare token-for-token.
4. **d** — That dropping/adding step is exactly what removes padding waste.
5. **a** — Cached prefix reuse cuts the most expensive part of attention for shared-prompt workloads (RAG, agents, chat).
6. **d** — `reduce-overhead` is the CUDA-graphs-enabled mode of torch.compile.
7. **a** — Multiplicative formula; memorize it for capacity planning. Doubles for K + V.
8. **d** — Triton is OpenAI's Python-style GPU kernel DSL; powers FlashAttention v2 + vLLM internals.
9. **d** — Optimizations stack roughly multiplicatively; published vLLM benchmarks show ~20-30× over HF transformers baselines.
10. **a** — vLLM defaults capture 80% of the wins; hand-rolling CUDA before profiling is premature.
