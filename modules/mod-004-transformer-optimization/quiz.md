# Module 04 — Quiz

10 questions. 70% pass.

### 1. FlashAttention reduces attention memory from:
- [x] a) O(N²) to O(N)
- [ ] b) O(N) to O(log N)
- [ ] c) O(N²) to O(N²) — same; only speed improved
- [ ] d) O(1) always

### 2. PagedAttention's value:
- [x] a) Fewer fragmented KV-cache allocations → more concurrent requests
- [ ] b) Smaller models
- [ ] c) Faster training
- [ ] d) Better accuracy

### 3. Speculative decoding requires:
- [x] a) A small draft model sharing tokenizer with the target
- [ ] b) Two equal-size models
- [ ] c) Pre-trained vocab swap
- [ ] d) Always slower

### 4. Continuous batching speeds serving by:
- [x] a) Dropping finished requests + adding new ones at every step (no fixed batches)
- [ ] b) Padding all requests to the longest
- [ ] c) Batching by time of day
- [ ] d) Disabling batching

### 5. Prefix caching benefits:
- [x] a) Workloads where many requests share a system prompt
- [ ] b) Random unique prompts
- [ ] c) Training only
- [ ] d) Embedding generation

### 6. torch.compile mode="reduce-overhead":
- [x] a) Enables CUDA graphs + aggressive fusion
- [ ] b) Disables optimization
- [ ] c) Reduces memory only
- [ ] d) Required for inference

### 7. KV cache size scales with:
- [x] a) batch × seq_len × layers × heads × head_dim × 2 × precision_bytes
- [ ] b) Just batch size
- [ ] c) Only sequence length
- [ ] d) Constant

### 8. Triton is:
- [x] a) A Python-style language for GPU kernel authoring
- [ ] b) An LLM
- [ ] c) An inference server
- [ ] d) A profiler

### 9. The combined speedup (FlashAttn + continuous batching + prefix cache + spec decode):
- [x] a) Compounds; realistic 20-30× over HuggingFace baseline
- [ ] b) Sums; rarely > 2×
- [ ] c) Cannot be combined
- [ ] d) Always 100×

### 10. The right place to start optimizing transformer inference:
- [x] a) Use vLLM defaults; profile + tune from there
- [ ] b) Write hand-rolled CUDA from scratch
- [ ] c) Add more GPUs first
- [ ] d) Quantize to int4 immediately

---

Answers: all `a`
