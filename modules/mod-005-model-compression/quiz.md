# Module 05 — Quiz

10 questions. 70% pass.

### 1. AWQ is most useful for:
- [ ] a) CPU-only inference acceleration
- [ ] b) Training-time acceleration
- [ ] c) Embedding model compression
- [x] d) Weight-only int4 quantization of large language models

### 2. The hardware feature that natively accelerates 2:4 structured sparsity is:
- [x] a) NVIDIA Sparse Tensor Cores (A100 generation onward)
- [ ] b) Any GPU with CUDA support
- [ ] c) CPU AVX-512 instructions
- [ ] d) TPU-only

### 3. QLoRA combines:
- [x] a) 4-bit base model quantization + LoRA fine-tuning
- [ ] b) Quantization-aware training + pruning
- [ ] c) Knowledge distillation + structured sparsity
- [ ] d) Pruning + post-training quantization

### 4. Distillation requires:
- [ ] a) A smaller training dataset
- [ ] b) Fewer GPUs than full training
- [ ] c) A different optimizer
- [x] d) A teacher model that already performs well on the target task

### 5. Typical AWQ int4 accuracy hit on a 7B+ LLM:
- [x] a) < 0.5pp drop on most academic benchmarks
- [ ] b) 5-10pp drop (substantial regression)
- [ ] c) Unmeasurable — no tooling exists
- [ ] d) > 20pp drop (model effectively broken)

### 6. A LoRA adapter for a 7B model at rank 16 is roughly:
- [ ] a) The same size as the base model
- [ ] b) Larger than the base model
- [ ] c) Zero bytes (LoRA stores nothing on disk)
- [x] d) ~50 MB

### 7. vLLM's LoRA hot-swap supports:
- [ ] a) Replacing the vLLM server with a custom one per adapter
- [ ] b) Requiring full retraining per request
- [ ] c) Latency that scales linearly with the number of adapters loaded
- [x] d) One base model + N adapters in memory; clients pick by name; sub-100ms first-request penalty

### 8. To make weights smaller while keeping architecture identical:
- [x] a) Quantization
- [ ] b) Distillation (changes the model)
- [ ] c) Pruning (changes weights to zeros — architecture varies)
- [ ] d) Fine-tuning (orthogonal to size)

### 9. To produce a smaller model with a different architecture:
- [ ] a) Quantization
- [ ] b) Pruning
- [ ] c) LoRA adaptation
- [x] d) Distillation

### 10. Stacking int8 quantization + 2:4 sparsity on a model:
- [ ] a) Cannot be combined
- [ ] b) Halves accuracy
- [ ] c) Doubles inference latency
- [x] d) Combines memory savings: ~6× reduction overall

---

## Answer key + rationale

1. **d** — AWQ is designed around LLM weight distributions; the activation-aware variant matters at scale.
2. **a** — Sparse Tensor Cores are the hardware path that makes 2:4 sparsity faster than dense compute.
3. **a** — QLoRA is the combo that puts 13B fine-tuning on a single 24GB GPU.
4. **d** — Distillation inherits the teacher's quality; a bad teacher produces a bad student.
5. **a** — AWQ's quality preservation is its main selling point vs naive int4.
6. **d** — Rank 16 → small adapter matrices → ~50MB even on 7B base.
7. **d** — Hot-swap is vLLM's killer multi-tenant feature; per-adapter latency is constant after first load.
8. **a** — Quantization is the only method that doesn't change the architecture or the trained weights' meaning.
9. **d** — Distillation lets you choose a smaller architecture and train it to mimic the larger one.
10. **d** — Methods are largely orthogonal; combined savings stack (not perfectly, but close).
