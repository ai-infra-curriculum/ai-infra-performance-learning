# Module 05 — Quiz

10 questions. 70% pass.

### 1. AWQ is most useful for:
- [x] a) Weight-only int4 quantization of large LLMs
- [ ] b) CPU-only inference
- [ ] c) Training acceleration
- [ ] d) Embedding generation

### 2. The hardware that accelerates 2:4 sparsity is:
- [x] a) NVIDIA Sparse Tensor Cores (A100+)
- [ ] b) Any GPU
- [ ] c) CPU AVX-512
- [ ] d) TPU only

### 3. QLoRA combines:
- [x] a) 4-bit base model quantization + LoRA fine-tuning
- [ ] b) Quantization-aware training + pruning
- [ ] c) Distillation + structured sparsity
- [ ] d) Pruning + quantization

### 4. Distillation requires:
- [x] a) A teacher model that already performs well on the target task
- [ ] b) A smaller dataset
- [ ] c) Fewer GPUs
- [ ] d) Different hyperparameters

### 5. The typical accuracy hit of AWQ int4 on a 7B+ model:
- [x] a) < 0.5pp on most benchmarks
- [ ] b) 5-10pp
- [ ] c) Unmeasurable
- [ ] d) > 20pp

### 6. LoRA adapter size for r=16 on a 7B model:
- [x] a) ~50 MB
- [ ] b) Same as base
- [ ] c) Larger than base
- [ ] d) Zero

### 7. vLLM LoRA hot-swap:
- [x] a) One base model + N adapters; clients pick by name; sub-100ms switch
- [ ] b) Replaces vLLM
- [ ] c) Requires retraining
- [ ] d) Adds latency proportional to N

### 8. The right method to keep architecture identical but smaller weights:
- [x] a) Quantization
- [ ] b) Distillation
- [ ] c) Pruning
- [ ] d) Fine-tuning

### 9. The right method when you need a different architecture:
- [x] a) Distillation
- [ ] b) Quantization
- [ ] c) Pruning
- [ ] d) LoRA

### 10. Stacking quantization + 2:4 sparsity:
- [x] a) Combines savings: ~6× memory reduction
- [ ] b) Cannot be combined
- [ ] c) Halves accuracy
- [ ] d) Doubles latency

---

Answers: all `a`
