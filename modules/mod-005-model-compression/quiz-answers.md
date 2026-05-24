# Module 05 — Quiz Answers

| # | Answer | Rationale |
|---|---|---|
| 1 | **d** | AWQ (Activation-aware Weight Quantization) is engineered around LLM weight distributions — weight-only int4 with per-channel scaling. |
| 2 | **a** | Sparse Tensor Cores (Ampere+) natively skip zeros in 2:4 patterns; on other hardware, "sparse" multiplies still touch the zero values. |
| 3 | **a** | QLoRA = quantize the frozen base model to 4-bit + train a LoRA adapter on top. Puts 13B fine-tuning on a single 24GB GPU. |
| 4 | **d** | Distillation transfers the teacher's behavior to the student via soft labels. A weak teacher produces a weak student. |
| 5 | **a** | AWQ's published results consistently show < 0.5pp drop on MMLU/HellaSwag/etc. for 7B+ models; smaller models suffer more. |
| 6 | **d** | At rank 16, adapter matrices `A: [d × 16]` + `B: [16 × d]` for d=4096 → ~50MB total even on a 7B base. |
| 7 | **d** | vLLM `--enable-lora` loads N adapters into the same base model; routing is by `model` field in the request. |
| 8 | **a** | Quantization changes the bit-width of weights but leaves architecture + structure unchanged. |
| 9 | **d** | Distillation is the only method that lets you change the architecture (different layer count, hidden size, etc.) while preserving behavior. |
| 10 | **d** | Quantization (~4× memory) + 2:4 sparsity (~2× compute & ~1.5× memory) stack to roughly 6× memory reduction overall. |
