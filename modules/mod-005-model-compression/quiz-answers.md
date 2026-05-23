# Module 05 — Quiz Answers

All `a`. Notes:
1. AWQ is weight-only int4 + activation-aware; designed for LLMs.
2. Sparse Tensor Cores natively skip zeros in 2:4 patterns.
3. QLoRA = quantize base + LoRA adapter.
4. Distillation inherits teacher quality; bad teacher = bad student.
5. AWQ on large models is nearly lossless; small models suffer more.
6. ~50 MB for r=16; varies with rank.
7. vLLM `--enable-lora` does multi-adapter serving.
8. Quantization keeps the architecture intact.
9. Distillation lets you change architecture entirely.
10. Compression methods are largely orthogonal; combine for compounded savings.
