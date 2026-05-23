# Module 04 — Quiz Answers

All `a`. Brief notes:
1. FlashAttn's tiling means intermediate N×N is never materialized.
2. Paged allocation eliminates external fragmentation in the KV cache.
3. Speculative decoding requires shared tokenizer; family fine-tunes work.
4. Continuous batching removes the head-of-line blocking of fixed batches.
5. Prefix cache shines for chat / agents / RAG with shared system prompts.
6. `reduce-overhead` enables CUDA graphs (low-overhead launch).
7. KV cache memory formula; commit it to memory.
8. Triton compiles Python-style kernels to PTX.
9. Optimizations compound multiplicatively in practice (not perfectly).
10. vLLM defaults are tuned by experts; start there, measure before going lower.
