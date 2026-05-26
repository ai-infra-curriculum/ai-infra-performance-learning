# Lecture 03: Memory Profiling

## PyTorch memory snapshot

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
# train one iteration
torch.cuda.memory._dump_snapshot("snapshot.bin")
torch.cuda.memory._record_memory_history(enabled=None)
```

Upload `snapshot.bin` to <https://pytorch.org/memory_viz>. You'll see every
allocation + deallocation + the call stack that made it.

## When OOM strikes

The most common patterns:
- **Activation memory** grows linearly with batch size + sequence length
- **Gradient checkpointing** halves activations at the cost of ~30% recompute
- **Optimizer state** for Adam = 2× model params; AdamW8bit = 1/4 of that
- **KV cache** in LLM inference: linear in seq len × batch × layers

`torch.cuda.memory_summary()` is a coarse but quick view.

## Reducing memory pressure

| Technique | Memory savings | Speed cost |
|---|---|---|
| Gradient checkpointing | ~50% activation | +30% time |
| Mixed precision (bf16) | ~50% activation + weights | none (often faster) |
| AdamW8bit | ~75% optimizer state | small |
| Tensor parallel | linear in TP degree | comm overhead |
| Pipeline parallel | linear in PP degree | bubbles |
| FlashAttention | quadratic → linear in seq | none (much faster) |

## Companion

[engineer-solutions/mod-107 ex-07 (gpu-memory-profiling)](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-107-gpu-computing/exercise-07-gpu-memory-profiling) ships baseline + optimized + snapshot scripts.
