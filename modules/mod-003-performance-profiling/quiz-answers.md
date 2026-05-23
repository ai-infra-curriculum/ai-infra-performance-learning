# Quiz Answers — Module 03

All 10 = `a`. Explanations:

1. **nsys** = system-level timeline; **ncu** = single-kernel deep dive.
2. **NVTX** ranges annotate your high-level phases so the timeline shows them.
3. **Roofline** is the canonical classifier of kernel limits.
4. Low on both axes = under-utilization (not enough warps or divergence).
5. **memory_viz** is the canonical visualizer for PyTorch snapshots.
6. **Gradient checkpointing** halves activations at ~30% wall-clock cost.
7. **AdamW8bit** quantizes optimizer state from fp32 → int8.
8. **FlashAttention** turns memory-quadratic attention into memory-linear via tiling.
9. Per-launch overhead is ~5μs; for tiny kernels, the overhead is the cost.
10. Warmup + sync + median = correct way to benchmark.
