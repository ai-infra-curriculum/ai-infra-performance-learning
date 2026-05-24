# Module 03 — Quiz Answers

| # | Answer | Rationale |
|---|---|---|
| 1 | **d** | nsys = system-level timeline (workflow); ncu = single-kernel deep dive (tuning). Different layers; both needed. |
| 2 | **a** | NVTX is the marker API both nsys + ncu consume to label timeline ranges with human-meaningful names. |
| 3 | **d** | Roofline plots arithmetic intensity (FLOPS/byte) vs achieved performance. Kernels sit under the bandwidth slope (memory-bound) or under the FLOPS ceiling (compute-bound). |
| 4 | **a** | Both axes low = GPU under-driven. Usually too few active warps or warp divergence is killing effective parallelism. Adding memory or batch size won't help. |
| 5 | **d** | `torch.cuda.memory._dump_snapshot()` produces a binary that uploads to memory_viz for interactive memory-trace exploration. |
| 6 | **d** | Activations dropped during forward, recomputed during backward. Halves activation memory at ~30% wall-clock cost. |
| 7 | **a** | 8-bit quantization of optimizer states (`m`, `v` for Adam) cuts the per-param optimizer overhead from 8 bytes (fp32 × 2) down to ~2. |
| 8 | **d** | FlashAttention tiles attention computation so the N×N attention matrix never materializes in HBM. Memory drops to O(N) and the tiled compute is often faster too. |
| 9 | **a** | Each kernel launch costs ~5μs of overhead. For sub-100μs kernels, batching or CUDA Graphs is the only way to escape the overhead floor. |
| 10 | **d** | Single-call timing measures cold-start (JIT compile, allocator warmup). Warmup + `torch.cuda.synchronize()` between calls + median over many runs is the correct protocol. |
