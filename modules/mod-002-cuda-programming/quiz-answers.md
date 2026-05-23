# Quiz Answers — Module 02

All 15 answers are option **a**. Brief explanations:

1. **Multiple of 32** — wasted threads in partial warps slow everything down.
2. **Coalesced** = consecutive threads → consecutive memory → one transaction.
3. **100-1000×** — shared memory is on-chip; global memory is off-chip HBM.
4. **Bank conflicts** = same bank, same warp, simultaneous access → serializes.
5. **Divergence** within a warp serializes branches → throughput halves per branch.
6. All four limit the maximum active threads per SM.
7. **Data reuse** is the entire point of tiling.
8. CUDAExtension wraps `nvcc` + `gcc` + `setuptools`.
9. Without `synchronize`, the host returns immediately after the kernel launch; you measure ~5μs not kernel execution.
10. Wasted compute means real throughput drops.
11. First param = grid; second = block. Easy to flip if you're new.
12. `__shfl_down_sync` is the warp-level primitive for cooperation without shared memory.
13. Heuristic only; always verify empirically.
14. Padding shifts the access pattern so all banks are touched evenly.
15. First-call overhead is non-trivial; warm up before benchmarking.
