# Ex 02: Tiled Matrix Multiply

Implement a tile-based matmul using shared memory. Target: ≥ 50% of cuBLAS
throughput at N=4096. Compare TS=16 vs TS=32 vs TS=64.

Report: TFLOPS, achieved occupancy (from `ncu`), bank conflict count, %
of cuBLAS perf.
