# Ex 03: Tree Reduction

Implement a parallel reduction (sum of N floats). Cover three variants:
1. Naive: each thread adds one pair; serial across iterations
2. Block-level reduction with shared memory
3. Warp shuffle (`__shfl_down_sync`) — no shared memory

Benchmark each at N=1M; explain the performance gap. Targets: variant 3
should reach ≥ 80% of `torch.sum` throughput.
