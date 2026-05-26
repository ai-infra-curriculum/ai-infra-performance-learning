# Module 08 — Quiz Answers

| # | Answer | Rationale |
|---|---|---|
| 1 | **c** | CUDA Graphs capture an op sequence and replay it as a single launch, amortizing per-kernel overhead. Big win for short-kernel workloads. |
| 2 | **a** | Independent ops on separate streams execute concurrently — overlap compute with H2D/D2H transfers to hide them. |
| 3 | **d** | NVLink Gen4 ≈ 900 GB/s per GPU; PCIe Gen4 ≈ 64 GB/s. TP across PCIe is rarely worth it. |
| 4 | **b** | InfiniBand is the standard inter-node fabric for HPC and ML training clusters — RDMA + low latency. |
| 5 | **c** | GPUDirect RDMA moves data directly between GPU memories across nodes without staging via CPU memory. |
| 6 | **a** | MIG provides true hardware isolation — different MIG instances can run different workloads with no contention for compute, memory bandwidth, or cache. |
| 7 | **c** | FP8 with Transformer Engine routinely shows 30-50% speedup with sub-1pp accuracy delta on transformer benchmarks. |
| 8 | **d** | FP8 has only 7-bit precision; per-tensor scaling factors are essential. TE manages this automatically. |
| 9 | **b** | Captured graphs are shape-specific; dynamic-shape workloads need either recapture per shape or fallback to eager mode. |
| 10 | **d** | nccl-tests immediately answers "is the fabric slow?" — rule out hardware/interconnect before debugging model code. |
