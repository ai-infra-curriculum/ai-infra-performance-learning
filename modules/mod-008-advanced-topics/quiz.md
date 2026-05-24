# Module 08 — Quiz

10 questions. 70% pass.

### 1. CUDA Graphs primarily eliminate:
- [ ] a) GPU memory pressure
- [ ] b) GPU thermal throttling
- [x] c) Per-kernel-launch overhead by capturing an op sequence + replaying it
- [ ] d) Inter-kernel synchronization

### 2. CUDA streams enable:
- [x] a) Overlapping independent operations (e.g., compute on one stream while copying on another)
- [ ] b) Reduced GPU memory consumption
- [ ] c) A requirement for any GPU code (they aren't)
- [ ] d) Higher numeric precision

### 3. NVLink vs PCIe (for GPU-to-GPU communication):
- [ ] a) Roughly the same speed
- [ ] b) PCIe is the faster option
- [ ] c) NVLink is a software-only abstraction
- [x] d) NVLink is ~10× faster than PCIe; effectively required for in-node tensor parallelism

### 4. InfiniBand's role in distributed ML training:
- [ ] a) A storage-only protocol
- [x] b) Inter-node fabric with RDMA + low latency for multi-node collectives
- [ ] c) A replacement for NVLink within a node
- [ ] d) Required even for single-GPU training

### 5. GPUDirect RDMA is:
- [ ] a) A replacement for NCCL
- [ ] b) Requires local SSDs
- [x] c) Remote GPU memory access without involving the CPU on either side
- [ ] d) Identical to PCIe peer access

### 6. MIG (Multi-Instance GPU) partitioning:
- [x] a) Splits one A100/H100 into hardware-isolated independent GPU instances
- [ ] b) A multi-GPU NCCL configuration
- [ ] c) A software-only sharing scheme
- [ ] d) Always degrades performance vs full GPU

### 7. FP8 training on H100 with Transformer Engine typically yields:
- [ ] a) Same speed as bf16
- [ ] b) Always significant accuracy loss
- [x] c) 30-50% training speedup with minimal accuracy loss
- [ ] d) Is required for any inference workload

### 8. The dynamic range issue with FP8:
- [ ] a) Same range as fp16
- [ ] b) Not a real concern
- [ ] c) Wider range than bf16
- [x] d) Narrow range — per-tensor scaling factors are required (Transformer Engine handles this automatically)

### 9. CUDA Graph capture is bounded by:
- [ ] a) GPU hardware model
- [x] b) Input shape — new shapes require recapture
- [ ] c) NVIDIA driver version
- [ ] d) Container runtime version

### 10. The right first diagnostic for "slow distributed training":
- [ ] a) Replace the model architecture
- [ ] b) Add more GPUs
- [ ] c) Switch frameworks
- [x] d) Run nccl-tests (all-reduce benchmark) — rules out interconnect-fabric issues

---

## Answer key + rationale

1. **c** — Graphs amortize per-launch overhead by replaying captured op sequences as one launch.
2. **a** — Independent ops on separate streams can execute concurrently, hiding transfer behind compute.
3. **d** — NVLink Gen4 ≈ 900 GB/s per GPU vs PCIe Gen4 ≈ 64 GB/s. TP across PCIe is rarely worth it.
4. **b** — IB is the standard inter-node fabric for HPC + ML training clusters.
5. **c** — GPUDirect RDMA copies GPU memory directly between nodes without staging via CPU.
6. **a** — MIG provides true hardware isolation; different instances can run different workloads with no contention.
7. **c** — FP8 with TE consistently shows 30-50% speedup with sub-1pp accuracy delta on transformer benchmarks.
8. **d** — FP8 has only 7-bit precision; scaling factors are essential. TE manages this automatically.
9. **b** — Captured graphs are shape-specific; dynamic-shape workloads need either recapture or fallback to eager.
10. **d** — nccl-tests immediately answers "is the fabric slow?" before debugging model code.
