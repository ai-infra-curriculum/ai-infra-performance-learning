# Module 08 — Quiz

10 questions. 70% pass.

### 1. CUDA Graphs eliminate:
- [x] a) Per-kernel-launch overhead by capturing + replaying ops
- [ ] b) Memory usage
- [ ] c) GPU heat
- [ ] d) Synchronization

### 2. CUDA streams enable:
- [x] a) Overlapping independent operations (compute + transfer)
- [ ] b) Reduced memory
- [ ] c) Required for any GPU code
- [ ] d) Higher precision

### 3. NVLink vs PCIe:
- [x] a) NVLink ~10× faster; required for in-node TP
- [ ] b) Same speed
- [ ] c) PCIe is faster
- [ ] d) NVLink is software only

### 4. InfiniBand role:
- [x] a) Inter-node fabric for distributed training (RDMA, low latency)
- [ ] b) Storage protocol only
- [ ] c) Replaces NVLink
- [ ] d) Required for single-GPU training

### 5. GPUDirect RDMA:
- [x] a) Remote GPU memory access without CPU involvement
- [ ] b) Replaces NCCL
- [ ] c) Requires SSDs
- [ ] d) Same as PCIe

### 6. MIG partitioning:
- [x] a) Splits one A100/H100 into hardware-isolated instances
- [ ] b) Multi-GPU NCCL config
- [ ] c) Software-only sharing
- [ ] d) Always reduces performance

### 7. FP8 on H100 with TE:
- [x] a) 30-50% training speedup with minimal accuracy loss
- [ ] b) Same speed as bf16
- [ ] c) Always lossy
- [ ] d) Required for inference

### 8. FP8 dynamic range issue:
- [x] a) Narrow range; per-tensor scaling factors required (TE handles this)
- [ ] b) Same as fp16
- [ ] c) No issue
- [ ] d) Wider than bf16

### 9. CUDA Graph capture is bounded by:
- [x] a) Input shape (recapture needed for new shapes)
- [ ] b) GPU model
- [ ] c) Driver version
- [ ] d) Container runtime

### 10. The right diagnostic for "slow distributed training":
- [x] a) Run nccl-tests to measure all-reduce; rules out fabric issues
- [ ] b) Replace the model
- [ ] c) Add more GPUs
- [ ] d) Switch to PyTorch Lightning

---

Answers: all `a`
