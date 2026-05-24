# Module 06 — Quiz

10 questions. 70% pass.

### 1. Tensor parallelism is the right tool when:
- [x] a) The model doesn't fit on one GPU AND fast interconnect (NVLink) is available
- [ ] b) The model fits comfortably on one GPU but you want redundancy
- [ ] c) Single-GPU latency is the only concern
- [ ] d) Storage I/O is the bottleneck

### 2. Pipeline parallelism vs tensor parallelism:
- [x] a) Lower communication cost than TP, but pipeline bubbles add latency
- [ ] b) Always faster than TP
- [ ] c) Functionally identical
- [ ] d) Never used in production

### 3. 3D parallelism is:
- [x] a) Tensor-parallel within a node + pipeline-parallel across nodes + data-parallel across replicas
- [ ] b) Three independent TP groups
- [ ] c) Three different model architectures stacked
- [ ] d) Random per-layer sharding

### 4. Configuring HPA on CPU utilization for GPU inference is:
- [x] a) Poor — CPU is rarely the bottleneck on GPU-saturated serving
- [ ] b) The default best practice
- [ ] c) Required by the Kubernetes spec
- [ ] d) Mandatory for any GPU workload

### 5. The right autoscaling metric for a vLLM deployment is:
- [x] a) `vllm:num_requests_waiting` (queue depth) or generated tokens/s
- [ ] b) CPU utilization
- [ ] c) Container memory utilization
- [ ] d) Disk I/O latency

### 6. Cold start time for a 70B LLM deployment is typically:
- [x] a) 5+ minutes for image pull on a fresh node + 30-90s weight load + warmup
- [ ] b) Sub-second
- [ ] c) Several hours
- [ ] d) Not measurable in practice

### 7. Round-robin load balancing for LLM serving is:
- [x] a) Poor — request cost varies wildly with generation length
- [ ] b) The recommended best practice
- [ ] c) Required by HTTP/2
- [ ] d) Identical to least-loaded routing

### 8. Prefix-aware routing helps most when:
- [x] a) Many requests share a system prompt — route to the replica with the cached prefix
- [ ] b) Always; it's a universal optimization
- [ ] c) Only for embedding workloads
- [ ] d) Required by Kubernetes service mesh

### 9. The typical "headroom above peak" for GPU serving capacity is:
- [x] a) 25-30% above measured peak traffic
- [ ] b) 200% above peak (always run at one-third utilization)
- [ ] c) Zero — run at peak to minimize cost
- [ ] d) Sized exactly to peak with no slack

### 10. When NOT to tensor-parallel a 7B model:
- [x] a) When it fits on one GPU; replicate to N pods instead of TP-sharding across N GPUs
- [ ] b) Always TP regardless of size
- [ ] c) Only on H100 hardware
- [ ] d) Required for accuracy

---

## Answer key + rationale

1. **a** — TP requires fast cross-GPU communication on every layer; only worth it when the model exceeds one GPU.
2. **a** — PP minimizes per-layer activation transfer but introduces "bubbles" at pipeline boundaries.
3. **a** — Frontier models use this exact stack to span multi-node clusters.
4. **a** — GPU jobs saturate the GPU; CPU stays underutilized. CPU-based HPA never triggers.
5. **a** — Queue depth + token throughput are what actually correlate with serving capacity.
6. **a** — Image is multi-GB; weights load into VRAM in tens of seconds; first request takes a few more seconds.
7. **a** — A 10-token request and a 4096-token request both count as "1 request" to round-robin — disastrous for queue balance.
8. **a** — Same system prompt routed to the same replica = prefix cache hit = 2-3× lower latency.
9. **a** — 25-30% is the standard headroom for incident tolerance + auto-scaling lag.
10. **a** — TP has communication overhead; for a model that fits, replicate horizontally for both redundancy + throughput.
