# Module 06 — Quiz

10 questions. 70% pass.

### 1. Tensor parallelism is appropriate when:
- [x] a) Model is larger than one GPU; fast interconnect (NVLink) available
- [ ] b) Model fits on one GPU comfortably
- [ ] c) Single-GPU latency is the only concern
- [ ] d) Storage is the bottleneck

### 2. Pipeline parallelism's cost vs tensor parallelism:
- [x] a) Lower comm cost but pipeline bubbles add latency
- [ ] b) Always faster than TP
- [ ] c) Same as TP
- [ ] d) Never used in practice

### 3. 3D parallelism combines:
- [x] a) TP within node + PP across nodes + DP across replicas
- [ ] b) Three TP groups
- [ ] c) Three different model architectures
- [ ] d) Random sharding

### 4. HPA on CPU for GPU inference:
- [x] a) Poor choice; CPU is rarely the bottleneck on GPU serving
- [ ] b) Default best practice
- [ ] c) Required
- [ ] d) Mandatory for Kubernetes

### 5. The right autoscaling metric for vLLM:
- [x] a) `num_requests_waiting` (queue depth) or tokens/s
- [ ] b) CPU utilization
- [ ] c) Memory utilization
- [ ] d) Disk I/O

### 6. Cold start for a 70B model:
- [x] a) 5+ min image pull + 30-90s weight load + warmup
- [ ] b) Sub-second
- [ ] c) Several hours
- [ ] d) Not measurable

### 7. Round-robin load balancing for LLM:
- [x] a) Poor; request cost varies wildly with generation length
- [ ] b) Best practice
- [ ] c) Required
- [ ] d) Same as least-loaded

### 8. Prefix-aware routing benefits:
- [x] a) When many requests share a system prompt; routes to cached replica
- [ ] b) Always
- [ ] c) Only for embedding
- [ ] d) Required by Kubernetes

### 9. Headroom above peak typical:
- [x] a) 25-30% above peak
- [ ] b) 200%
- [ ] c) Zero
- [ ] d) Set to exact peak

### 10. When NOT to tensor-parallel a 7B model:
- [x] a) When it fits on one GPU; replicate instead
- [ ] b) Always tensor-parallel
- [ ] c) Only on H100
- [ ] d) Required for accuracy

---

Answers: all `a`
