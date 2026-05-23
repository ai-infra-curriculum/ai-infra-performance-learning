# Module 07 — Quiz

10 questions. 70% pass.

### 1. The best serving framework for an autoregressive LLM:
- [x] a) vLLM or TGI (continuous batching, paged attention)
- [ ] b) TorchServe
- [ ] c) FastAPI alone
- [ ] d) TensorFlow Serving

### 2. The best framework for an embedding model:
- [x] a) Triton or TorchServe
- [ ] b) vLLM
- [ ] c) Curl wrapper
- [ ] d) Argo Workflows

### 3. Canary deployment fits:
- [x] a) Risky model change with measurable success metrics
- [ ] b) Trivial bugfix
- [ ] c) Schema-breaking change
- [ ] d) Cron job

### 4. Shadow deployment serves:
- [x] a) 0% real traffic; mirror traffic to candidate
- [ ] b) 100% traffic
- [ ] c) 50/50 split
- [ ] d) Internal users only

### 5. Spot instances are appropriate for:
- [x] a) Batch training (with checkpointing)
- [ ] b) Critical inference
- [ ] c) Etcd
- [ ] d) Load balancers

### 6. The auto-revert metric on canary:
- [x] a) 5xx rate or success rate or custom (model accuracy)
- [ ] b) Image size
- [ ] c) Number of replicas
- [ ] d) Time of day

### 7. Per-tier routing saves cost by:
- [x] a) Sending simple requests to cheap models; escalating only when needed
- [ ] b) Always using the cheapest model
- [ ] c) Always using the most expensive
- [ ] d) Random routing

### 8. Continuous batching speedup:
- [x] a) 5-10× over naive padded batching
- [ ] b) Marginal
- [ ] c) Slower
- [ ] d) Required for accuracy

### 9. AWQ int4 + spec decoding stack:
- [x] a) Combines: 50-75% memory savings + 2-3× throughput
- [ ] b) Cannot be combined
- [ ] c) Halves accuracy
- [ ] d) Locks in fp32

### 10. The right place to start production deployment:
- [x] a) Pick the right framework first; then containerize + observability + cost
- [ ] b) Start with HPA
- [ ] c) Always custom FastAPI
- [ ] d) Always vLLM

---

Answers: all `a`
