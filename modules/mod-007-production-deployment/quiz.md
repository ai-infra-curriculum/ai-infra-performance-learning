# Module 07 — Quiz

10 questions. 70% pass.

### 1. The best serving framework for an autoregressive LLM:
- [x] a) vLLM or TGI — continuous batching + paged attention
- [ ] b) TorchServe — model-based serving for PyTorch
- [ ] c) FastAPI alone (with a generation loop)
- [ ] d) TensorFlow Serving

### 2. The best serving framework for an embedding model (high-QPS request/response):
- [x] a) Triton Inference Server or TorchServe
- [ ] b) vLLM
- [ ] c) A bash curl wrapper
- [ ] d) Argo Workflows

### 3. Canary deployment fits best for:
- [ ] a) Trivial bugfixes with no metric impact
- [ ] b) Schema-breaking changes (blue-green is better)
- [ ] c) Cron jobs
- [x] d) Risky model changes with measurable success metrics that gate the rollout

### 4. Shadow deployment serves:
- [x] a) 0% real traffic; mirror traffic to the candidate; ignore its responses
- [ ] b) 100% real traffic to the new version
- [ ] c) 50/50 split between old + new
- [ ] d) Internal users only

### 5. Spot instances are appropriate for:
- [ ] a) Critical real-time inference
- [ ] b) etcd / control plane
- [ ] c) Ingress load balancers
- [x] d) Batch training that can checkpoint + resume

### 6. The auto-revert metric on a canary should be:
- [ ] a) Image size delta
- [ ] b) Replica count
- [ ] c) Time of day
- [x] d) Success rate or 5xx rate or a custom signal (model accuracy delta)

### 7. Per-tier routing saves cost by:
- [x] a) Sending simple requests to cheap models; escalating to expensive only when needed
- [ ] b) Always using the cheapest model
- [ ] c) Always using the most expensive model
- [ ] d) Routing randomly across tiers

### 8. Continuous batching's typical speedup over naive padded batching:
- [x] a) 5-10× higher throughput
- [ ] b) Marginal (<10%)
- [ ] c) Strictly slower
- [ ] d) Required for accuracy (changes outputs)

### 9. Stacking AWQ int4 + speculative decoding:
- [x] a) Combines: ~50-75% memory savings + 2-3× throughput
- [ ] b) Cannot be combined (mutually exclusive)
- [ ] c) Halves accuracy
- [ ] d) Locks the model into fp32

### 10. The right place to start production deployment work is:
- [x] a) Pick the right serving framework first; then containerize, add observability, attribute cost
- [ ] b) Start with HPA configuration
- [ ] c) Custom FastAPI always
- [ ] d) vLLM regardless of workload

---

## Answer key + rationale

1. **a** — vLLM/TGI implement the LLM-specific tricks (continuous batching, paged attention) that other frameworks lack.
2. **a** — Triton + TorchServe are tuned for high-QPS request/response patterns; vLLM's tricks are wasted on stateless embeddings.
3. **d** — Canary's value is the gated rollout. Trivial changes don't need it; breaking changes need blue-green.
4. **a** — Shadow = 0% real traffic. The point is validation without risk.
5. **d** — Spot = preemptible = need to tolerate eviction. Checkpoint-able batch is the canonical fit.
6. **d** — Metrics that reflect actual user impact: success rate, error rate, latency, accuracy delta.
7. **a** — Cost varies 10-100× across tiers; the trick is sending each request to the cheapest model that handles it.
8. **a** — Documented vLLM benchmarks consistently show 5-10× over naive batching on real chat workloads.
9. **a** — AWQ → smaller weights → more KV cache headroom; spec decoding → tokens per second up.
10. **a** — Framework choice constrains everything downstream. Pick wrong + you're stuck.
