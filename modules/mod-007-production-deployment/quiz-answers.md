# Module 07 — Quiz Answers

| # | Answer | Rationale |
|---|---|---|
| 1 | **a** | vLLM/TGI bake in continuous batching + paged attention. Other frameworks would need months of custom work to match. |
| 2 | **a** | Triton + TorchServe are tuned for stateless request/response with dynamic batching. vLLM's KV-cache machinery is wasted on embedding workloads. |
| 3 | **d** | Canary's value is gating on real metrics. Trivial bugfixes don't need it; schema-breaking changes need blue-green's atomic flip. |
| 4 | **a** | Shadow's defining property: real traffic mirrored, but responses ignored — zero risk to users. |
| 5 | **d** | Spot pricing comes with eviction risk. Batch training that can checkpoint + resume is the canonical workload type that tolerates it. |
| 6 | **d** | Auto-gate metrics must reflect user impact: error rate, latency p95, accuracy delta. Image size + replica count tell you nothing about correctness. |
| 7 | **a** | Tier routing exploits the cost variance across model sizes. Cheap default + selective escalation = 50-70% savings. |
| 8 | **a** | Documented vLLM benchmarks consistently show 5-10× on real chat workloads with variable request lengths. |
| 9 | **a** | AWQ frees up VRAM (more KV cache → more concurrent requests); spec decoding raises tokens/sec. Independent optimizations stack. |
| 10 | **a** | Framework choice is the most consequential decision; all subsequent layers (k8s, observability, cost) depend on it. |
