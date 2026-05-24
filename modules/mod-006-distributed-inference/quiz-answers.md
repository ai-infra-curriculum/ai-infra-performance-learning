# Module 06 — Quiz Answers

| # | Answer | Rationale |
|---|---|---|
| 1 | **a** | TP requires high-bandwidth interconnect (NVLink) on every layer. Only worth the comm overhead when the model can't fit on one GPU. |
| 2 | **a** | PP saves activation-transfer bandwidth but creates "bubbles" at pipeline boundaries where some stages idle. |
| 3 | **a** | 3D = TP (within node) + PP (across nodes) + DP (across replicas). Required for 100B+ models on multi-node clusters. |
| 4 | **a** | GPU jobs typically run at <10% CPU; CPU-based HPA never triggers. Use queue-based metrics instead. |
| 5 | **a** | `vllm:num_requests_waiting` is the canonical metric; tokens/s also works. Both reflect actual saturation. |
| 6 | **a** | Image pull (multi-GB) + weight load (HBM init) + warmup (first kernel compile) compose into a real 5-10 min cold start. |
| 7 | **a** | RR distributes request *count* evenly. LLM requests vary 10-100× in cost. Least-loaded or token-aware routing is needed. |
| 8 | **a** | Consistent hashing on the system prompt → same replica → prefix cache hit → 2-3× faster TTFT. |
| 9 | **a** | 25-30% above peak gives room for spikes + scale-up lag without burning cost. |
| 10 | **a** | TP costs comm overhead. A 7B model that fits on one GPU should be replicated horizontally; you get redundancy + throughput without the comm tax. |
