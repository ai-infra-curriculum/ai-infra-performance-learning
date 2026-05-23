# Module 06 — Quiz Answers

All `a`. Brief notes:
1. TP shines on multi-GPU with NVLink for sub-replica models.
2. PP saves comm cost but introduces bubbles at start/end.
3. 3D parallelism is how 100B+ models are served.
4. CPU-based HPA misfits GPU workloads.
5. Custom Prometheus metrics via the adapter.
6. Cold-start cost is real and non-trivial; mitigate.
7. Round-robin distributes count, not load.
8. Prefix-aware routing maximizes cache hits.
9. ~25% headroom balances cost vs availability.
10. Don't over-parallelize; replicate for redundancy.
