# Ex 05: End-to-End Optimization Cycle

Pick a training script that runs slow. Apply the cycle:
1. nsys: find phase-level bottleneck
2. ncu: find kernel-level bottleneck within the phase
3. Apply one optimization (gradient checkpointing, mixed precision, larger batch)
4. Re-profile; verify improvement
5. Repeat until p95 step time is 2× better

Deliverable: before/after profiles + `OPTIMIZATION_LOG.md`.
