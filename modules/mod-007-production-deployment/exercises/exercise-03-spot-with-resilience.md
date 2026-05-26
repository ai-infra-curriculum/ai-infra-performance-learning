# Ex 03: Spot Instances with Resilience

Move batch model training to spot instances. Add:
- PodDisruptionBudget
- Karpenter spot pool with auto-recovery
- Graceful checkpointing on spot interruption

Measure $/training-run savings vs on-demand.

Companion: engineer-solutions/mod-104 ex-15.
