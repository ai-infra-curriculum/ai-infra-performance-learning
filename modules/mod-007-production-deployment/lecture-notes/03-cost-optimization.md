# Lecture 03: Cost Optimization

## Cost levers

| Lever | Savings | Risk |
|---|---|---|
| Spot for batch | 60-70% | spot interruption |
| Reserved (1-3y) | 40-50% | locked into capacity |
| Right-sizing | 20-40% | requires telemetry |
| Quantization (mod-005) | 50-75% | small accuracy hit |
| Spec decoding (mod-004) | 30-50% throughput | matched draft model needed |
| Continuous batching (mod-004) | 5-10× throughput | none |
| Per-model routing (cheap → expensive) | 50-70% | requires good classifier |

## Compounding

Combining: continuous batching + AWQ int4 + spec decoding + spot for batch
+ tier-routing → typical 5-10× lower $/request than naive baseline.

## Companion

- engineer-solutions/mod-110 ex-08 (cost routing)
- engineer-solutions/mod-104 ex-15 (cluster cost optimization)
- engineer-solutions/mod-106 ex-12 (per-model cost attribution)
