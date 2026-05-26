# Lecture 02: Multi-Replica Autoscaling

## Three axes

| Axis | Mechanism |
|---|---|
| HPA on CPU | not useful for GPU inference (CPU is rarely the bottleneck) |
| HPA on custom metric | scale on `gpu_queue_depth` or `tokens_per_second` |
| KEDA on Kafka lag | scale on backlogged requests in queue |

## Per-replica concurrency

A single vLLM replica handles N concurrent requests (continuous batching).
"Replicas to add" is not "requests / 1" — it's "requests / capacity_per_replica".

Measure capacity_per_replica empirically; vary based on model + GPU + sequence length.

## Cold start

Adding a replica for a 70B model:
- Pull 140 GB image: 5+ minutes on first node
- Load weights into GPU: 30-90 seconds
- Warmup: 10-30 seconds (first request slow)

Strategies:
- **Pre-pull image** on every node (DaemonSet)
- **Pre-warm pods** kept idle below traffic floor
- **Predictive scaling**: anticipate traffic from time-of-day patterns

## Cost of over-scaling

GPUs are expensive. Don't over-scale.
- Right-size capacity headroom to 25-30% above peak
- Use spot/preemptible for batch (mod-104 ex-15)
- Scale down aggressively (1h cool-down) during predictable troughs
