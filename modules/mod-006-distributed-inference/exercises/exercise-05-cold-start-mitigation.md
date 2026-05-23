# Ex 05: Cold Start Mitigation

For Llama-70B, measure cold-start time (pod created → first request served).
Apply mitigations:
- Pre-pull image via DaemonSet
- Pre-warm 1 replica per node
- Slow-start routing (gradual ramp to new replicas)

Re-measure end-to-end latency on a scale-up event.
