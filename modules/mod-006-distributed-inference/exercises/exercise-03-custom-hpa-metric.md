# Ex 03: HPA on Custom Metric

Configure HPA to scale vLLM replicas on `vllm:num_requests_waiting` (queue depth)
instead of CPU. Requires Prometheus + Prometheus Adapter.

Demonstrate: load spike → queue grows → replicas scale up → queue drains.
