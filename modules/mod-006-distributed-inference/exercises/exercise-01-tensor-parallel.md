# Ex 01: Tensor-Parallel vLLM Deployment

Deploy Llama-13B with `--tensor-parallel-size 2`. Compare:
- TP=1 (single GPU OOM or quantized)
- TP=2 (split across 2 GPUs)

Measure: TTFT, throughput, latency p95.
