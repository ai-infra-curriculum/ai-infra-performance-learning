# Ex 01: Pick a Framework

Given three different model types, pick the right serving framework for each
and defend the choice with a benchmark, not a vibe. The point is to map model
shape and traffic pattern onto framework strengths (continuous batching,
dynamic batching, KV-cache paging, ONNX/TensorRT backends).

## Workloads

- An LLM (e.g., Mistral-7B or Llama-3-8B), chat-style, variable-length output.
- A vision classifier (e.g., ResNet-50 or ViT), fixed-shape, high QPS.
- An embedding model (e.g., `bge-base` / `e5`), short inputs, batch-friendly.

## Tasks

1. For each workload, shortlist two candidate frameworks from: vLLM, TGI,
   TensorRT-LLM, Triton Inference Server, TorchServe, or Ray Serve.
2. Stand up at least one candidate per workload and run a load test
   (`locust`, `k6`, or vLLM's `benchmark_serving.py`) at a realistic
   concurrency. Capture p50/p99 latency, throughput (tokens/s or images/s),
   and peak GPU memory.
3. For the vision and embedding models, exercise dynamic/request batching and
   show the throughput delta with batching on vs off.
4. Decide a winner per workload and state the decisive metric.

## Acceptance criteria

- Each decision is backed by a measured number, not a doc quote.
- The LLM choice explicitly accounts for continuous batching and KV-cache
  paging behavior under concurrent requests.
- The vision/embedding choices show a measured batching throughput gain.
- At least one rejected candidate per workload, with the reason it lost.

Justify all three choices in `FRAMEWORK_DECISIONS.md`: workload profile,
candidates, benchmark table, winner, and the trade-off you accepted.

Companion: engineer-solutions/mod-108 (serving framework selection).
