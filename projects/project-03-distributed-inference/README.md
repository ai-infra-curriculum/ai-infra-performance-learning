# Project 3: High-Performance LLM Inference System

> **Tier**: 4 (Capstone-grade)
> **Track**: AI/ML Performance Engineering
> **Estimated effort**: 80 hours
> **Complexity**: Advanced+
> **Primary modules**: mod-007 (Production Deployment), mod-004 (Transformer Optimization), mod-009 (Distributed Inference)
> **Secondary modules**: mod-003 (Profiling), mod-005 (Compression), mod-008 (Advanced Topics)

## 1. Overview

Build a **production-grade LLM serving system** that sustains **>= 1000
requests/sec** with **P99 < 100 ms** time-to-first-token (TTFT) on a
single H100 node serving a 7B-parameter decoder-only model (Llama-3 8B
or Mistral-7B). The system must implement, from scratch and integrated
end-to-end:

- **Continuous batching** with iteration-level scheduling.
- **PagedAttention** with a block-table-driven KV cache manager.
- **Dynamic scheduling** that mixes prefill and decode in the same
  forward pass (chunked prefill).
- **Token streaming** via Server-Sent Events (SSE) and a websocket
  control plane.
- **Speculative decoding** (draft + target verify) as a stretch path.
- **Prometheus metrics** with latency histograms, queue depth gauges,
  KV cache occupancy, and per-tenant counters.
- **SLO enforcement**: admission control, request shedding, and a
  cost-aware admission policy.
- **Reproducible load test** via a Locust + custom Python benchmarker
  that produces a poisson-arrival, mixed-prompt workload.

The deliverable is not a thin wrapper around vLLM — it is a from-scratch
serving engine that the candidate can defend line-by-line, with vLLM
0.6+ and TensorRT-LLM 0.10+ used **only** as performance references in
the benchmark.

This is the project that proves the candidate can ship a serving
system an SRE team will actually run in production.

## 2. Performance targets (the gates you must hit)

Measured on **1x NVIDIA H100 SXM5 80GB** (A100 80GB acceptable with
reported derate). Model: Llama-3 8B BF16 (or Mistral-7B BF16).
Workload: ShareGPT-style mixed prompts, mean input 512 tokens,
mean output 256 tokens, poisson arrivals.

| Metric | Target | Gate |
|--------|--------|------|
| Sustained throughput (1000 req/sec workload) | **>= 1000 req/sec successfully completed** | Hard |
| TTFT P50 (time to first token) | **<= 40 ms** | Hard |
| TTFT P99 | **<= 100 ms** | Hard |
| TPOT P50 (time per output token, steady) | **<= 25 ms** | Hard |
| TPOT P99 | **<= 60 ms** | Hard |
| End-to-end P99 (200-token completion) | **<= 5.0 s** | Hard |
| KV cache hit rate (prefix caching enabled) | **>= 40%** on workload with shared system prompts | Hard |
| GPU utilization under load (SM busy) | **>= 80%** | Hard |
| KV cache fragmentation (PagedAttention) | **<= 5%** | Hard |
| Goodput (requests meeting SLO) | **>= 95%** at offered load 1000 req/sec | Hard |
| Throughput vs vLLM 0.6 reference | **within 10% (slower OK)** | Hard |
| Throughput vs TensorRT-LLM 0.10 reference | **within 25% (slower OK)** | Soft |
| Recovery from OOM / cache exhaustion | graceful 429 + Retry-After | Hard |

Workload mix is defined in `bench/workloads/sharegpt_mixed.yaml` and is
the canonical input for all benchmark numbers.

## 3. Learning outcomes

By the end of this project a learner will be able to:

1. Design a request lifecycle for an LLM serving system from
   admission to completion, including queue priorities and backpressure.
2. Implement **continuous batching** (Yu et al., Orca 2022) with
   per-iteration scheduling that mixes prefill and decode requests.
3. Implement **PagedAttention** (Kwon et al., vLLM 2023) with a block
   table indirection, virtual KV-cache addressing, and copy-on-write
   for prefix sharing.
4. Implement **chunked prefill** to keep latency bounded under load
   while sustaining decode throughput.
5. Implement **speculative decoding** (Leviathan et al. 2023) with a
   draft model + target verification step.
6. Operate the system with a Prometheus + Grafana stack: define SLIs,
   set SLOs, write alerts.
7. Run a real load test (poisson arrivals, mixed lengths) and
   interpret the result distribution.
8. Defend the system architecture against a reference implementation
   (vLLM, TensorRT-LLM) with measured numbers and a clear
   "where we are better, where we are worse, why" story.

## 4. Prerequisites

### 4.1 Hardware (recommended)

- **Strongly preferred**: 1x NVIDIA H100 SXM5 80GB on a host with
  >= 256 GB system RAM, >= 2 TB NVMe.
- **Acceptable**: 1x A100 80GB SXM. Throughput target relaxes to 600
  req/sec; TTFT targets stand.
- **For development**: 1x A10G 24GB or RTX 4090. KV cache shrinks
  proportionally; targets do not transfer; the candidate must report
  on the production-class GPU before submission.
- 25 Gbps NIC for the load-test client (loopback acceptable but the
  numbers are noted as such).

### 4.2 Software (pinned)

- CUDA Toolkit **12.4**+
- cuDNN **9.x**
- PyTorch **2.4**+ built against CUDA 12.4
- **vLLM 0.6+** (reference baseline only)
- **TensorRT-LLM 0.10+** (reference baseline only)
- **FlashAttention 2.6+** or **FlashAttention-3** on H100
- **NVIDIA Triton Inference Server 24.07+** (optional comparison
  baseline)
- Python **3.10** or **3.11**
- **FastAPI 0.110+** for the HTTP control plane
- **uvicorn 0.30+** or **hypercorn 0.16+** as ASGI server
- **Prometheus 2.50+** and **Grafana 10+**
- **Locust 2.25+** for load testing
- **Redis 7.x** for distributed state (request tracking, rate limit)
- Nsight Systems **2024.2**+ for end-to-end timeline analysis

### 4.3 Knowledge

- Completion of mod-004 (Transformer Optimization), mod-007
  (Production Deployment).
- Comfort with async Python (`asyncio`, `aiohttp`).
- Familiarity with Kubernetes manifests (the engine ships with one;
  the candidate isn't required to deploy it).
- Understanding of the Orca and vLLM papers.

## 5. Deliverables

Per `deliverables/README.md`, the final submission must include:

- `src/engine/` — core serving engine (scheduler, batcher, executor).
- `src/cache/` — PagedAttention block manager + prefix cache.
- `src/api/` — FastAPI app with `/v1/completions`, `/v1/chat/completions`
  (OpenAI-compatible), and `/health`, `/metrics`.
- `src/spec/` — speculative decoding (draft model + verify).
- `src/kernels/` — CUDA / Triton kernels for paged attention
  (or vetted wrappers around FlashAttention with paged KV support).
- `bench/` — load test harness, workload definitions, baseline
  comparison runner.
- `infra/` — Helm chart, Prometheus rules, Grafana dashboard JSON.
- `tests/` — pytest suite (unit + integration + load smoke).
- `Dockerfile` — multi-stage build, < 4 GB final image.
- `Makefile` — `make all`, `make bench`, `make verify`, `make load`.
- `reports/` — load test results, latency CDFs, SLO compliance reports,
  baseline comparison.

## 6. Week-by-week breakdown

This is an **eight-week** project at ~10 hours/week (or six weeks at
~13 h/week). Phases are sequential — do not start a phase before the
previous one's gate passes.

### Week 1 — Foundations: model loading + naive serving (8-10h)

**Gate**: An OpenAI-compatible `/v1/completions` endpoint streams
tokens from a Llama-3 8B model loaded in BF16. No batching yet.

- Load model with FlashAttention enabled.
- Implement a single-request execution path.
- Implement SSE streaming.
- Add `/health` and `/metrics` (empty Prometheus registry).

### Week 2 — Static batching + iteration-level batching (10-12h)

**Gate**: System sustains 100 req/sec at TTFT P50 < 80 ms with
naive static batching (gather-N-requests-then-run).

- Implement a request queue.
- Implement a static batcher (wait up to T ms or until N requests
  arrive).
- Move to **iteration-level batching**: at every model step, decide
  which requests are still alive and pack them into the next forward.
- Verify correctness: per-request tokens identical to single-request
  baseline.

### Week 3 — PagedAttention KV cache (12-15h)

**Gate**: System sustains 400 req/sec, KV cache fragmentation < 5%,
no OOM on a 200-request burst at mean prompt 512 + output 256.

- Allocate a single contiguous KV cache buffer at startup
  (`num_blocks * block_size * num_heads * head_dim * 2`).
- Implement a `BlockManager` that hands out 16-token blocks via a
  free-list.
- Build a per-request `block_table: List[int]` mapping logical position
  to physical block.
- Wire the paged attention kernel: either FlashAttention with
  `paged_kv_block_size` or a custom Triton kernel.
- Implement copy-on-write semantics for prefix sharing.
- Add prefix cache: hash the prompt prefix, reuse blocks across
  requests that share it.

### Week 4 — Continuous batching + chunked prefill (10-12h)

**Gate**: System sustains 700 req/sec with TTFT P99 < 150 ms.

- Implement the continuous batching scheduler:
  - At step T, choose `decode_batch` (alive requests) +
    `prefill_batch` (newly admitted requests).
  - Mixed-mode forward pass that handles both in one launch.
- Implement **chunked prefill**: long prompts split into 512-token
  chunks so a single long prompt does not block decode steps.
- Add admission control: refuse requests when queue depth > K or KV
  cache free blocks < J.

### Week 5 — Streaming + control plane + observability (8-10h)

**Gate**: Grafana dashboard shows request rate, TTFT P50/P99, TPOT,
GPU SM busy %, KV cache free blocks. Alerts fire on SLO violation.

- Implement OpenAI-compatible chat streaming with `delta` events.
- Add cancellation: client disconnect within 100 ms removes the
  request from the next batch.
- Wire Prometheus client: histograms (TTFT, TPOT, E2E), gauges (queue
  depth, KV blocks free, KV blocks allocated, requests in flight),
  counters (admitted, rejected, OOM, completed).
- Build the Grafana dashboard (commit JSON).
- Define Prometheus recording rules and alerts (TTFT P99 SLO,
  goodput < 95%).

### Week 6 — Speculative decoding + CUDA Graphs (10-12h)

**Gate**: Speculative decoding lifts throughput by >= 1.5x at fixed
target model, with <= 1% accept-rate-driven extra cost. CUDA Graphs
capture of decode-only step shaves >= 30% latency at small batches.

- Add a draft model (~1B Llama-style for an 8B target).
- Implement the draft-then-verify loop with rejection sampling.
- Track accept rate; auto-tune draft length per request.
- Capture decode-only step into CUDA Graphs; replay per step.

### Week 7 — Load test + SLO tuning (10-12h)

**Gate**: All hard performance gates from Section 2 met against the
canonical `sharegpt_mixed.yaml` workload at 1000 req/sec offered.

- Locust scenario: poisson arrivals at configurable rate, mixed
  ShareGPT prompts, configurable max tokens.
- Run sweep: 100, 500, 1000, 1500 req/sec offered.
- Plot latency CDFs, throughput-vs-offered-load, goodput, queue depth.
- Tune scheduler weights (decode vs prefill prioritization, chunk
  size, admission thresholds).

### Week 8 — Baseline comparison + production hardening (10-12h)

**Gate**: vLLM 0.6+ comparison within 10% on the same workload;
container image < 4 GB; `make verify` reproduces.

- Run the exact same `sharegpt_mixed.yaml` against vLLM 0.6+ and
  TensorRT-LLM 0.10+ (where setup permits).
- Plot ours vs vLLM vs TRT-LLM on the same CDF axes.
- Write `reports/baseline_comparison.md` with "where we win, where we
  lose, why".
- Multi-stage Dockerfile, non-root user, read-only root filesystem,
  graceful shutdown via SIGTERM, healthcheck endpoint.

## 7. Architecture pointer

See [`architecture.md`](./architecture.md) for the full component
diagram, request lifecycle, scheduler design, and trade-off discussion
(continuous batching vs static, paged vs contiguous KV, draft model
selection for speculative decoding, etc.).

## 8. Step-by-step build guide

See [`STEP_BY_STEP.md`](./STEP_BY_STEP.md). Canonical "do this exactly"
path with code snippets, profiling commands, load test recipes, and
the gotcha list (scheduler thrash, KV cache compaction stalls, SSE
client cancellation races, Prometheus cardinality blowup, etc.).

## 9. Rubric summary

Full rubric in [`rubric.md`](./rubric.md). High-level dimensions:

1. **Throughput** (req/sec sustained)
2. **Latency** (TTFT P50/P99, TPOT P50/P99)
3. **Goodput / SLO compliance** (% of requests within SLO)
4. **KV cache management** (fragmentation, hit rate, no OOM under load)
5. **Code quality** (engine is line-by-line defensible)
6. **Observability** (metrics, dashboard, alerts)
7. **Engineering judgment** (scheduler design, admission policy,
   baseline comparison)
8. **Production hardening** (container, cancellation, graceful
   shutdown, security)

Pass = all hard performance gates met AND rubric score >= 3/5 in every
dimension. Distinction = rubric score >= 4/5 in at least 6 of 8.

## 10. Success criteria checklist

- [ ] Sustained >= 1000 req/sec on Llama-3 8B BF16 on H100.
- [ ] TTFT P50 <= 40 ms, P99 <= 100 ms.
- [ ] TPOT P50 <= 25 ms, P99 <= 60 ms.
- [ ] Goodput >= 95% at offered 1000 req/sec.
- [ ] PagedAttention fragmentation <= 5%.
- [ ] Prefix cache hit rate >= 40% on workload with shared system
      prompts.
- [ ] GPU SM busy >= 80% under sustained load.
- [ ] Within 10% of vLLM 0.6+ throughput on identical workload.
- [ ] Grafana dashboard captures TTFT, TPOT, throughput, KV cache,
      queue depth.
- [ ] OOM / cache exhaustion returns 429 with Retry-After.
- [ ] Client disconnect cancels request within 100 ms.
- [ ] Container image < 4 GB; runs as non-root.
- [ ] `make verify` reproduces all numbers within 5% on the same host.

## 11. Related modules

| Module | Why it matters |
|--------|----------------|
| mod-003 Performance Profiling | Nsight Systems for end-to-end timeline; SOL% under load. |
| mod-004 Transformer Optimization | Attention bottleneck; FlashAttention paged variant. |
| mod-005 Model Compression | Optional INT8 / FP8 weights using Project 1 output. |
| mod-007 Production Deployment | Container, K8s, SLO definition, graceful shutdown. |
| mod-008 Advanced Topics | Speculative decoding, FlashDecoding++, FP8 inference. |
| mod-009 Distributed Inference | Single-node here; tensor-parallel is the stretch path. |

## 12. Stretch goals (for distinction)

- **Tensor parallelism**: shard the model across 2 GPUs with NCCL all-reduce;
  re-run targets at higher throughput.
- **FP8 weights**: quantize via Project 1's pipeline; measure speedup.
- **Multi-LoRA serving**: hot-swap LoRA adapters per request without
  re-prefilling.
- **Disaggregated prefill/decode**: separate processes (or GPUs) for
  prefill and decode (a la DistServe / Splitwise).
- **Lookahead decoding** or **medusa heads** for speculative without
  a draft model.
- **Persistent connections**: HTTP/2 + connection pooling on the load
  test side; lower per-request overhead.

## 13. Out of scope

- Multi-node distributed serving (tensor parallelism is the stretch
  goal; pipeline parallelism is not in scope).
- Training infrastructure.
- Quantization training (use Project 1's output if needed).
- Custom CUDA kernel authoring (use Project 2's output or
  FlashAttention upstream).
- Image / multimodal inputs.

## 14. References

- Yu et al., **"Orca: A Distributed Serving System for Transformer-Based
  Generative Models"**, OSDI 2022. (Continuous batching origin.)
- Kwon et al., **"Efficient Memory Management for Large Language Model
  Serving with PagedAttention"**, SOSP 2023. (vLLM paper.)
- Leviathan et al., **"Fast Inference from Transformers via Speculative
  Decoding"**, ICML 2023.
- Chen et al., **"Accelerating Large Language Model Decoding with
  Speculative Sampling"**, 2023.
- Agrawal et al., **"SARATHI: Efficient LLM Inference by Piggybacking
  Decodes with Chunked Prefills"**, 2023.
- Zhong et al., **"DistServe: Disaggregating Prefill and Decoding for
  Goodput-Optimized LLM Serving"**, OSDI 2024.
- Patel et al., **"Splitwise: Efficient Generative LLM Inference Using
  Phase Splitting"**, ISCA 2024.
- vLLM source: github.com/vllm-project/vllm
- TensorRT-LLM source: github.com/NVIDIA/TensorRT-LLM
- Google SRE Workbook, chapters on SLOs and error budgets.
