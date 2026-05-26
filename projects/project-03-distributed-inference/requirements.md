# Requirements — High-Performance LLM Inference System

Every "MUST" is a gate. Every "SHOULD" is graded in the rubric.

## 1. Scope

### 1.1 In scope

- Single-node serving of one 7-8B BF16 decoder-only LLM
  (Llama-3 8B or Mistral-7B).
- Continuous batching with iteration-level scheduling.
- PagedAttention KV cache management.
- Prefix caching.
- Chunked prefill.
- Token streaming (SSE) with cancellation.
- Speculative decoding (draft + target verify).
- Prometheus metrics + Grafana dashboard.
- Load testing with reproducible workloads.
- vLLM 0.6+ and TensorRT-LLM 0.10+ comparison.
- Multi-tenant request tagging (per-API-key counters).
- Admission control + rate limiting.

### 1.2 Out of scope

- Multi-node distributed serving (single H100 is the target).
- Pipeline parallelism (tensor parallelism is a stretch goal only).
- Multimodal inputs (text only).
- Training / fine-tuning.
- Authentication (a single API key suffices; full OAuth is out of
  scope).
- Custom CUDA kernel authoring (use FlashAttention upstream or the
  output of Project 2).

## 2. Functional requirements

### FR-1: HTTP API (OpenAI-compatible)

- MUST expose `/v1/completions` (legacy) and `/v1/chat/completions`
  with streaming + non-streaming modes.
- MUST conform to OpenAI request/response schema for the subset of
  fields used (`model`, `messages`, `max_tokens`, `temperature`,
  `top_p`, `stream`, `stop`).
- MUST return JSON-formatted error responses on validation failure
  (HTTP 400) and capacity exhaustion (HTTP 429 with `Retry-After`).
- MUST expose `/health` (liveness) and `/ready` (readiness).
- MUST expose `/metrics` for Prometheus scraping.

### FR-2: Model loading

- MUST load Llama-3 8B (or Mistral-7B) weights from a configurable
  path or HuggingFace identifier.
- MUST use BF16 by default.
- MUST integrate FlashAttention 2.6+ (or FlashAttention-3 on H100)
  for the attention forward path.
- SHOULD support optional FP8 weights via TransformerEngine for a
  rubric bonus.

### FR-3: Iteration-level batching

- MUST schedule at the per-iteration granularity: at every forward
  step, re-decide which requests participate.
- MUST mix prefill and decode requests in the same forward pass.
- MUST evict completed requests from the batch immediately, not at
  the end of the prefill+decode group.

### FR-4: PagedAttention KV cache

- MUST allocate a single contiguous KV cache pool at startup of size
  `num_blocks * block_size * num_kv_heads * head_dim * 2 bytes_per_elt`.
- MUST manage allocation via a `BlockManager` with a free-list.
- MUST maintain a per-request `block_table` mapping logical position
  to physical block.
- MUST support `block_size` configurable (default 16 tokens).
- MUST support copy-on-write for prefix sharing.
- MUST report fragmentation (fraction of blocks held but unused
  internal positions) as a Prometheus gauge.

### FR-5: Prefix cache

- MUST hash the prompt prefix at block-aligned boundaries and reuse
  blocks across requests with matching prefixes.
- MUST report `prefix_cache_hits_total` and `prefix_cache_misses_total`
  as Prometheus counters.
- MUST support a configurable cache size + LRU eviction.

### FR-6: Chunked prefill

- MUST split long prompts into chunks (default 512 tokens) so that
  a long-prompt admission does not block decode steps for >= one
  chunk's worth of latency.
- MUST be tuneable via config.

### FR-7: Continuous batching scheduler

- MUST implement a scheduler that, per iteration, picks:
  - A `decode_batch` from alive requests.
  - A `prefill_batch` from admitted-but-not-yet-prefilled requests
    (or chunks thereof).
- MUST respect KV cache budget when admitting prefill.
- MUST prefer decode over prefill when KV cache is tight (avoid
  unbounded TTFT degradation under load).

### FR-8: Token streaming

- MUST stream tokens as SSE `data: {json}\n\n` events.
- MUST emit a final `data: [DONE]\n\n` marker.
- MUST close the SSE connection cleanly on completion or cancellation.
- MUST handle client disconnect: the request MUST be removed from the
  scheduler within 100 ms and its KV cache blocks freed.

### FR-9: Speculative decoding

- MUST support a configurable draft model.
- MUST implement the draft-then-verify loop using rejection sampling
  (Leviathan-style).
- MUST track accept rate per request and adapt draft length.
- MUST be toggleable per request (`use_speculative=true`).

### FR-10: Admission control + rate limiting

- MUST refuse requests when `queue_depth > max_queue_depth`.
- MUST refuse requests when KV cache free blocks < `min_free_blocks`.
- MUST enforce a per-API-key rate limit using Redis as the counter
  store.
- MUST emit `rejected_admission_total{reason}` counter.

### FR-11: Prometheus metrics

- MUST export at minimum:
  - `inference_requests_total{status,model}`
  - `inference_ttft_seconds` (histogram)
  - `inference_tpot_seconds` (histogram)
  - `inference_e2e_seconds` (histogram)
  - `inference_tokens_generated_total`
  - `kv_cache_blocks_allocated`
  - `kv_cache_blocks_free`
  - `kv_cache_fragmentation_ratio`
  - `prefix_cache_hits_total`, `prefix_cache_misses_total`
  - `scheduler_queue_depth`
  - `gpu_sm_busy_ratio` (sampled via NVML)
  - `gpu_memory_used_bytes` (sampled via NVML)
- Label cardinality MUST be bounded (no labels containing request IDs).

### FR-12: Grafana dashboard

- MUST ship a dashboard JSON checked into `infra/grafana/`.
- MUST include panels for: request rate, TTFT P50/P99, TPOT P50/P99,
  GPU SM busy %, KV cache occupancy and fragmentation, queue depth,
  goodput.

### FR-13: Load test harness

- MUST be a single command: `make load workload=sharegpt_mixed rate=1000`.
- MUST use poisson arrivals.
- MUST output JSONL of per-request timings.
- MUST plot a CDF and a throughput-vs-offered-load chart.

### FR-14: Baseline comparison

- MUST be able to run the exact same workload against vLLM 0.6+
  (`make bench-vllm`) and TensorRT-LLM 0.10+ (`make bench-trt-llm`)
  with the same client.
- MUST produce a side-by-side comparison report.

### FR-15: Graceful shutdown

- MUST handle SIGTERM by:
  1. Stop accepting new requests (return 503 on `/ready`).
  2. Drain in-flight requests with a 60-second timeout.
  3. Force-cancel remaining at timeout with explicit error.
  4. Free GPU memory cleanly.

## 3. Performance requirements (hard gates)

Measured on **1x H100 SXM5 80GB**, Llama-3 8B BF16, workload
`bench/workloads/sharegpt_mixed.yaml`.

| ID | Metric | Target |
|----|--------|--------|
| PR-1 | Sustained throughput | >= 1000 req/sec successfully completed |
| PR-2 | TTFT P50 | <= 40 ms |
| PR-3 | TTFT P99 | <= 100 ms |
| PR-4 | TPOT P50 | <= 25 ms |
| PR-5 | TPOT P99 | <= 60 ms |
| PR-6 | E2E P99 (200-token completion) | <= 5.0 s |
| PR-7 | Prefix cache hit rate | >= 40% on the shared-system-prompt workload |
| PR-8 | GPU SM busy under load | >= 80% |
| PR-9 | KV cache fragmentation | <= 5% |
| PR-10 | Goodput at 1000 req/sec offered | >= 95% within SLO |
| PR-11 | Throughput vs vLLM 0.6 | within 10% (slower OK) |
| PR-12 | OOM / cache exhaustion response | HTTP 429 with `Retry-After` |
| PR-13 | Client disconnect cancellation | <= 100 ms |
| PR-14 | Container size | <= 4 GB compressed |
| PR-15 | Throughput vs TensorRT-LLM 0.10 | within 25% (soft) |

A100-only acceptance: PR-1 relaxes to >= 600 req/sec; other gates
stand.

## 4. Non-functional requirements

### NFR-1: Code structure

- File size <= 800 LOC; function size <= 50 LOC.
- Type hints on every public function; `mypy --strict` passes.
- `ruff` and `black` pass.
- Clear separation: `engine/`, `cache/`, `api/`, `spec/`, `kernels/`.

### NFR-2: Testing

- >= 80% statement coverage on `src/`.
- Unit tests for: block manager, prefix cache hash + COW, scheduler
  fairness, admission policy, request lifecycle state machine,
  speculative verify rejection.
- Integration tests:
  - Toy model end-to-end (DistilGPT2 or a 100M test model) runs in CI.
  - Streaming + cancellation test.
  - OOM + 429 + Retry-After test.
- Load smoke test: 60 seconds at 100 req/sec must pass in CI on a
  small model.

### NFR-3: Containerization

- Multi-stage Dockerfile.
- Final image based on `nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04`
  (or `nvcr.io/nvidia/pytorch:24.07-py3` if size budget permits).
- Runs as non-root user (`uid=1000`).
- Healthcheck via `/health`.
- `STOPSIGNAL SIGTERM`.

### NFR-4: Observability

- All logs structured JSON to stderr in production mode.
- Trace ID in every log line for request correlation.
- Prometheus metrics endpoint at `/metrics`.
- Optional OpenTelemetry export (env-gated).

### NFR-5: Security

- No secrets in code or configs.
- API key loaded from env (`INFERENCE_API_KEY`).
- All admin endpoints (`/admin/...`) require the same key.
- Per-API-key rate limits enforced.
- Read-only root filesystem in K8s manifest.

### NFR-6: Documentation

- `docs/api.md` with full OpenAI-compatible schema and deviations.
- `docs/scheduler.md` with state diagram and tuning knobs.
- `docs/runbook.md` with on-call procedures (cache exhaustion, OOM,
  GPU disappearance).

## 5. Constraints

- Single-GPU only by default.
- BF16 weights by default (FP8 a stretch).
- One model per process by default (multi-model is a stretch).
- Python primary; targeted CUDA / Triton kernels OK where justified.

## 6. Assumptions

- Reviewer has at least one H100 80GB (preferred) or A100 80GB.
- Reviewer has the Llama-3 8B or Mistral-7B weights legally available.
- Reviewer can run `nvidia-smi -lgc` to lock clocks.
- Reviewer has Docker / containerd + nvidia-container-toolkit.

## 7. Dependencies (external)

| Component | Version | Reason |
|-----------|---------|--------|
| CUDA Toolkit | 12.4+ | FlashAttention / paged variant |
| PyTorch | 2.4+ | BF16, SDPA paged variant hooks |
| FlashAttention | 2.6+ (or FA3) | Attention forward |
| vLLM | 0.6+ | Baseline comparison |
| TensorRT-LLM | 0.10+ | Baseline comparison (optional) |
| FastAPI | 0.110+ | HTTP layer |
| uvicorn | 0.30+ | ASGI server |
| Prometheus | 2.50+ | Metrics scrape |
| Grafana | 10+ | Dashboard |
| Locust | 2.25+ | Load test |
| Redis | 7+ | Rate limit state |
| pynvml | latest | GPU telemetry |

## 8. Acceptance test sketch

```
git clone <repo>
cd project-03-distributed-inference
docker build -t llm-serve:dev .

# Start the serving stack (engine + prometheus + grafana + redis)
docker compose up -d

# Run the canonical workload
make load workload=sharegpt_mixed rate=1000 duration=300

# Compare against vLLM
make bench-vllm workload=sharegpt_mixed rate=1000 duration=300

# Generate the report
make report
```

Acceptance is granted if:

1. `make load` exits 0.
2. `reports/load_summary.md` shows all PR-* hard gates met.
3. `reports/baseline_comparison.md` shows within-10%-of-vLLM throughput.
4. `make verify` reproduces the load test numbers within 5%.
5. Grafana dashboard loads cleanly with no missing panels.

## 9. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Clock unlocking causes flaky TTFT | High | High | Hard pre-test gate; abort load run if unlocked |
| KV cache fragmentation explodes under varied lengths | High | High | PagedAttention by design; gauge + alert |
| SSE cancellation race leaks KV blocks | Medium | Medium | Per-request cleanup hook with idempotent free |
| Prometheus cardinality blowup (request-id labels) | Medium | High | Code review + lint rule; bounded label set |
| Scheduler thrash under burst arrival | Medium | High | Admission control + tested under poisson with sigma sweep |
| vLLM version skew makes comparison unfair | Medium | Low | Pin in Dockerfile; report version in `baseline_comparison.md` |
| Draft model wrong tokenizer kills speculative | Low | High | Init-time assertion that draft and target share vocab |
| Long-prompt admission stalls all decodes | High | High | Chunked prefill (FR-6) |
| GPU OOM under simultaneous prefill | Medium | High | Admission control on free blocks (FR-10) |
| Locust client itself becomes bottleneck | Medium | Medium | Distributed Locust workers; client-side CPU pinning |

## 10. Glossary

- **TTFT**: Time to first token. Latency from request arrival to the
  first streamed token.
- **TPOT**: Time per output token. Steady-state inter-token latency.
- **Goodput**: Throughput of requests that meet their SLO.
- **PagedAttention**: KV cache management scheme using fixed-size
  blocks with a per-request block table.
- **Continuous batching**: Per-iteration request scheduling that
  packs decode + prefill into the same forward pass.
- **Chunked prefill**: Splitting long prompts into chunks to bound
  per-iteration latency.
- **Speculative decoding**: Predicting K tokens with a draft model
  and verifying with the target model in one forward.
- **Prefix cache**: Reusing KV blocks across requests that share a
  common prompt prefix.
- **SLO**: Service Level Objective.
- **SSE**: Server-Sent Events.
