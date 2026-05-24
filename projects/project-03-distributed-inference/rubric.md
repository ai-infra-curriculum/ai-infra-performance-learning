# Rubric — High-Performance LLM Inference System

Reviewers score each dimension 1-5. **Pass** requires all hard
performance gates from `requirements.md` Section 3 AND >= 3/5 in
every dimension. **Distinction** requires >= 4/5 in at least 6 of 8.

Hard gates fail the project regardless of rubric score:

- PR-1..PR-14 missed.
- `make verify` fails to reproduce within 5%.
- KV cache leak that grows monotonically over a 30-minute run.
- vLLM baseline not run on identical hardware with the same workload.
- Any unhandled exception in the request hot path under load.

## 1. Scoring scale

| Level | Meaning |
|-------|---------|
| 1 | Missing or fundamentally broken. |
| 2 | Partial; works at low load; collapses under stress. |
| 3 | Meets requirements; defensible. (Pass bar.) |
| 4 | Exceeds requirements with measured ablations. |
| 5 | Reference-quality; reviewer would deploy this to production. |

## 2. Dimensions

### D1. Throughput (weight: highest)

Sustained req/sec at the canonical workload.

| Level | Sustained req/sec (H100, Llama-3 8B BF16) |
|-------|---------------------------------------------|
| 1 | < 200 |
| 2 | 200-600 |
| 3 | 1000 (PR-1 met) |
| 4 | 1200 with goodput >= 95% |
| 5 | 1500 with goodput >= 95% and within 5% of vLLM |

### D2. Latency (TTFT + TPOT)

P50 and P99 at offered 1000 req/sec.

| Level | Evidence |
|-------|----------|
| 1 | TTFT P99 > 500 ms or TPOT P99 > 200 ms. |
| 2 | TTFT P99 200-500 ms. |
| 3 | TTFT P50 <= 40 ms, P99 <= 100 ms; TPOT P50 <= 25 ms, P99 <= 60 ms (PR-2..PR-5). |
| 4 | TTFT P99 <= 75 ms; TPOT P99 <= 45 ms. |
| 5 | TTFT P99 <= 50 ms; CUDA Graph + speculative both contributing measurably with documented numbers. |

### D3. Goodput / SLO compliance

% of requests within SLO at offered 1000 req/sec.

| Level | Goodput |
|-------|---------|
| 1 | < 70% |
| 2 | 70-90% |
| 3 | >= 95% (PR-10) |
| 4 | >= 98% with admission policy ablation documented. |
| 5 | >= 99% with cost-aware admission and per-tenant SLO classes. |

### D4. KV cache management

Fragmentation, prefix cache hit rate, no leaks under sustained load.

| Level | Evidence |
|-------|----------|
| 1 | OOM under sustained 500 req/sec. |
| 2 | Fragmentation > 10%; prefix cache absent. |
| 3 | Fragmentation <= 5% (PR-9); prefix cache hit rate >= 40% (PR-7). |
| 4 | Block leak test passes (30-min run, free count returns to baseline at idle). COW correctness test passes. |
| 5 | Fragmentation <= 2%; prefix cache hit rate >= 60% on shared-prompt workload; documented per-workload tuning of `block_size`. |

### D5. Code quality

Engine is line-by-line defensible.

| Level | Evidence |
|-------|----------|
| 1 | Thin wrapper around vLLM. |
| 2 | Scheduler is a 500-line god-method. |
| 3 | Module layout matches `architecture.md`; `mypy --strict`, `ruff`, `black` pass. |
| 4 | New scheduler can be plugged in via protocol with < 100 LOC of changes. ADRs document non-obvious choices. |
| 5 | Pull request reviewer in 30 minutes can walk every box of the request lifecycle and explain it. |

### D6. Observability

Metrics, dashboard, alerts.

| Level | Evidence |
|-------|----------|
| 1 | No Prometheus metrics. |
| 2 | Metrics exist but cardinality blow-up risk; dashboard incomplete. |
| 3 | All FR-11 metrics exported; Grafana dashboard with all required panels; alerts for SLO violations. |
| 4 | OTLP tracing; per-tenant counters; alert runbook in `docs/runbook.md`. |
| 5 | Recording rules for cost-per-request; per-tenant SLO dashboards; cost-attribution panel. |

### D7. Engineering judgment

Scheduler design, admission policy, baseline comparison.

| Level | Evidence |
|-------|----------|
| 1 | Defaults uncritically used; vLLM baseline missing. |
| 2 | One scheduler config; no sweep; vLLM comparison incomplete. |
| 3 | Scheduler tuning knobs documented; baseline comparison written. |
| 4 | Chunked prefill ablation showing TTFT improvement; speculative decoding accept rate sweep; "where we win, where we lose vs vLLM" documented. |
| 5 | Disaggregated prefill/decode or tensor-parallel ablation with numbers; per-workload tuning guide. |

### D8. Production hardening

Container, cancellation, graceful shutdown, security.

| Level | Evidence |
|-------|----------|
| 1 | Runs in a `python -m` from a notebook. |
| 2 | Dockerfile exists but image > 8 GB; runs as root. |
| 3 | Multi-stage Dockerfile, < 4 GB, non-root, SIGTERM drain working, healthcheck. |
| 4 | Read-only root FS in K8s manifest; PodDisruptionBudget; resource limits; pre-stop drain. |
| 5 | Chaos test: kill -9 to client mid-stream, kill -SIGTERM during peak load, NVML returns "disconnected" — all handled cleanly. |

## 3. Bonus rubric (distinction only)

- **B1**: Tensor parallelism across 2 GPUs with NCCL; throughput
  number reported.
- **B2**: FP8 weights via TransformerEngine with measured speedup and
  accuracy table.
- **B3**: Multi-LoRA serving: hot-swap LoRA per request without
  re-prefilling.
- **B4**: Disaggregated prefill/decode (DistServe / Splitwise style).
- **B5**: Medusa heads or lookahead decoding without a draft model.
- **B6**: OpenTelemetry traces shipped to Tempo/Jaeger.

## 4. Anti-patterns (auto-deduction)

- **Wrapping vLLM and calling it our engine**: auto-fail. The point
  is to build it.
- **Reporting numbers without locking GPU clocks**: -2 on D1 and D2.
- **`request.is_disconnected()` awaited in the hot loop**: -1 on D2
  (TPOT suffers).
- **Request-id used as a Prometheus label**: -2 on D6 (cardinality
  bomb).
- **No vLLM comparison run on the same hardware**: -2 on D7.
- **No chunked prefill (long prompts block decodes)**: -1 on D2 and
  D7.
- **Synchronous I/O on the hot path (blocking file reads, etc.)**:
  -1 on D5.
- **Container runs as root**: -1 on D8.
- **No SIGTERM drain (kills in-flight requests)**: -1 on D8.

## 5. Reviewer checklist

```
[ ] Hard gates PR-1..PR-14 met?      (else fail)
[ ] make verify reproduces in Docker? (else fail)
[ ] D1 Throughput:           _/5
[ ] D2 Latency:              _/5
[ ] D3 Goodput / SLO:        _/5
[ ] D4 KV cache management:  _/5
[ ] D5 Code quality:         _/5
[ ] D6 Observability:        _/5
[ ] D7 Engineering judgment: _/5
[ ] D8 Production hardening: _/5

Total: __/40
Pass:        all >= 3 and hard gates met
Distinction: >= 6 dimensions at 4+
```

## 6. How to defend your score

For each dimension, the reviewer expects ONE artifact + ONE sentence:

| Dimension | Artifact | Sentence |
|-----------|----------|----------|
| D1 | `reports/throughput_vs_offered.png` | "Knee at 1300 req/sec; sustained 1150 within SLO." |
| D2 | `reports/latency_cdf.png` | "TTFT P99 78 ms; speculative cut 22 ms; CUDA Graphs cut 8 ms." |
| D3 | `reports/goodput_chart.png` | "Goodput holds at 96% up to 1200 req/sec offered; degrades cleanly above." |
| D4 | `reports/kv_cache_timeseries.png` | "Fragmentation steady at 3.1%; prefix hit rate 52% on shared-system workload." |
| D5 | `src/engine/scheduler.py` (line count + complexity) | "Scheduler is 180 LOC; protocol allows alternate impl." |
| D6 | Grafana dashboard screenshot + `infra/alerts.yaml` | "Dashboard captures TTFT, TPOT, KV, GPU; 4 SLO alerts wired." |
| D7 | `reports/baseline_comparison.md` | "Within 7% of vLLM; we lose on long-prompt prefill due to no FlashInfer." |
| D8 | `infra/helm/values.yaml` + kill-test log | "Drains 84 in-flight requests in 12 s on SIGTERM; PodDisruptionBudget set." |
