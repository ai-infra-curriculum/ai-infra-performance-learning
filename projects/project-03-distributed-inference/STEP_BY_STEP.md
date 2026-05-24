# Step-by-Step Build Guide — High-Performance LLM Inference System

This is the canonical walkthrough. Do the phases in order; do not skip
a gate check. Every code snippet here is illustrative; the real
implementation lives in `src/`.

The full build is eight weeks at ~10h/week. Use this document as your
weekly checklist.

---

## Phase 0 — Environment, before week 1 (2-3h)

### 0.1 Verify hardware and lock clocks

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi -q -d CLOCK
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# H100 SXM5 — base clocks
sudo nvidia-smi --persistence-mode=1
sudo nvidia-smi --lock-gpu-clocks=1830,1830
sudo nvidia-smi --lock-memory-clocks=2619

# A100 80GB SXM — base clocks
# sudo nvidia-smi --lock-gpu-clocks=1410,1410
# sudo nvidia-smi --lock-memory-clocks=1593
```

**Gate 0**: `nvidia-smi -q -d CLOCK` shows clocks locked.

### 0.2 Repo skeleton

```
project-03-distributed-inference/
  README.md
  requirements.md
  architecture.md
  STEP_BY_STEP.md
  rubric.md
  Makefile
  Dockerfile
  docker-compose.yaml
  pyproject.toml
  src/
    engine/
      __init__.py
      queue.py
      scheduler.py
      executor.py
      sampler.py
      request.py
    cache/
      block_manager.py
      prefix_cache.py
      cow.py
    kernels/
      paged_attention.py
    spec/
      draft_target.py
    api/
      app.py
      schemas.py
      streaming.py
      admission.py
      rate_limit.py
    obs/
      metrics.py
      nvml_sampler.py
      logger.py
    main.py
  bench/
    workloads/
      sharegpt_mixed.yaml
      synthetic_short.yaml
      synthetic_long.yaml
    locust_runner.py
    poisson_client.py
    compare_baselines.py
  infra/
    prometheus.yaml
    grafana/
      dashboard.json
      datasources.yaml
    alerts.yaml
    helm/
  tests/
    unit/
    integration/
    load/
  reports/
  deliverables/
  docs/
    api.md
    scheduler.md
    runbook.md
```

### 0.3 Dockerfile (multi-stage)

```Dockerfile
# Stage 1: builder
FROM nvcr.io/nvidia/pytorch:24.07-py3 AS builder
WORKDIR /build
COPY pyproject.toml ./
RUN pip wheel --wheel-dir=/wheels \
        vllm==0.6.* tensorrt-llm==0.10.* \
        flash-attn==2.6.* --no-build-isolation \
        fastapi uvicorn pydantic redis pynvml \
        prometheus-client locust

# Stage 2: runtime
FROM nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN useradd -m -u 1000 inference
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip && rm -rf /var/lib/apt/lists/*
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels \
        fastapi uvicorn pydantic redis pynvml prometheus-client
COPY --chown=inference:inference src/ /app/src/
COPY --chown=inference:inference infra/ /app/infra/
USER inference
WORKDIR /app
HEALTHCHECK --interval=10s CMD curl -f http://localhost:8000/health || exit 1
STOPSIGNAL SIGTERM
CMD ["python3", "-m", "src.main"]
```

---

## Phase 1 — Naive serving (week 1, 8-10h)

### 1.1 Single-request execution path

```python
# src/engine/executor.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class NaiveExecutor:
    def __init__(self, model_id: str):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).cuda()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def generate(self, prompt: str, max_tokens: int):
        ids = self.tok(prompt, return_tensors="pt").input_ids.cuda()
        out = self.model.generate(ids, max_new_tokens=max_tokens,
                                  do_sample=False)
        yield self.tok.decode(out[0][ids.shape[1]:])
```

### 1.2 FastAPI app + SSE

```python
# src/api/app.py
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
import json

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    async def stream():
        for tok in executor.generate(body["messages"][-1]["content"],
                                     body.get("max_tokens", 256)):
            if await req.is_disconnected():
                return
            yield json.dumps({"delta": {"content": tok}})
        yield "[DONE]"
    return EventSourceResponse(stream())

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/metrics")
def metrics():
    from prometheus_client import generate_latest
    return Response(generate_latest(), media_type="text/plain")
```

### Gate 1

- [ ] `/v1/chat/completions` streams tokens via SSE.
- [ ] `/health` returns 200.
- [ ] `/metrics` returns valid Prometheus output (empty is OK here).
- [ ] Single-request latency < 200 ms TTFT on short prompt.

---

## Phase 2 — Iteration-level batching (week 2, 10-12h)

### 2.1 Request abstraction

```python
# src/engine/request.py
from dataclasses import dataclass, field
from enum import Enum

class State(Enum):
    QUEUED = "queued"; PREFILLING = "prefilling"
    DECODING = "decoding"; DONE = "done"; CANCELLED = "cancelled"

@dataclass
class Request:
    id: str
    prompt_ids: list[int]
    max_tokens: int
    output_ids: list[int] = field(default_factory=list)
    state: State = State.QUEUED
    block_table: list[int] = field(default_factory=list)
    arrival_time: float = 0.0
    first_token_time: float | None = None
    is_disconnected: callable = lambda: False
```

### 2.2 Iteration-level scheduler v0 (no paging yet)

```python
# src/engine/scheduler.py
class SimpleContBatchScheduler:
    def __init__(self, max_batch=128):
        self.max_batch = max_batch
        self.alive: list[Request] = []
        self.queue: list[Request] = []

    def step(self):
        # admit
        while self.queue and len(self.alive) < self.max_batch:
            r = self.queue.pop(0)
            r.state = State.PREFILLING
            self.alive.append(r)
        # remove completed
        self.alive = [r for r in self.alive if r.state != State.DONE]
        return self.alive
```

### 2.3 Packed forward (varlen)

The trick: batch is a concatenation of N requests' "next" tokens
(decode) and M requests' "next prefill chunk" (prefill). We use
FlashAttention's varlen API.

```python
input_ids   = torch.cat([torch.tensor(r.prompt_ids) if r.state == PREFILLING
                         else torch.tensor([r.output_ids[-1]])
                         for r in batch]).cuda()
cu_seqlens  = torch.tensor([0] + cumulative_lengths).cuda().int()
max_seqlen  = max(seq_lengths)
hidden      = model.forward(input_ids,
                            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
```

### Gate 2

- [ ] 100 req/sec sustained for 60 seconds.
- [ ] TTFT P50 < 80 ms.
- [ ] Per-request output identical to single-request baseline (bit
      check on greedy decoding).

### Gotchas

- **`cu_seqlens` must be int32, not int64**: FA varlen will silently
  produce garbage otherwise.
- **`max_seqlen` must be at least the longest seq in the batch**:
  off-by-one causes incorrect masks.

---

## Phase 3 — PagedAttention (week 3, 12-15h)

### 3.1 KV pool allocation

```python
# src/cache/block_manager.py
BLOCK_SIZE = 16

class BlockManager:
    def __init__(self, num_blocks, num_layers, num_kv_heads, head_dim,
                 dtype=torch.bfloat16):
        self.num_blocks = num_blocks
        # Single contiguous buffer per layer.
        # Shape: (num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim)
        self.kv = torch.empty(
            (num_layers, 2, num_blocks, BLOCK_SIZE, num_kv_heads, head_dim),
            dtype=dtype, device="cuda")
        self.free: list[int] = list(range(num_blocks))
        self.refcount = [0] * num_blocks

    def alloc(self, n=1) -> list[int]:
        if len(self.free) < n: raise NoFreeBlocks
        out = [self.free.pop() for _ in range(n)]
        for b in out: self.refcount[b] = 1
        return out

    def free_blocks(self, block_ids):
        for b in block_ids:
            self.refcount[b] -= 1
            if self.refcount[b] == 0:
                self.free.append(b)

    def can_admit(self, prompt_len: int, max_tokens: int) -> bool:
        need = (prompt_len + max_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        return len(self.free) >= need

    def fragmentation(self) -> float:
        used_blocks = self.num_blocks - len(self.free)
        if used_blocks == 0: return 0.0
        # Computed from per-request block_tables; sketch only here.
        return 0.0  # see real impl
```

### 3.2 Per-request block table

```python
# In Request:
#   block_table: list[int]  -- one block_id per BLOCK_SIZE-aligned chunk
#   logical_len: int        -- number of tokens stored in KV
#
# When a request grows past `len(block_table) * BLOCK_SIZE`,
# allocate one more block and append to block_table.
```

### 3.3 Wire paged attention

Use FlashAttention's `flash_attn_with_kvcache`:

```python
from flash_attn import flash_attn_with_kvcache

def paged_attention(q, kv_pool_k, kv_pool_v, block_table, context_lens):
    # q:               (total_tokens, num_heads, head_dim)
    # kv_pool_k/v:     (num_blocks, block_size, num_kv_heads, head_dim)
    # block_table:     (batch, max_blocks)  int32, padded with -1
    # context_lens:    (batch,)             int32
    return flash_attn_with_kvcache(
        q.unsqueeze(0),                 # (1, T, H, D) packed varlen
        kv_pool_k, kv_pool_v,
        cache_seqlens=context_lens,
        block_table=block_table,
        causal=True,
    )
```

### 3.4 Prefix cache + COW

```python
# src/cache/prefix_cache.py
import xxhash

class PrefixCache:
    def __init__(self, bm: BlockManager, max_entries=10000):
        self.bm = bm
        self.entries: dict[int, list[int]] = {}  # hash -> block_ids
        self.lru = []

    def lookup_and_acquire(self, prompt_ids) -> list[int]:
        """Walk prompt in BLOCK_SIZE chunks; for each prefix hash,
        try to reuse blocks. Returns the prefix's block_ids."""
        out = []
        h = xxhash.xxh64()
        for i in range(0, len(prompt_ids) - BLOCK_SIZE + 1, BLOCK_SIZE):
            h.update(bytes(prompt_ids[i:i+BLOCK_SIZE]))
            key = h.intdigest()
            if key in self.entries:
                blks = self.entries[key]
                for b in blks: self.bm.refcount[b] += 1
                out.extend(blks)
                # MOVE-TO-FRONT for LRU
            else:
                break
        return out

    def insert(self, key: int, block_ids: list[int]):
        if len(self.entries) >= self.max_entries: self._evict()
        self.entries[key] = block_ids
```

### Gate 3

- [ ] 200-request burst at mean prompt 512 + max_tokens 256 completes
      without OOM.
- [ ] 400 req/sec sustained for 60 seconds.
- [ ] Fragmentation < 5% across the run.
- [ ] Prefix cache hit rate non-zero on a workload with shared system
      prompts.

### Gotchas

- **FlashAttention block_table expects int32 with -1 padding**: if you
  pass int64 or 0 padding you get NaN.
- **Refcount underflow on cancellation race**: free() called twice
  silently corrupts the free list. Use a set-based free list for
  development and switch to a deque only after tests pass.
- **Prefix cache + COW deadlock under concurrent writes**: serialize
  COW on a per-block lock; never hold the global cache lock during
  GPU memcpy.

---

## Phase 4 — Continuous batching + chunked prefill (week 4, 10-12h)

### 4.1 Mixed-mode iteration

Decode requests contribute 1 token each. Prefill chunks contribute
`chunk_size` tokens. They are concatenated into one packed batch.

```python
def build_iteration(decode_batch, prefill_chunks):
    input_ids, positions, slot_mapping, block_tables, context_lens = [], [], [], [], []
    cu = [0]
    for r in decode_batch:
        input_ids.append(r.output_ids[-1])
        positions.append(r.logical_len)
        slot_mapping.append(r.block_table[r.logical_len // BLOCK_SIZE] * BLOCK_SIZE
                            + r.logical_len % BLOCK_SIZE)
        block_tables.append(r.block_table)
        context_lens.append(r.logical_len)
        cu.append(cu[-1] + 1)
    for (r, chunk) in prefill_chunks:
        input_ids.extend(chunk.tokens)
        for j in range(len(chunk.tokens)):
            pos = r.logical_len + j
            positions.append(pos)
            slot_mapping.append(r.block_table[pos // BLOCK_SIZE] * BLOCK_SIZE
                                + pos % BLOCK_SIZE)
        block_tables.append(r.block_table)
        context_lens.append(r.logical_len + len(chunk.tokens))
        cu.append(cu[-1] + len(chunk.tokens))
    return Packed(...)
```

### 4.2 Chunked prefill

```python
# src/engine/request.py
def next_prefill_chunk(self, remaining: int) -> Chunk | None:
    if self.logical_len >= len(self.prompt_ids): return None
    start = self.logical_len
    end = min(start + min(CHUNK_SIZE, remaining), len(self.prompt_ids))
    if end == start: return None
    return Chunk(tokens=self.prompt_ids[start:end], req=self)
```

### 4.3 Admission control

```python
# src/api/admission.py
class AdmissionController:
    def can_admit(self, r: Request) -> tuple[bool, str]:
        if self.scheduler.queue_depth() >= MAX_QUEUE:
            return False, "queue_full"
        if self.bm.free_count() < MIN_FREE_BLOCKS:
            return False, "kv_cache_tight"
        if not self.bm.can_admit(len(r.prompt_ids), r.max_tokens):
            return False, "would_exhaust_kv"
        return True, ""
```

### Gate 4

- [ ] 700 req/sec sustained for 60 seconds.
- [ ] TTFT P99 < 150 ms (we'll tighten further later).
- [ ] No OOM under sustained 1000 req/sec offered load (some 429s OK).

---

## Phase 5 — Streaming, cancellation, observability (week 5, 8-10h)

### 5.1 Cancellation hook

```python
# src/api/app.py — inside the SSE stream
async def stream():
    cancel_event = asyncio.Event()
    asyncio.create_task(watch_disconnect(req, cancel_event))
    async for tok in engine.tokens_for(request_id):
        if cancel_event.is_set():
            engine.cancel(request_id)   # frees KV blocks within next iter
            return
        yield format_sse(tok)
    yield format_sse_done()

async def watch_disconnect(req, ev):
    while not ev.is_set():
        if await req.is_disconnected():
            ev.set(); return
        await asyncio.sleep(0.05)   # 50 ms poll => ~100 ms p99 cancel
```

### 5.2 Prometheus metrics

```python
# src/obs/metrics.py
from prometheus_client import Counter, Histogram, Gauge

TTFT_BUCKETS = [.005,.01,.02,.04,.08,.12,.2,.4,.8,1.6,3.2]
TPOT_BUCKETS = [.005,.01,.015,.02,.025,.03,.05,.08,.15]
E2E_BUCKETS  = [.1,.25,.5,1.0,2.0,4.0,8.0,16.0,30.0]

REQ_TOTAL = Counter("inference_requests_total", "", ["status","model"])
TTFT = Histogram("inference_ttft_seconds", "", ["model"], buckets=TTFT_BUCKETS)
TPOT = Histogram("inference_tpot_seconds", "", ["model"], buckets=TPOT_BUCKETS)
E2E  = Histogram("inference_e2e_seconds", "", ["model"], buckets=E2E_BUCKETS)
TOKENS = Counter("inference_tokens_generated_total", "", ["model"])
KV_FREE = Gauge("kv_cache_blocks_free", "")
KV_ALLOC = Gauge("kv_cache_blocks_allocated", "")
KV_FRAG = Gauge("kv_cache_fragmentation_ratio", "")
PFX_HIT = Counter("prefix_cache_hits_total", "")
PFX_MISS = Counter("prefix_cache_misses_total", "")
Q_DEPTH = Gauge("scheduler_queue_depth", "")
SM_BUSY = Gauge("gpu_sm_busy_ratio", "")
GPU_MEM = Gauge("gpu_memory_used_bytes", "")
REJECTED = Counter("rejected_admission_total", "", ["reason"])
```

### 5.3 NVML sampler

```python
# src/obs/nvml_sampler.py
import pynvml, asyncio
async def sample_loop():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    while True:
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        SM_BUSY.set(u.gpu / 100.0)
        GPU_MEM.set(m.used)
        await asyncio.sleep(0.25)
```

### 5.4 Grafana dashboard

Build in the UI; export JSON to `infra/grafana/dashboard.json`.
Required panels:

- Request rate (requests/sec, by status).
- TTFT P50 / P99 timeseries.
- TPOT P50 / P99 timeseries.
- E2E P99 timeseries.
- Goodput % within SLO.
- KV cache: blocks_free, blocks_allocated, fragmentation.
- Prefix cache hit rate.
- Queue depth.
- GPU SM busy %, GPU memory used.
- Rejections by reason (stacked).

### Gate 5

- [ ] Streaming works end-to-end via curl with `-N` (no buffering).
- [ ] Cancellation frees blocks within 100 ms of disconnect.
- [ ] Grafana dashboard renders without panel errors.
- [ ] Alert fires when TTFT P99 > 100 ms for 2 min.

### Gotchas

- **Don't put request_id in labels**: Prometheus cardinality
  explodes; switch to logs or traces.
- **`request.is_disconnected()` is async and slow**: cache the result
  with a 50 ms poll instead of awaiting in the hot loop.
- **SSE keep-alive comments needed for browser clients**: every 15 s
  send `:keepalive\n\n`.

---

## Phase 6 — Speculative decoding + CUDA Graphs (week 6, 10-12h)

### 6.1 Draft + verify loop

```python
# src/spec/draft_target.py
@torch.inference_mode()
def speculative_step(draft, target, request, K=4):
    # 1. Draft K tokens autoregressively
    draft_tokens, draft_probs = [], []
    for _ in range(K):
        logits = draft.forward(request)
        tok, p = sample_with_prob(logits)
        draft_tokens.append(tok); draft_probs.append(p)
        request.append_provisional(tok)

    # 2. Target verifies all K in one forward
    target_logits = target.forward_batched(request, last_n=K)
    target_probs = softmax(target_logits)

    # 3. Accept or reject each
    accepted = []
    for i in range(K):
        r = uniform(0,1)
        ratio = target_probs[i][draft_tokens[i]] / draft_probs[i]
        if r <= min(1.0, ratio):
            accepted.append(draft_tokens[i])
        else:
            corrected_p = relu(target_probs[i] - draft_probs[i])
            corrected_p /= corrected_p.sum()
            accepted.append(sample(corrected_p))
            break

    request.commit(accepted)
    return len(accepted)
```

### 6.2 CUDA Graphs for decode-only step

```python
# Decode-only iterations are highly repeatable shape: 1 token per
# request, fixed batch size. Capture per batch-size bucket.

graphs: dict[int, torch.cuda.CUDAGraph] = {}

def replay_decode(batch_size: int, inputs):
    if batch_size not in graphs:
        # Warm
        for _ in range(5): model(inputs)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): out_buf = model(inputs)
        graphs[batch_size] = (g, out_buf, inputs)
    g, out_buf, inp_buf = graphs[batch_size]
    inp_buf.copy_(inputs)
    g.replay()
    return out_buf.clone()
```

Capture per bucket (e.g. powers of 2: 1, 2, 4, 8, 16, 32, 64, 128).
Snap actual batch sizes to the next bucket up.

### Gate 6

- [ ] Speculative decoding +>= 1.5x throughput at K=4.
- [ ] CUDA Graphs decode step shaves >= 30% latency at small batch.
- [ ] Accept rate >= 65% on the canonical workload at K=4.

### Gotchas

- **Draft tokenizer != target tokenizer = silent corruption**: assert
  vocab equality at init.
- **CUDA Graph captures stale tensor addresses**: always copy into a
  pre-allocated input buffer; never re-allocate.
- **CUDA Graphs + dropout in inference mode**: dropout is off in
  inference, but some implementations still leave RNG state mutation
  in the graph. Audit for `torch.empty` calls inside the forward.

---

## Phase 7 — Load test + SLO tuning (week 7, 10-12h)

### 7.1 Poisson client

```python
# bench/poisson_client.py
import asyncio, aiohttp, random, time, json

async def submit(session, prompt, max_tokens, results):
    t0 = time.perf_counter()
    first = None
    async with session.post(URL, json={...}) as r:
        async for line in r.content:
            if not line.startswith(b"data: "): continue
            if first is None: first = time.perf_counter()
            if line.strip() == b"data: [DONE]": break
    t1 = time.perf_counter()
    results.append({"ttft": first - t0, "e2e": t1 - t0})

async def driver(rate_qps: float, duration: float):
    results = []
    async with aiohttp.ClientSession() as s:
        end = time.perf_counter() + duration
        tasks = []
        while time.perf_counter() < end:
            tasks.append(asyncio.create_task(submit(s, *next_prompt(), results)))
            # Poisson inter-arrival
            await asyncio.sleep(random.expovariate(rate_qps))
        await asyncio.gather(*tasks)
    return results
```

### 7.2 Workload spec

```yaml
# bench/workloads/sharegpt_mixed.yaml
source: sharegpt_v3
sample_size: 100000
prompt_length_distribution:
  mean: 512
  p99: 2048
output_length_distribution:
  mean: 256
  p99: 1024
shared_system_prompt:
  enabled: true
  text: "You are a helpful assistant. Respond concisely."
  apply_to_fraction: 0.6
slo:
  ttft_p50_ms: 40
  ttft_p99_ms: 100
  tpot_p50_ms: 25
  tpot_p99_ms: 60
  e2e_p99_s: 5.0
```

### 7.3 Sweep

```bash
for rate in 100 250 500 750 1000 1250 1500; do
    make load workload=sharegpt_mixed rate=$rate duration=300 \
         > reports/load_${rate}.json
done
python -m bench.plot_sweep reports/load_*.json \
    > reports/throughput_vs_offered.png
```

### Gate 7

- [ ] All hard PR-* gates met at offered 1000 req/sec.
- [ ] Goodput chart shows clean knee somewhere between 1000 and 1500.

### Tuning knobs (typical productive values)

| Knob | Default | When to raise | When to lower |
|------|---------|----------------|----------------|
| `max_decode_batch_size` | 96 | TTFT good, TPOT high | TPOT good, TTFT high |
| `max_prefill_tokens_per_iter` | 1024 | Long prompts queued up | Decode P99 too high |
| `chunk_size` | 512 | TTFT spiky on long prompts | Decode P99 climbs |
| `low_water_mark` | 0.10 * num_blocks | Decode P99 spikes | Throughput plateaus |
| `block_size` | 16 | Mostly long prompts | Mostly short prompts |
| `spec_K` | 4 | Accept rate high | Accept rate < 50% |

---

## Phase 8 — Baseline comparison + hardening (week 8, 10-12h)

### 8.1 vLLM baseline

```bash
# Run vLLM as a sidecar
docker run -d --gpus all -p 8001:8000 \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:v0.6.3 \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 8192

# Run the same workload against it
python -m bench.poisson_client \
    --url http://localhost:8001/v1/chat/completions \
    --workload bench/workloads/sharegpt_mixed.yaml \
    --rate 1000 --duration 300 \
    --out reports/vllm_load.jsonl
```

### 8.2 TensorRT-LLM baseline

```bash
# Build engine ahead of time (one-time)
trtllm-build --checkpoint_dir /tmp/llama3_8b_ckpt \
    --output_dir /tmp/llama3_8b_engine \
    --gemm_plugin auto \
    --max_batch_size 256 --max_seq_len 8192

# Serve via Triton or trtllm-serve
trtllm-serve serve /tmp/llama3_8b_engine --port 8002 &
python -m bench.poisson_client --url http://localhost:8002/... \
    --rate 1000 --duration 300 \
    --out reports/trtllm_load.jsonl
```

### 8.3 Side-by-side comparison

```bash
python -m bench.compare_baselines \
    --ours reports/load_1000.jsonl \
    --vllm reports/vllm_load.jsonl \
    --trtllm reports/trtllm_load.jsonl \
    > reports/baseline_comparison.md
```

### 8.4 Production hardening checklist

- [ ] Multi-stage Dockerfile; final image < 4 GB compressed.
- [ ] Non-root user (uid 1000).
- [ ] Read-only root filesystem in K8s manifest.
- [ ] Resource limits (cpu, memory, nvidia.com/gpu).
- [ ] Liveness + readiness probes pointed at `/health` + `/ready`.
- [ ] PreStop hook for graceful shutdown:
      `lifecycle: { preStop: { exec: { command: ["/app/drain.sh"] } } }`
- [ ] `terminationGracePeriodSeconds: 90`.
- [ ] PodDisruptionBudget for rolling updates.
- [ ] PromQL recording rules in `infra/alerts.yaml`.

### Gate 8

- [ ] vLLM within 10% throughput.
- [ ] TensorRT-LLM run logged (within 25% soft).
- [ ] Container builds, runs, drains on SIGTERM cleanly.
- [ ] `make verify` reproduces all numbers within 5%.

### Final gotchas

- **Locust client itself bottlenecks at high QPS**: distribute Locust
  workers; or use `poisson_client.py` with uvloop.
- **Prometheus scrape interval too high masks bursts**: set 1s scrape
  for the inference job during benchmarks; 15s for steady state.
- **Comparing apples to oranges**: vLLM defaults to different max
  context, different sampling, different batching limits. Match them
  explicitly in `bench/configs/`.

---

## Appendix A — Profiling commands

```bash
# Full end-to-end timeline under load
nsys profile -o profiles/nsys_e2e \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    python -m src.main &
SERVER_PID=$!
sleep 30
python -m bench.poisson_client --rate 500 --duration 60
kill -SIGUSR1 $SERVER_PID  # stop nsys capture
```

```bash
# Hottest kernel under load
ncu -o profiles/ncu_hot \
    --launch-skip 5000 --launch-count 1 \
    --set full \
    python -m src.main bench-mode --workload sharegpt_mixed
```

## Appendix B — Recommended Prometheus alerts

```yaml
# infra/alerts.yaml
groups:
- name: inference_slo
  rules:
  - alert: TtftSloBurning
    expr: histogram_quantile(0.99, sum by (le) (rate(inference_ttft_seconds_bucket[5m]))) > 0.1
    for: 2m
    annotations:
      summary: "TTFT P99 > 100 ms for 2 min"
  - alert: GoodputBelow95
    expr: rate(inference_requests_total{status="ok"}[5m])
        / rate(inference_requests_total[5m]) < 0.95
    for: 5m
  - alert: KvCacheNearlyFull
    expr: kv_cache_blocks_free < 50
    for: 1m
  - alert: GpuOverheating
    expr: gpu_temperature_celsius > 80
    for: 30s
```

## Appendix C — Common failure modes and fixes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| TTFT spikes under burst | Scheduler over-admits prefill | Lower `max_prefill_tokens_per_iter`; raise `low_water_mark` |
| TPOT P99 climbs steadily | KV cache thrashing; COW frequent | Increase prefix cache size; raise `block_size` |
| OOM after N minutes | Block leak on cancellation | Audit cancel path; add refcount sanity check periodic task |
| GPU idle while queue full | Admission too conservative | Lower `low_water_mark`; raise `max_decode_batch_size` |
| vLLM beats us by 2x | Our paged attention is the slow path | Verify FA paged kernel is actually called (nsys trace); ensure block_table is int32 |
| Decode CUDA Graph wrong output | Captured stale shape | Re-capture per shape bucket; assert input shape matches captured shape |
| Prefix cache hit rate ~0% | Hash includes random per-request tokens | Hash only the prefix BEFORE the user's per-request part |
| TPOT measured wrong | Including tokenizer decode time | Measure inter-token gap from server side, not client side |
| Grafana panel empty | Metric registered twice or wrong name | Check Prometheus `/metrics` endpoint manually |
| Container size > 4 GB | CUDA dev image instead of runtime | Switch base to `cuda:12.4-runtime-ubuntu22.04` in final stage |
