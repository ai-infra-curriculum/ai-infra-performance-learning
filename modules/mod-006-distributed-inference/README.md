# Module 6 — Distributed Inference

> **Track:** AI/ML Performance Engineer
> **Time budget:** ~30 hours (3 weeks part-time)
> **Module status:** first build, 2026-11-06

Module 05 made the model itself smaller — AWQ int4, 2:4 sparsity, LoRA —
and most 7B–13B models now fit on a single modern GPU. Module 06 picks up
where that story runs out: the models that *do not* fit on one GPU, the
traffic patterns that *do not* fit on one replica, and the serving stack
you have to build once you can no longer answer "add more GPU" with "give
me a bigger GPU." The content splits cleanly into two halves. The first
is how a single model is sharded across many GPUs — tensor parallelism,
pipeline parallelism, and the 3D combination used for frontier-scale
models. The second is how many copies of that model are run together —
autoscaling on a GPU-aware metric, load-balancing requests whose cost
varies by orders of magnitude, and absorbing the multi-minute cold start
of a large LLM replica without dropping traffic.

The thread running through both halves is **communication cost**. Every
multi-GPU trick in this module adds some form of cross-device traffic —
an all-reduce per transformer layer for tensor parallelism, activation
passing between stages for pipeline parallelism, a request hop to a
prefix-affinity replica for the routing layer — and the only way to tell
whether the trick is paying off is to measure that communication against
the compute it enables. NCCL, NVLink, InfiniBand, and the host-side
request router are all part of the critical path now; the Module 03
profiling discipline extends naturally to them (`nsys` with NCCL ranges,
queue-depth timelines, load-balancer access logs) but the units change
from kernels to messages.

The posture this module trains is **do not reach for parallelism until
measurement forces you to**. Tensor-paralleling a 7B model across two
GPUs when it already fits on one is slower and more expensive than
running two independent replicas. Pipeline-paralleling a model that
already fits inside one node is slower than tensor parallelism alone.
Autoscaling that fires on CPU utilization for a GPU-saturated workload
never fires at all. Every optimization here has a specific regime where
it wins and a larger regime where it just adds operational surface area;
the learning objectives below are all about recognising which regime you
are in before you turn the knob.

## Learning objectives

By the end of this module, you will be able to:

1. **Explain**, in one minute, how tensor parallelism shards each layer's
   weights across N GPUs and why the per-layer all-reduce traffic is the
   term that determines whether TP is worth it; name the hardware
   requirement that makes it practical in production (NVLink-class
   intra-node interconnect). Source: Megatron-LM paper
   ([1909.08053](https://arxiv.org/abs/1909.08053)) and the 3D-parallelism
   follow-up ([2104.04473](https://arxiv.org/abs/2104.04473)).
2. **Describe** pipeline parallelism as stage-partitioned inference with
   micro-batches, state the origin of the pipeline "bubble", and explain
   why PP wins on throughput at scale but loses on single-request
   latency. Source: GPipe paper
   ([1811.06965](https://arxiv.org/abs/1811.06965)).
3. **Compose** 3D parallelism — tensor parallel within a node, pipeline
   parallel across nodes, data parallel across replicas — for a given
   model size and cluster topology, and justify the split with reference
   to interconnect bandwidth (NVLink intra-node, InfiniBand/Ethernet
   inter-node).
4. **Serve** a model with vLLM's `--tensor-parallel-size` and
   `--pipeline-parallel-size` flags, verify that output tokens are
   numerically unchanged versus a single-GPU reference, and report TTFT,
   throughput, and p95 latency versus the single-GPU (or replicated)
   baseline. Source: vLLM paper
   ([2309.06180](https://arxiv.org/abs/2309.06180)) and the
   [distributed serving docs](https://docs.vllm.ai/en/latest/serving/distributed_serving.html).
5. **Configure** Kubernetes HPA (or KEDA) to scale vLLM replicas on a
   GPU-aware metric — `vllm:num_requests_waiting` or generated tokens per
   second via the Prometheus Adapter — rather than CPU utilization, and
   explain in one sentence why the default CPU-based HPA never fires on
   a GPU-saturated workload.
6. **Choose** a load-balancing strategy for LLM traffic (least-loaded,
   power-of-two-choices, or prefix-affinity via consistent hashing),
   justify the choice against the request-cost distribution of the
   workload, and state the regression mode of round-robin on
   long-tailed generation lengths.
7. **Plan** the cold-start path for a large-model replica — image pull,
   weight load into HBM, warmup — and apply mitigations (DaemonSet
   image pre-pull, pre-warmed idle replicas, load-balancer slow-start)
   so that a scale-up event does not spike user-visible p95 latency.

Every exercise targets one or more of these; the quiz checks the
underlying concepts.

## Prerequisites

You should be comfortable with:

- **Module 03** — `nsys`, `ncu`, and the measure → change → re-measure
  loop. Every distributed deployment in this module is scored with
  before/after numbers (TTFT, throughput, p95, queue depth); without
  those, "TP=4 feels faster" is not a passing answer. NCCL calls show
  up in `nsys` as their own ranges and are the thing you time when
  diagnosing a TP or PP configuration.
- **Module 04** — the vLLM serving stack, continuous batching, prefix
  caching, and the KV-cache size formula. The autoscaling and
  load-balancing chapters assume you already know what
  `vllm:num_requests_waiting` is counting and why prefix caching makes
  routing decisions matter.
- **Module 05** — quantization and LoRA. A 70B model in fp16 needs ~140
  GB of weights (two 80 GB GPUs minimum); the same model in AWQ int4
  needs ~35 GB (fits on one). The right first question before reaching
  for TP is "did I quantize yet?" — the two chapters compose, and
  compression often shrinks the parallelism problem out of existence.
- **Multi-GPU hardware access** — at minimum a 2-GPU node for Exercise
  01 and a 4-GPU-plus node (or two 4-GPU nodes with NVLink intra-node
  and 100 GbE / InfiniBand inter-node) for Exercise 02. If you do not
  have multi-node hardware, substitute a smaller target model and
  single-node TP — the measured shape of the TP vs replication tradeoff
  is the same, only the absolute numbers scale down.
- **Kubernetes basics** — pods, deployments, services, HPA. Exercises
  03–05 run on a Kubernetes cluster with Prometheus and the Prometheus
  Adapter (or KEDA) installed. A local `kind` / `minikube` cluster is
  enough for the HPA manifests; the actual scaling events require real
  GPU nodes.
- **A working CUDA toolchain + NCCL** — NCCL ships with recent CUDA
  toolkits and is the collective-communication backend for every TP
  and PP configuration here. You do not write NCCL calls by hand, but
  `NCCL_DEBUG=INFO` and NCCL's own topology-detection logs are the
  tools you reach for when a TP job hangs on startup.

If you do not have the right hardware locally, a short-lived multi-GPU
cloud instance (an 8×A100 or 8×H100 node) is cheaper than owning the
hardware for a module's worth of work; the exercises are measured in
minutes to low single-digit hours of GPU time, not days.

## Module structure

The lecture is split across three chapter files in the
[`lecture-notes/`](lecture-notes/) directory. Read in order:

| Chapter | What's in it |
|---|---|
| [`lecture-notes/01-tensor-pipeline-parallel.md`](lecture-notes/01-tensor-pipeline-parallel.md) | Tensor parallelism as per-layer weight sharding with an all-reduce per layer, and why the communication term scales with `batch × seq_len × hidden_dim`; pipeline parallelism as stage-partitioning with micro-batches and the bubble it introduces; 3D parallelism (TP intra-node + PP inter-node + DP across replicas); the replicate-vs-shard decision for models that already fit on one GPU. |
| [`lecture-notes/02-multi-replica-autoscaling.md`](lecture-notes/02-multi-replica-autoscaling.md) | Why CPU-based HPA never fires for GPU inference; the two GPU-aware autoscaling signals (queue depth via `vllm:num_requests_waiting`, generated tokens per second); KEDA on message-broker lag for batch workloads; cold-start arithmetic for a 70B replica (image pull, weight load, warmup); pre-pull / pre-warm / predictive-scale mitigations and the headroom-above-peak rule of thumb. |
| [`lecture-notes/03-load-balancing.md`](lecture-notes/03-load-balancing.md) | Why round-robin fails on LLM traffic (10-token and 2000-token requests count equally); least-loaded and power-of-two-choices routing; prefix-affinity via consistent hashing on the system-prompt hash to maximise prefix-cache hit rate; the common failure modes (dead-replica stickiness, slow-loris, cold-start thundering herd) and the LB-level mitigations for each. |

| Exercise | What it covers |
|---|---|
| [`exercises/exercise-01-tensor-parallel.md`](exercises/exercise-01-tensor-parallel.md) | Deploy Llama-13B with vLLM `--tensor-parallel-size 2` across two GPUs on one node. Compare to the TP=1 baseline (quantized to fit, or OOM-as-expected). Report TTFT, throughput, and p95 latency; identify the regime where TP wins and the regime where replication wins. |
| [`exercises/exercise-02-pipeline-parallel.md`](exercises/exercise-02-pipeline-parallel.md) | Serve a 70B model across two nodes with `--tensor-parallel-size 4 --pipeline-parallel-size 2` (TP intra-node over NVLink, PP inter-node over InfiniBand / 100 GbE). Compare to single-node TP=8 if available. Report where PP wins on throughput and where it loses on latency. |
| [`exercises/exercise-03-custom-hpa-metric.md`](exercises/exercise-03-custom-hpa-metric.md) | Configure a Kubernetes HPA to scale vLLM replicas on `vllm:num_requests_waiting` (queue depth) using Prometheus + Prometheus Adapter. Drive a load spike; watch the queue grow, replicas scale up, and the queue drain. Compare to a CPU-based HPA on the same workload (which will not fire). |
| [`exercises/exercise-04-prefix-aware-routing.md`](exercises/exercise-04-prefix-aware-routing.md) | Build a thin FastAPI router in front of N vLLM replicas. Use consistent hashing on the request's system-prompt hash to route the same prompt to the same replica, maximising prefix-cache hit rate. Measure cache-hit rate and end-to-end throughput against a round-robin baseline. |
| [`exercises/exercise-05-cold-start-mitigation.md`](exercises/exercise-05-cold-start-mitigation.md) | Measure cold-start time (pod created → first token served) for a Llama-70B replica on a fresh node. Apply the three standard mitigations — DaemonSet image pre-pull, one pre-warmed replica per node, slow-start routing on the load balancer — and re-measure end-to-end p95 latency on a scale-up event. |

Plus:

| File | What's in it |
|---|---|
| [`quiz.md`](quiz.md) / [`quiz-answers.md`](quiz-answers.md) | 10-question multiple-choice check on the learning objectives. 70% to pass. |
| [`resources.md`](resources.md) | Primary sources — the Megatron-LM TP paper, the GPipe pipeline-parallelism paper, the vLLM paper and distributed-serving docs, NCCL, the Prometheus Adapter, and KEDA. |

Companion repo:
[`engineer-solutions/mod-110`](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-110-llm-infrastructure)
holds a full vLLM deep dive (ex-03) and a multi-tenant serving scaffold
(ex-07) that mirror Exercises 01 and 04 in production-shaped detail.
Useful after a first pass at the exercises here; avoid it before.

## How to work through this module

1. Read the three chapter files in `lecture-notes/` in order. Plan for
   roughly 4–5 hours of reading plus small code-alongs. Chapter 01 is
   the longest and the most paper-dense — the all-reduce arithmetic
   and the pipeline-bubble intuition are both worth slowing down on,
   because every later capacity-planning conversation on a distributed
   inference team rests on them.
2. Do the five exercises in order. Exercise 01 (TP) is the cheapest to
   run and the one that makes the "replicate before you shard" rule
   concrete. Exercise 02 (PP) wants multi-node hardware and is where
   the inter-node bandwidth constraint shows up. Exercises 03–05 move
   from single-model sharding to the replica-fleet layer and run on
   Kubernetes; they are independent of each other but share the
   Prometheus + vLLM stack, so set that up once.
3. Profile every change. "I turned on TP=2 and it was faster" is not
   what this module is training — the goal is before/after TTFT,
   throughput, and p95 numbers for each exercise, plus the NCCL /
   queue-depth trace that explains them. Save the `nsys` outputs with
   NCCL ranges and the vLLM `/metrics` scrapes together with each
   write-up; one without the other is not a result.
4. Verify correctness before you celebrate speed. A TP or PP
   configuration must produce the same output distribution as the
   single-GPU reference on a fixed prompt + seed; if it does not, a
   collective is wired wrong or a rank is dropping tokens. Every
   exercise lists a correctness check — run it against a held-out
   prompt set before the benchmark numbers.
5. Benchmark honestly. Every measurement assumes warmup requests (the
   first request after a cold start is not the number you report), a
   steady-state load generator, and the median of several runs. For
   autoscaling and cold-start exercises, report both the steady-state
   p95 and the scale-up transient — a system that is fine at steady
   state can still spike on a traffic burst, and that spike is what
   on-call sees.
6. Take the quiz **without** scrolling to `quiz-answers.md`. If you
   miss more than three questions, reread the chapter that covers
   them before moving on to Module 07.

## Assessment criteria for this module

You are ready to move on when you can, on a blank page:

- State the tensor-parallelism one-line claim (each layer's weights
  are sharded across N GPUs; a per-layer all-reduce combines partial
  results) and the one hardware requirement that makes it practical
  (NVLink-class intra-node interconnect — inter-node TP over Ethernet
  or even InfiniBand is almost always a performance trap).
- Draw the pipeline-parallel bubble diagram — stages × micro-batches,
  idle slots at the start and end of the pipeline — and state the
  rule of thumb that enough micro-batches makes the bubble a
  single-digit percentage overhead, while too few makes it dominant.
- Sketch a 3D-parallelism layout for a 70B-class model on two 8-GPU
  NVLink nodes (TP=4 intra-node, PP=2 inter-node, DP=1 within the
  replica; multiple such replicas for data-parallel scaling), and
  explain in a sentence why TP goes inside the NVLink boundary and PP
  crosses it.
- Write a Kubernetes HPA snippet that scales on a custom Prometheus
  metric — `vllm:num_requests_waiting` with a target average of ~5 —
  and explain why this metric fires exactly when the user-visible
  latency starts to degrade, where a CPU-based HPA on the same
  deployment never fires at all.
- Name the three load-balancing strategies that work for LLM traffic
  (least-loaded, power-of-two-choices, prefix-affinity via consistent
  hashing on the system prompt) and the one that does not (round-robin
  on request count), with one sentence on why round-robin fails
  (request cost varies by orders of magnitude with generation length).
- Enumerate the three cold-start phases for a large-LLM replica
  (image pull, weight load into HBM, warmup) with rough numbers for a
  70B model (~140 GB image: minutes on a fresh node; 30–90 s weight
  load; tens of seconds of warmup), and the three mitigations that
  attack each phase (DaemonSet image pre-pull, pre-warmed idle
  replicas, LB slow-start to a new replica).
- Explain in one minute when **not** to distribute — a 7B model that
  fits on one GPU after quantization is faster and cheaper to serve
  as N independent replicas than TP-sharded across N GPUs, because
  the per-layer all-reduce is pure overhead when the model already
  fits.

Module 07 picks up here and moves from *making a model span many GPUs*
to *running the resulting service in production*: containerization,
serving-framework choice (vLLM vs TGI vs Triton), streaming APIs,
monitoring, and cost-per-token accounting. The distributed-inference
building blocks from this module are what Module 07 wires into a
production control plane — autoscaling policies become SLO-driven
instead of ad-hoc, load-balancing becomes part of the service-mesh
configuration, and the measurement discipline from Module 03 is still
the thing that tells you whether the resulting stack earns its
operational complexity.
