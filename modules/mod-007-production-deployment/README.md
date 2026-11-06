# Module 7 — Production Deployment

> **Track:** AI/ML Performance Engineer
> **Time budget:** ~25 hours (2–3 weeks part-time)
> **Module status:** first build, 2026-11-06

Module 06 taught a model how to span many GPUs and how a fleet of
replicas scales behind a load balancer. Module 07 is the step where
that fleet stops being an engineering demo and starts being a service
somebody pages on. The boundary between the two is thin and easy to
underestimate: a model that benchmarks well on a developer workstation
can still miss SLOs the first time it meets real traffic because the
*framework*, the *release process*, or the *cost structure* around the
model was never deliberately chosen. This module is the three decisions
that live between a working model and a reliable product.

Three concrete choices structure the module. First, **pick the serving
framework** — vLLM/TGI for autoregressive LLM generation, Triton or
TorchServe for high-QPS vision and embedding traffic, FastAPI only when
there is no better fit; the wrong choice here foregoes 5–10× throughput
on continuous batching alone (vLLM paper, [2309.06180](https://arxiv.org/abs/2309.06180)).
Second, **pick the release pattern** — rolling for low-risk re-trains,
blue-green for schema-breaking changes, canary with auto-revert gates
for risky model swaps, shadow (0% real traffic, mirrored requests) for
brand-new architectures; see the [Argo Rollouts canary docs](https://argo-rollouts.readthedocs.io/en/stable/features/canary/)
for the mechanics and the gate configuration. Third, **attribute and
compress cost** — spot capacity for preemptible batch work, reserved
instances for the always-on baseline, right-sizing from telemetry, and
tiered routing that sends cheap requests to cheap models; the levers
compound to roughly 5–10× lower $/request versus a naive "one model,
one instance type, always-on-demand" baseline.

The posture this module trains is **pick the framework first, then
everything else lines up**. The HPA, the autoscaler, the deployment
pattern, the observability stack, and the cost model are all downstream
of that one choice, and backing out of the wrong framework after the
rest of the stack is already built is the most expensive refactor on
the roadmap. Everything Module 06 taught about autoscaling on queue
depth, routing on prefix affinity, and absorbing cold-starts still
applies — this module adds the release-safety and $/request scaffolding
around it.

## Learning objectives

By the end of this module, you will be able to:

1. **Choose** a serving framework from the production-grade set — vLLM,
   TGI, Triton Inference Server, TorchServe, BentoML, Ray Serve — for a
   given workload (autoregressive LLM, vision classifier, embedding
   model, composed multi-model pipeline), and justify the choice with a
   measured p50/p99 latency and throughput number rather than a doc
   quote. Sources: [vLLM docs](https://docs.vllm.ai/), the
   [TGI README](https://github.com/huggingface/text-generation-inference),
   [Triton Inference Server](https://github.com/triton-inference-server/server),
   [TorchServe](https://pytorch.org/serve/).
2. **Explain**, in one minute, why continuous batching plus
   PagedAttention delivers ~5–10× throughput on chat-style LLM workloads
   where naive request-level batching stalls — short requests no longer
   wait behind long ones, and KV-cache memory is paged instead of
   reserved at the max sequence length. Source: vLLM paper
   ([2309.06180](https://arxiv.org/abs/2309.06180)).
3. **Match** a deployment pattern to a change's risk profile: rolling
   for same-architecture retrains, blue-green for schema breaks,
   canary with an auto-revert gate for risky model swaps, shadow
   (traffic mirrored, responses discarded) for architecture changes.
   Source: [Argo Rollouts progressive-delivery docs](https://argo-rollouts.readthedocs.io/en/stable/).
4. **Configure** an Argo Rollouts canary with an auto-revert analysis
   template on a measurable signal (5xx rate, p95 latency, or a
   model-quality delta via A/B shadow) and demonstrate the gate firing
   on a deliberately-broken candidate before the rollout finishes.
   Source: [Argo Rollouts analysis templates](https://argo-rollouts.readthedocs.io/en/stable/features/analysis/).
5. **Place** workloads on the correct capacity tier — spot for
   checkpoint-able batch, reserved for always-on inference baseline,
   on-demand for burstable overflow and anything on the control plane
   path — and quantify the $/run saving on a batch job moved to spot
   with a `PodDisruptionBudget` and graceful checkpoint on
   preemption. Source: [Karpenter spot docs](https://karpenter.sh/docs/concepts/nodepools/).
6. **Design** a multi-tier routing policy that sends simple requests
   to a small/cheap model (e.g. a local 7B) and escalates to a larger
   model only when a confidence or classifier signal demands it, and
   report the cost and quality delta against an always-large baseline.
7. **Ship** a model end-to-end from "trained" to "in production" with
   the six production essentials: a signed container image, a Helm
   chart parameterised for dev/staging/prod, an HPA on a GPU-aware
   custom metric (not CPU), SLO + burn-rate alerts, cost tags on every
   resource, and a runbook that an on-call engineer can read at 2 a.m.

Every exercise targets one or more of these; the quiz checks the
underlying concepts.

## Prerequisites

You should be comfortable with:

- **Module 04** — the vLLM serving stack, continuous batching,
  PagedAttention, and the serving-side intuition that makes Lecture 01
  click. If you cannot name what `vllm:num_requests_waiting` measures
  or why PagedAttention improves KV-cache utilisation, re-read Module
  04 Chapter 02 before starting this one.
- **Module 05** — quantization, LoRA, and the composition rules. The
  cost-lever table in Chapter 03 treats AWQ int4 and speculative
  decoding as discount factors on the $/request baseline; without
  Module 05 the arithmetic does not translate.
- **Module 06** — HPA on GPU-aware metrics, replica-fleet
  load-balancing, and cold-start mitigations. Exercise 05 of this
  module ships an HPA + SLO + runbook stack on top of the fleet
  patterns Module 06 built; the two compose rather than duplicate.
- **Kubernetes basics plus Argo Rollouts and Karpenter** — Exercise 02
  runs an Argo Rollouts canary with an auto-revert analysis template,
  and Exercise 03 moves a batch workload to a Karpenter-managed spot
  pool with a `PodDisruptionBudget`. A local `kind` or `minikube`
  cluster is enough to install the operators; the actual spot
  preemption and canary gating require a cloud cluster (or a scripted
  preemption test on a local cluster) to be meaningful.
- **A metrics stack** — Prometheus + Grafana, plus the Prometheus
  Adapter for custom-metric HPAs. Argo Rollouts' analysis templates
  query Prometheus by default, so the two share the same signal
  source. See the [Prometheus best-practices
  guide](https://prometheus.io/docs/practices/) for the metric-naming
  and histogram conventions the exercises assume.
- **A load generator** — `locust`, `k6`, or vLLM's
  [`benchmark_serving.py`](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py).
  Every framework-choice decision in this module is scored with a load
  test; "vLLM is faster" without a measured p99 is not an acceptable
  answer.

If you do not have the hardware or the cloud budget for every
exercise, Exercises 01, 04, and 05 can be downscaled to a 7B model on a
single consumer GPU; Exercise 02 and Exercise 03 need a Kubernetes
cluster (local is fine for the manifests, cloud is better for the
real preemption and canary signals).

## Module structure

The lecture is split across three chapter files in the
[`lecture-notes/`](lecture-notes/) directory. Read in order:

| Chapter | What's in it |
|---|---|
| [`lecture-notes/01-serving-frameworks.md`](lecture-notes/01-serving-frameworks.md) | The eight frameworks you will see in production — FastAPI + custom, TorchServe, Triton Inference Server, TensorFlow Serving, vLLM, TGI, BentoML, Ray Serve — and a decision tree keyed to model type and traffic shape. The anti-patterns to avoid: vLLM on classification, FastAPI alone for high-QPS LLM traffic (no continuous batching), one framework per model (fragmented ops). |
| [`lecture-notes/02-deployment-patterns.md`](lecture-notes/02-deployment-patterns.md) | Rolling, blue-green, canary, and shadow deployments mapped to risk profile; Argo Rollouts' auto-revert gates on 5xx rate, p95 latency, or a model-quality delta; when to reach for each pattern and when it is overkill. |
| [`lecture-notes/03-cost-optimization.md`](lecture-notes/03-cost-optimization.md) | The seven cost levers — spot for batch, reserved commitments, right-sizing from telemetry, quantization (Module 05), speculative decoding (Module 04), continuous batching (Module 04), and tiered per-request routing — with rough savings bands and the risk each lever carries. How the levers compound to a 5–10× $/request reduction versus a naive baseline. |

| Exercise | What it covers |
|---|---|
| [`exercises/exercise-01-pick-a-framework.md`](exercises/exercise-01-pick-a-framework.md) | For three workloads (chat LLM, vision classifier, embedding model) shortlist two serving frameworks each, run a realistic load test with `locust`/`k6`/`benchmark_serving.py`, and record p50/p99 latency, throughput, and peak GPU memory. Decide a winner per workload and defend the pick with the decisive metric in `FRAMEWORK_DECISIONS.md`. |
| [`exercises/exercise-02-canary-deployment.md`](exercises/exercise-02-canary-deployment.md) | Configure an Argo Rollouts canary for a vLLM deployment with a success-rate auto-gate (analysis template querying Prometheus). Deploy a deliberately-broken candidate and verify the gate reverts the rollout before it reaches 100% traffic. |
| [`exercises/exercise-03-spot-with-resilience.md`](exercises/exercise-03-spot-with-resilience.md) | Move a batch workload (training or large offline inference) to a Karpenter-managed spot pool. Add a `PodDisruptionBudget`, graceful checkpointing on preemption, and auto-recovery. Report the $/run saving versus the on-demand baseline. |
| [`exercises/exercise-04-multi-tier-routing.md`](exercises/exercise-04-multi-tier-routing.md) | Build a request router that classifies difficulty and routes small → medium → large models (e.g. local 7B → mid-sized hosted → frontier). Use confidence-based escalation; measure cost and quality against an always-large baseline. |
| [`exercises/exercise-05-end-to-end-deploy.md`](exercises/exercise-05-end-to-end-deploy.md) | Take a model from "trained" to "in production" with all six essentials: signed container image, Helm chart parameterised for dev/staging/prod, HPA on a custom GPU-aware metric, SLO + burn-rate alerts, cost-tagged resources, and an on-call runbook. |

Plus:

| File | What's in it |
|---|---|
| [`quiz.md`](quiz.md) / [`quiz-answers.md`](quiz-answers.md) | 10-question multiple-choice check on the learning objectives. 70% to pass. |
| [`resources.md`](resources.md) | Primary sources — vLLM, TGI, Triton, TorchServe, BentoML, Ray Serve, Argo Rollouts, Karpenter, plus the companion engineer-solutions modules. |

Companion repos:
[`engineer-solutions/mod-106 ex-08`](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-106)
holds fully-implemented YAML manifests for all four deployment
strategies (rolling, blue-green, canary, shadow), and
[`engineer-solutions/mod-110`](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-110-llm-infrastructure)
holds a multi-tenant LLM serving platform (ex-07) and a cost-routing
reference (ex-08) that mirror Exercises 02 and 04 in
production-shaped detail. Useful after a first pass at the exercises
here; avoid them before.

<!-- needs-research: confirm the exact engineer-solutions path shape; mod-104, mod-106, mod-108, mod-109, mod-110 are referenced throughout the module but the public layout of that repo should be cross-checked before linking from course-facing material. -->

## How to work through this module

1. Read the three chapter files in `lecture-notes/` in order. Plan for
   roughly 3–4 hours of reading plus small code-alongs. Chapter 01 is
   the shortest but the most consequential — the framework choice
   constrains every exercise after it, so slow down on the decision
   tree and the anti-patterns.
2. Do the five exercises in order. Exercise 01 is the cheapest to run
   and the one that makes the framework decision tree concrete.
   Exercise 02 (canary) wants a Kubernetes cluster with Argo Rollouts
   installed; Exercise 03 (spot) wants Karpenter and a cloud account
   with a spot quota; Exercises 04 and 05 are independent but assume
   the Prometheus + Grafana stack from Module 06 is already running.
3. Benchmark every framework decision. "vLLM is faster" is not a
   framework pick; the acceptance criterion for Exercise 01 is a
   measured p50/p99 and throughput number per candidate. Save the
   load-generator output alongside the `FRAMEWORK_DECISIONS.md`
   write-up — one without the other is not a result.
4. Prove the canary gate before you trust it. The point of Exercise
   02 is not that Argo Rollouts promotes a healthy candidate; it is
   that the auto-revert fires on a broken one. Deploy a deliberately
   broken version (a model that returns 5xx or an empty string on
   known prompts) and verify the rollback happens before the canary
   percentage advances.
5. Measure cost, do not estimate it. Exercise 03 wants a real $/run
   number before and after the spot migration; cloud-provider cost
   explorers surface this with a 24-hour lag, so plan the exercise
   over at least two days and compare on-demand vs spot cost from the
   bill, not from the on-screen hourly rate.
6. Attribute cost per model, not per cluster. Exercise 05's
   cost-tagging requirement is what makes the $/request number in
   Exercise 04 defensible; without per-model tags, a router that
   claims to save 50% is claiming savings against an unknown
   denominator.
7. Take the quiz **without** scrolling to `quiz-answers.md`. If you
   miss more than three questions, reread the chapter that covers
   them before moving on to Module 08.

## Assessment criteria for this module

You are ready to move on when you can, on a blank page:

- Match four workloads (autoregressive LLM, high-QPS vision
  classifier, high-QPS embedding model, multi-step composed pipeline)
  to the correct serving framework and name one anti-pattern for each
  (vLLM on classification wastes continuous batching; Triton on
  autoregressive LLM lacks PagedAttention; FastAPI alone on a
  high-QPS LLM has no continuous batching; one framework per model
  fragments ops across the fleet).
- State the one-line claim for continuous batching plus PagedAttention
  (short requests no longer wait behind long ones; KV cache is paged
  rather than reserved at max sequence length) and the rough
  speedup band on chat-style workloads (~5–10× versus naive padded
  batching, per the vLLM paper).
- Match four deployment patterns to four risk profiles: rolling for
  same-architecture retrains, blue-green for schema-breaking changes,
  canary with an auto-revert gate for risky model swaps, shadow (0%
  real traffic, mirrored requests) for brand-new architectures.
- Sketch an Argo Rollouts canary with an analysis template that
  auto-reverts on a Prometheus query (e.g. success-rate < 0.99 over a
  5-minute window). Explain which signals belong on the gate (error
  rate, p95 latency, model-quality delta via shadow A/B) and which do
  not (image-size delta, replica count, time of day).
- Place three workloads on the right capacity tier: batch training
  that can checkpoint → spot (with a `PodDisruptionBudget` and
  graceful checkpoint on preemption); always-on inference baseline →
  reserved; burstable overflow → on-demand. Explain in one sentence
  why the control-plane (etcd, ingress load balancers, scheduler) is
  never on spot.
- Enumerate the seven cost levers (spot for batch, reserved
  commitments, right-sizing from telemetry, quantization, speculative
  decoding, continuous batching, tiered per-request routing) with a
  rough savings band for each, and explain why compounding them lands
  on 5–10× lower $/request versus a naive baseline rather than on the
  sum of their individual savings.
- Draw the six-piece production-readiness checklist (signed container
  image, parameterised Helm chart, HPA on a GPU-aware custom metric,
  SLO + burn-rate alerts, cost tags on every resource, on-call
  runbook) and explain why skipping any one of them makes the other
  five less useful — an SLO without a runbook pages on-call without
  telling them what to do; cost tags without a Helm chart cannot be
  rolled forward to the next deployment.

Module 08 picks up here and moves from *running a reliable production
service* to *the frontier techniques that make it faster still* —
PagedAttention internals, speculative decoding, Flash Decoding, MoE
routing, and the research-to-production loop that keeps the stack
current. The frameworks and release patterns from this module are the
substrate those advanced techniques land in: a speculative-decoding
prototype that cannot be canaried with auto-revert is a speculative-
decoding prototype that never reaches production, and the measurement
discipline from Module 03 is still the thing that tells you whether a
new paper's claim survives contact with your workload.
