# Job Requirements — AI Infrastructure Performance Engineer

- **Role owner:** `ai-infra-performance-learning` (level 35)
- **Research window:** 2026-04-06 → 2026-07-06 (90 days)
- **Postings analyzed:** 34
- **Sources:** company career pages (NVIDIA, Anthropic, OpenAI, Meta, Google, Tesla, GM, Perplexity, xAI, Together AI, Cerebras, Scale AI, Fireworks, Nuance Labs, Baseten, Zoox, Waabi, Graphcore, Microsoft MAI) plus Greenhouse / Lever / Ashby / Menlo VC / General Catalyst / BuiltIn / Workday mirrors.

## Continuity summary

Every requirement mentioned in ≥30% of sampled postings maps cleanly onto an
existing module or exercise. **No new modules, exercises, or projects are
proposed this cycle** — see `.aicg/curriculum-plan-delta.json`. Emerging
trends (SGLang, NVIDIA Dynamo, Blackwell/GB200 naming, disaggregated
prefill/decode, Rust hot-path serving) are captured here for freshness edits
to existing content and are not treated as gaps.

## Requirement coverage (grouped by owner)

Requirements are listed by primary curriculum owner. Percentages are share
of the 34 sampled postings mentioning the requirement explicitly.

### Owned here — Module 1: GPU Fundamentals (`modules/mod-001-gpu-fundamentals/`)

| Requirement | Freq | Coverage |
|---|---|---|
| GPU memory hierarchy / bandwidth / occupancy | 65% | Lessons 1.3, 1.6; exercises 02–04 |
| GPU architecture (Hopper / Blackwell / H100 / B200 / GB200) | 41% | Lesson 1.2 architecture evolution; freshness edit needed to name Blackwell/GB200 |
| Roofline / arithmetic intensity | 15% | Lesson 1.6, exercise 04-roofline-analysis |

**Evidence sample:**
- NVIDIA — *GPU Performance Engineer, Neural Reconstruction* — [builtinsf.com](https://www.builtinsf.com/job/gpu-performance-engineer-neural-reconstruction/9512338) — 2026-07-06 — "*deep understanding of GPU architecture and memory hierarchy*"
- Together AI — *LLM Inference Frameworks & Optimization Engineer* — [greenhouse.io](https://job-boards.greenhouse.io/togetherai/jobs/4687884007) — 2026-07-06 — lists *"H100, H200, B200, GB200"*
- Anthropic — *Performance Engineer, GPU* — [menlovc.com](https://jobs.menlovc.com/companies/anthropic/jobs/69674569-performance-engineer-gpu) — 2026-03-08 — "*roofline models, GPU utilization, occupancy analysis*"

### Owned here — Module 2: CUDA Programming (`modules/mod-002-cuda-programming/`)

| Requirement | Freq | Coverage |
|---|---|---|
| CUDA kernel programming | 76% | Lessons 2.1–2.7; exercises 01–05 |
| C++/CUDA extensions / custom operators | 44% | Lesson 2.6, exercise 05-pytorch-extension |
| CUTLASS / CuTe DSL | 18% | Not yet named; addressable via existing kernel-writing exercises (freshness edit) |
| Triton (kernel DSL) | 35% | Owned in Module 4 exercise 05-write-triton-kernel |

**Evidence sample:**
- xAI — *MTS, CUDA/GPU Kernel* — [greenhouse.io](https://job-boards.greenhouse.io/xai/jobs/4427873007) — 2026-07-06 — "*utilizing CuTe/CUTLASS*"
- Perplexity — *AI Inference Engineer (MTS)* — [ashbyhq.com](https://jobs.ashbyhq.com/perplexity/8a976851-9bef-4b07-8d36-567fa9540aef) — 2026-07-06 — "*Rust, Python, CUDA, and CuTe DSL*"
- NVIDIA — *Senior Software Engineer, CUTLASS Performance* — [jobs.nvidia.com](https://jobs.nvidia.com/careers/job/893395379556) — 2026-07-06 — CUTLASS performance work

### Owned here — Module 3: Performance Profiling (`modules/mod-003-performance-profiling/`)

| Requirement | Freq | Coverage |
|---|---|---|
| Profiling (Nsight Compute / Systems / PyTorch profiler / NVTX) | 53% | Lessons 3.1–3.3; exercises 01–03 |
| Memory profiling / snapshots | 18% | Exercise 04-memory-snapshot |
| Observability (Prometheus / Grafana / DCGM) for GPU fleet | 18% | Owned by Module 7 (Lesson 7.4). DCGM naming is a freshness edit. |

**Evidence sample:**
- GM — *Senior AI/ML Capacity & Performance Engineer* — [gm.com](https://search-careers.gm.com/en/jobs/jr-202607049/senior-ai-ml-capacity-and-performance-engineer/) — 2026-06-17 — "*DCGM, nvidia-smi, Grafana*"
- NVIDIA — *AI Inference Performance Engineer* — [jobs.nvidia.com](https://jobs.nvidia.com/careers/job/893393953033) — 2026-07-06 — "*profile kernels with Nsight Compute / Systems*"

### Owned here — Module 4: Transformer Optimization (`modules/mod-004-transformer-optimization/`)

| Requirement | Freq | Coverage |
|---|---|---|
| Transformer / attention internals | 59% | Lessons 4.1–4.4; whole module |
| Flash Attention (any version) | 21% | Lessons 4.3–4.4; exercise 01-flash-attention |
| torch.compile / graph capture | 18% | Exercise 02-torch-compile |
| vLLM prefix caching | (subset of serving-framework 62%) | Exercise 03-vllm-prefix-caching |
| Speculative decoding | 29% | Exercise 04-speculative-decoding; Lesson 8.2 |
| Triton kernel authoring | 35% | Exercise 05-write-triton-kernel |
| Kernel fusion (RoPE, LayerNorm, GELU) | 24% | Lessons 4.5–4.7 |

**Evidence sample:**
- Anthropic — *Performance Engineer, GPU* — [menlovc.com](https://jobs.menlovc.com/companies/anthropic/jobs/69674569-performance-engineer-gpu) — 2026-03-08 — "*Flash Attention, kernel fusion, INT8/FP8 quantization*"
- Together AI — *LLM Inference Frameworks & Optimization Engineer* — [greenhouse.io](https://job-boards.greenhouse.io/togetherai/jobs/4687884007) — 2026-07-06 — "*speculative decoding*"

### Owned here — Module 5: Model Compression (`modules/mod-005-model-compression/`)

| Requirement | Freq | Coverage |
|---|---|---|
| Quantization (INT8 / INT4 / FP8 / AWQ / GPTQ / mixed precision) | 65% | Lessons 5.1–5.4; exercises 01-awq-quantize, 02-int8-static; FP8 in Module 8 exercise 05 |
| TensorRT | 50% | Lesson 5.7 |
| TensorRT-LLM specifically | (subset of serving-framework 62%) | Applies TensorRT + Module 6/7 techniques; SGLang/Dynamo/TRT-LLM naming is a freshness edit to `mod-007/exercises/exercise-01-pick-a-framework.md` |
| Structured sparsity (2:4) | 12% | Exercise 03-2-4-sparsity |
| LoRA / distillation | 18% | Exercises 04-lora-finetune, 05-distillation |

**Evidence sample:**
- Nuance Labs — *MTS Model Optimization & Inference* — [greenhouse.io](https://job-boards.greenhouse.io/nuancelabs/jobs/4277592009) — 2026-07-06 — "*INT8, INT4, GPTQ, AWQ*"
- NVIDIA — *AI Inference Performance Engineer* — [jobs.nvidia.com](https://jobs.nvidia.com/careers/job/893393953033) — 2026-07-06 — quantization as day-1 responsibility

### Owned here — Module 6: Distributed Inference (`modules/mod-006-distributed-inference/`)

| Requirement | Freq | Coverage |
|---|---|---|
| Tensor parallel | 41% | Lesson 6.2; exercise 01-tensor-parallel |
| Pipeline parallel | 41% | Exercise 02-pipeline-parallel |
| NCCL / RDMA / RoCE collective communication | 35% | Lesson 6.3; Module 8 exercise 03-nccl-tests |
| KV cache / PagedAttention / prefix caching | 44% | Module 4 exercise 03 + Module 8 lesson 8.1 |
| Continuous batching | 26% | Module 7 lesson 7.2 |
| MoE / expert parallelism | 18% | Module 8 lesson 8.3 (MoE optimization) |
| Prefix-aware routing | (implicit in serving stack) | Exercise 04-prefix-aware-routing |
| Cold-start mitigation | (implicit in serving stack) | Exercise 05-cold-start-mitigation |
| Disaggregated prefill/decode | 18% | Not explicitly covered but is an incremental extension of KV cache + prefix-aware routing; under 30% threshold, no addition proposed <!-- needs-research: track next cycle if trend persists --> |

**Evidence sample:**
- Together AI — *LLM Inference Frameworks & Optimization Engineer* — [greenhouse.io](https://job-boards.greenhouse.io/togetherai/jobs/4687884007) — 2026-07-06 — "*KV cache systems like Mooncake and PagedAttention*", "*Mixture of Experts (MoE) parallelism*"
- Anthropic — *Performance Engineer, GPU* — [menlovc.com](https://jobs.menlovc.com/companies/anthropic/jobs/69674569-performance-engineer-gpu) — 2026-03-08 — "*designing multi-node communication strategies*"

### Owned here — Module 7: Production Deployment (`modules/mod-007-production-deployment/`)

| Requirement | Freq | Coverage |
|---|---|---|
| Serving-framework selection (vLLM / TGI / TensorRT-LLM / Triton IS / SGLang / Dynamo) | 62% | Exercise 01-pick-a-framework already benchmarks vLLM/TGI/TensorRT-LLM/Triton/TorchServe/Ray Serve. Adding SGLang and NVIDIA Dynamo to the candidate list is a freshness edit — not a new exercise. |
| Continuous / dynamic batching | 26% | Lesson 7.2 |
| Load balancing / routing / rate limiting | 24% | Lesson 7.3; Module 6 exercise 04-prefix-aware-routing |
| Cost per inference / token | 18% | Lesson 7.5 |
| Prometheus / Grafana monitoring | 18% | Lesson 7.4 |
| Canary deployment | (implicit) | Exercise 02-canary-deployment |
| Autoscaling / capacity / spot resilience | 29% | Exercise 03-spot-with-resilience; Module 6 exercise 03-custom-hpa-metric |
| Multi-tier routing | (implicit) | Exercise 04-multi-tier-routing |
| Shadow traffic / automated baseline comparison | 18% | Under 30% — folded into exercise 02-canary-deployment as freshness edit |

**Evidence sample:**
- Nuance Labs — [greenhouse.io](https://job-boards.greenhouse.io/nuancelabs/jobs/4277592009) — "*proven proficiency with inference serving frameworks like vLLM, SGLang, and TensorRT-LLM*"
- NVIDIA — *AI Inference Performance Engineer* — [jobs.nvidia.com](https://jobs.nvidia.com/careers/job/893393953033) — "*work directly within TensorRT-LLM, SGLang, and vLLM*"
- Cerebras — *Staff Inference ML Runtime Engineer* — [greenhouse.io](https://job-boards.greenhouse.io/cerebrassystems/jobs/7523546003) — "*familiarity with LLM serving frameworks such as vLLM, SGLang, and TensorRT-LLM*"

### Owned here — Module 8: Advanced Topics (`modules/mod-008-advanced-topics/`)

| Requirement | Freq | Coverage |
|---|---|---|
| CUDA graphs | (implicit in low-overhead serving) | Exercise 01-cuda-graphs |
| Stream overlap | (implicit) | Exercise 02-stream-overlap |
| NCCL tests | 35% | Exercise 03-nccl-tests |
| MIG partitioning | (implicit) | Exercise 04-mig-partition |
| FP8 training | 65% quantization umbrella | Exercise 05-fp8-training |
| PagedAttention | 44% KV cache umbrella | Lesson 8.1 |
| Flash Decoding / INT4 / sparse attention | 21% umbrella | Lesson 8.3 |

**Evidence sample:**
- Anthropic — *Performance Engineer, GPU* — 2026-03-08 — "*INT8/FP8 quantization*"
- NVIDIA — *Senior Software Engineer, CUTLASS Performance* — 2026-07-06 — kernel-level FP8/FP4

## Linked to other roles (not owned here)

For requirements a Performance Engineer JD lists but that a lower-level role
already covers in depth, link to the owner rather than duplicating.

| Requirement | Freq | Primary owner | Where to point students |
|---|---|---|---|
| Kubernetes / container orchestration | 44% | `ai-infra-junior-engineer-learning` (level 10) and `ai-infra-engineer-learning` (level 20) | Prerequisite; touched here in Module 7 for GPU-specific pod/HPA concerns |
| Cloud (AWS/GCP/Azure) + Terraform | 35% | `ai-infra-engineer-learning` (level 20) | Prerequisite |
| PyTorch / JAX framework internals (framework side, not kernel side) | 59% | Kernel side is owned here (Mod 2/4). Broader PyTorch training-loop internals are owned by `ai-infra-ml-platform-learning` (level 30) |
| Rust / Go for services (non-hot-path) | 41% | `ai-infra-engineer-learning` (level 20) | Prerequisite for anyone whose runtime lives in Rust; the performance role assumes the language is picked up on the job. |

## External resources (out-of-scope items, ≥15% and <30%, tracked for freshness)

These do not clear the "propose new content" bar this cycle but are worth
pointing students to:

- **SGLang** — [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang). Fold into `mod-007/exercises/exercise-01-pick-a-framework.md` candidate list.
- **NVIDIA Dynamo** — [github.com/ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo). Same as above; also relevant to Module 6 (disaggregated).
- **Disaggregated prefill/decode** — Perplexity, vLLM V1, TensorRT-LLM all ship this. 18% frequency. <!-- needs-research: reassess if next cycle shows ≥30% -->
- **Multimodal / diffusion inference (Whisper, Parakeet, Kokoro, SNAC, Encodec)** — 24% frequency. Nuance Labs and Together AI Voice-AI roles emphasize it. <!-- needs-research: reassess next cycle -->
- **AMD MI300X / TPU v5–v6 / Trainium enablement (XLA, Triton, NeuronX)** — 26%. Would require a full portable-kernel track; below threshold. <!-- needs-research: track OpenAI AMD role trajectory -->
- **Correctness / numerical regression pipelines across accelerators** — 24%. Fold into `mod-003/exercises/exercise-05-end-to-end-optimization.md` as a freshness edit rather than a new exercise.
- **CUTLASS 3.x / CuTe DSL** — 18%. Reachable via Module 2 exercises with a curated reference; no new exercise required at this frequency.

## Posting evidence index

Complete posting list (employer, title, URL, date, location) lives in
`.aicg/job-requirements.json` under `postings[]`. See that file for the full
34-posting corpus.
