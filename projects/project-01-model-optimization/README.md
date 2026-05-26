# Project 1: Production Model Optimization Pipeline

> **Tier**: 4 (Capstone-grade)
> **Track**: AI/ML Performance Engineering
> **Estimated effort**: 60 hours
> **Complexity**: Advanced
> **Primary modules**: mod-003 (Profiling), mod-005 (Compression), mod-007 (Deployment)
> **Secondary modules**: mod-004 (Transformer Optimization), mod-008 (Advanced Topics)

## 1. Overview

Build a **production-grade model optimization pipeline** that ingests an
unmodified PyTorch baseline (a vision model and an encoder-only LM) and emits
a fleet of optimized variants — FP16, INT8 PTQ, INT8 QAT, structured-pruned,
KD-compressed, ONNX, and a final **TensorRT 10** engine — alongside a fully
automated benchmark harness that compares every variant on a fixed hardware
SKU with reproducible numbers.

The pipeline is not "a notebook that produces a smaller model." It is a CLI +
library + Hydra-configured workflow with:

- Deterministic reproducibility (pinned seeds, pinned CUDA, pinned cuDNN
  algorithms where measurable).
- Per-variant accuracy regression checks with **hard fail-stop** gates.
- A benchmark runner that captures latency distributions (P50/P90/P99/P99.9),
  throughput, peak memory, and energy (via NVML).
- Nsight Systems + Nsight Compute integration for the worst-case kernel from
  each variant.
- A signed manifest (`compression_manifest.yaml`) describing every artifact.

This is the project that proves the candidate can ship optimization work that
infra and SRE teams will actually deploy.

## 2. Performance targets (the gates you must hit)

These are hard targets. Numbers in this table are evaluated on a single
**NVIDIA A100 80GB SXM** (PCIe variant accepted with reported derate) at
batch size 1 and batch size 32, FP32 baseline reference, BERT-base + ResNet-50.

| Metric                                  | Baseline (FP32) | Target (final TRT engine) | Gate |
|-----------------------------------------|-----------------|---------------------------|------|
| Latency speedup (bs=1, ResNet-50)       | 1.0x            | **3.0x or better**        | Hard |
| Latency speedup (bs=1, BERT-base)       | 1.0x            | **3.0x or better**        | Hard |
| Throughput speedup (bs=32, ResNet-50)   | 1.0x            | **3.5x or better**        | Hard |
| Model size on disk                      | 1.0x            | **0.25x or smaller**      | Hard |
| Top-1 accuracy drop (ResNet-50, IN-1k val) | 0%           | **<= 2.0 percentage pts** | Hard |
| GLUE-MNLI accuracy drop (BERT-base)     | 0%              | **<= 2.0 percentage pts** | Hard |
| Memory bandwidth utilization (peak kernel) | n/a          | **>= 70%**                | Soft |
| Determinism: same-input bit-exact (FP16) | n/a            | bit-exact within 1 ULP    | Soft |
| Pipeline end-to-end runtime (full sweep) | n/a            | **<= 90 minutes on A100** | Soft |

Soft gates are scored in the rubric but do not block.

## 3. Learning outcomes

By the end of this project a learner will be able to:

1. Architect a multi-stage compression pipeline that composes quantization,
   pruning, and distillation without invalidating earlier stages.
2. Choose between **PTQ, QAT, AWQ, GPTQ, and SmoothQuant** based on the model
   structure and target hardware, and defend the choice with measurements.
3. Calibrate INT8 quantization using percentile / entropy / MSE-minimization
   calibrators and explain when each is correct.
4. Convert PyTorch → ONNX → TensorRT with explicit precision constraints,
   tactic selection, and timing cache reuse.
5. Build a benchmark harness that produces statistically defensible latency
   numbers (warmup, lockstep clock, CUDA event timing, CUDA Graphs replay).
6. Identify and triage common failure modes: activation overflow, calibration
   drift, dead-channel pruning, BatchNorm folding bugs, sparse FP16 NaNs.
7. Profile the final engine with Nsight Compute and explain where it sits on
   the roofline.

## 4. Prerequisites

### 4.1 Hardware (recommended)

- 1 x NVIDIA A100 40GB or 80GB (preferred), H100 acceptable, A10G/RTX 3090
  acceptable with reduced batch sizes.
- 128 GB system RAM, 1 TB NVMe.
- Linux x86_64 (Ubuntu 22.04 LTS recommended).

### 4.2 Software (pinned)

- CUDA Toolkit **12.4**+
- cuDNN **9.x**
- PyTorch **2.3**+ with CUDA 12.4 build
- TensorRT **10.0**+ (LLM-grade builder API)
- ONNX **1.16**+, ONNX Runtime GPU **1.18**+
- Python **3.10** or **3.11**
- NVIDIA Nsight Systems **2024.2**+, Nsight Compute **2024.2**+
- `pynvml`, `nvidia-smi`, `nvtop` for runtime telemetry
- Optional: NVIDIA TransformerEngine for FP8 ablation, `bitsandbytes` for
  INT4 sanity checks, `neural-compressor` for cross-vendor PTQ comparison

### 4.3 Knowledge

- Completion of mod-003 (Profiling) and mod-005 (Compression).
- Comfortable reading Nsight Compute `Memory Chart` and `Source View`.
- Basic understanding of ONNX opsets and how export-time tracing differs from
  scripting.

## 5. Deliverables

Per the project [`deliverables/README.md`](./deliverables/README.md), final
submission must include:

- `src/` — full Python library (importable, with type hints, no Jupyter-only
  code paths).
- `cli/optimize.py` — Hydra-driven CLI entry point.
- `cli/benchmark.py` — benchmark runner.
- `configs/` — Hydra YAMLs for every supported variant.
- `engines/` — built TRT engines + ONNX intermediates + timing caches.
- `reports/benchmark_*.json` — machine-readable benchmark output.
- `reports/benchmark_*.md` — human-readable summary with tables.
- `reports/roofline_*.png` — at least 1 roofline plot per variant.
- `profiles/*.nsys-rep` — Nsight Systems trace of the worst-case run.
- `profiles/*.ncu-rep` — Nsight Compute report of the hottest kernel per
  variant.
- `compression_manifest.yaml` — signed artifact manifest.
- `tests/` — pytest suite (unit + integration + smoke).
- `Dockerfile` — reproducible build environment.
- `Makefile` — single-command repro: `make all`.

## 6. Week-by-week breakdown

This is a **six-week** project at ~10 hours/week, or four weeks at 15
hours/week. The phases are sequential — do not start a phase before the
previous one's gate passes.

### Week 1 — Foundations and baseline (8-10h)

Set up the reproducible environment, capture baseline numbers, and build the
benchmarking skeleton. **Gate**: FP32 ResNet-50 and BERT-base baselines pass
accuracy and have a documented, reproducible latency distribution.

- Bring up Docker image with pinned CUDA/cuDNN/PyTorch/TRT.
- Write `BaselineRunner` that loads model + dataset and runs `K` warmup +
  `N` measured iterations using CUDA events, not Python time.
- Capture: P50, P90, P99, P99.9, std-dev. Reject if std-dev/P50 > 5%.
- Capture peak memory via `torch.cuda.max_memory_allocated`.
- Record GPU clock state (`nvidia-smi -q -d CLOCK`) — abort if not locked.

### Week 2 — PTQ and FP16 (10-12h)

Implement INT8 PTQ and FP16 paths. **Gate**: FP16 variant accuracy delta <
0.2%, INT8 PTQ variant accuracy delta < 1.5% (final TRT engine target is
2%, leaving headroom for downstream stages).

- FP16: `model.half()` for ResNet-50, autocast for BERT-base. Verify no
  hidden FP32 cast in attention by Nsight Systems trace inspection.
- INT8 PTQ: build calibration loader (256-1024 samples), implement entropy
  and percentile calibrators, run sensitivity sweep across all `Conv2d` and
  `Linear` layers.
- Persist calibration ranges per-tensor in JSON for downstream reuse.

### Week 3 — QAT and structured pruning (10-12h)

**Gate**: QAT variant within 0.5% of FP32 accuracy; pruned variant at 30%
channel sparsity within 1.0% of FP32.

- Wire QAT using `torch.ao.quantization.prepare_qat_fx` with custom
  `qconfig_mapping`. Train 3-5 epochs. Watch for BN-folding bugs.
- Structured pruning: implement L2-norm channel pruning with iterative
  schedule (10% → 20% → 30% over 3 rounds, with fine-tuning between).
- Implement **dependency-aware** pruning so that downstream conv input
  channels match upstream output channels.

### Week 4 — Knowledge distillation (8-10h)

**Gate**: Student model (ResNet-18 from ResNet-50 teacher; DistilBERT-style
from BERT-base) within 2.5% of teacher pre-quantization.

- Implement soft-target loss with temperature T=4.0, alpha sweep.
- Implement intermediate-layer feature distillation (FitNets-style) for at
  least one layer pair.
- Verify gradient flow with `torch.autograd.gradcheck` on a minimal example.

### Week 5 — TensorRT engine building (10-12h)

**Gate**: Final TRT engine hits hard performance targets in Section 2.

- ONNX export with `opset >= 17`, dynamic axes for batch dim.
- TRT builder config: `kFP16`, `kINT8` precision flags, `kSTRICT_TYPES`
  where needed, layer-precision overrides for accuracy-sensitive layers.
- Use `IInt8EntropyCalibrator2` with the calibration data persisted in
  week 2.
- Enable **timing cache** persistence so reproducible builds are fast.
- Build per-batch-size engines (1, 8, 32) or use optimization profiles.

### Week 6 — Benchmark, profile, report (8-10h)

**Gate**: All deliverables produced and `make all` runs end-to-end from a
clean checkout.

- Full benchmark sweep across all variants × batch sizes × seq lengths.
- Nsight Systems trace of each variant; Nsight Compute on the hottest
  kernel of the final TRT engine.
- Roofline plot using measured FLOPs and bytes; classify each kernel as
  memory-bound or compute-bound.
- Write `reports/benchmark_summary.md` with a single table that an SRE
  can read in 30 seconds.

## 7. Architecture pointer

See [`architecture.md`](./architecture.md) for the full component diagram,
data flow, and trade-off discussion (PTQ vs QAT, AWQ vs GPTQ, layer-wise
vs tensor-wise quantization, structured vs unstructured pruning).

## 8. Step-by-step build guide

See [`STEP_BY_STEP.md`](./STEP_BY_STEP.md). That document is the canonical
"do this exactly" path with code snippets, profiling commands, and the
full gotcha list.

## 9. Rubric summary

Full rubric in [`rubric.md`](./rubric.md). High-level dimensions:

1. **Correctness** (accuracy gates, no silent regressions)
2. **Throughput** (bs=32 numbers vs target)
3. **Latency** (P99 numbers vs target)
4. **Memory** (peak GPU memory, model size on disk)
5. **Code quality** (type hints, tests, modularity)
6. **Reproducibility** (single-command repro, deterministic seeds)
7. **Profiling depth** (roofline, Nsight reports, defensible analysis)
8. **Engineering judgment** (right tool for the job, clean trade-offs)

Pass = all hard performance gates met AND rubric score >= 3/5 in every
dimension. Distinction = rubric score >= 4/5 in at least 6 of 8.

## 10. Success criteria checklist

- [ ] FP32 baseline reproducible within 5% latency std-dev.
- [ ] FP16 variant: latency >= 1.5x speedup, accuracy drop <= 0.2 pp.
- [ ] INT8 PTQ variant: accuracy drop <= 1.5 pp.
- [ ] INT8 QAT variant: accuracy drop <= 0.5 pp.
- [ ] Pruned variant: 30% sparsity, accuracy drop <= 1.0 pp.
- [ ] Distilled student: within 2.5 pp of teacher.
- [ ] Final TRT engine: 3x bs=1 speedup, 3.5x bs=32, 75% smaller, <=2pp drop.
- [ ] Nsight Compute report attached for hottest kernel of final engine.
- [ ] Roofline plot attached.
- [ ] `compression_manifest.yaml` signed (sha256 per artifact).
- [ ] `make all` succeeds from a clean checkout inside the provided Docker
      image with no manual steps.

## 11. Related modules

| Module | Why it matters for this project |
|--------|---------------------------------|
| mod-003 Performance Profiling | All measurements depend on the profiling discipline established here. |
| mod-004 Transformer Optimization | BERT-base path uses attention bottleneck analysis from this module. |
| mod-005 Model Compression | Core techniques (PTQ, QAT, pruning, KD) are taught here in depth. |
| mod-007 Production Deployment | Output engines feed directly into Project 3's serving system. |
| mod-008 Advanced Topics | INT4 / AWQ / sub-byte ablation is an optional stretch path. |

## 12. Stretch goals (for distinction)

- Add **AWQ** (Activation-aware Weight Quantization) as a 4-bit alternative
  to GPTQ; compare accuracy and speedup.
- Add **SmoothQuant** for INT8 LLMs where activation outliers kill naive
  PTQ.
- Add **NVIDIA TransformerEngine FP8** path for H100 targets.
- Add **CUDA Graphs** capture of the final engine for further latency
  reduction at small batch sizes.
- Build a small Streamlit dashboard reading `benchmark_*.json` so a PM can
  see the wins without reading code.

## 13. Out of scope

- LLM serving / batching (that's Project 3).
- Custom CUDA kernel authoring (that's Project 2).
- Multi-GPU compression (single-GPU compression only here).
- Training from scratch (we always start from a pretrained checkpoint).

## 14. References

- NVIDIA TensorRT Developer Guide (v10.x)
- "A White Paper on Neural Network Quantization", Qualcomm AI Research, 2021
- Han et al., "Deep Compression", ICLR 2016
- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network", 2015
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
  Compression and Acceleration", 2023
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
  Quantization for Large Language Models", ICML 2023
- NVIDIA Nsight Compute and Nsight Systems user guides
