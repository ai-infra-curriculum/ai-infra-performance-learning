# Requirements — Production Model Optimization Pipeline

This document is the contract between the project specification and the
candidate's implementation. Every "MUST" is a gate. Every "SHOULD" is graded
in the rubric.

## 1. Scope

### 1.1 In scope

- Single-GPU model optimization for two reference models:
  - **ResNet-50** (vision), ImageNet-1k weights from `torchvision`.
  - **BERT-base-uncased** (NLP), HuggingFace weights, MNLI fine-tuned head.
- Optimization techniques:
  - FP16 (mixed precision and pure half).
  - INT8 PTQ with multiple calibrators (entropy, percentile, MSE).
  - INT8 QAT.
  - Structured channel pruning with iterative fine-tuning.
  - Knowledge distillation (teacher-student).
  - ONNX export.
  - TensorRT engine build with explicit precision and timing-cache reuse.
- A benchmark runner that produces statistically defensible numbers.
- Nsight Systems / Nsight Compute integration.
- A signed manifest of all artifacts.

### 1.2 Out of scope

- Multi-GPU training.
- LLM-scale models (>1B params).
- Custom CUDA kernels (covered in Project 2).
- Serving / batching infrastructure (covered in Project 3).
- INT4 / AWQ / FP8 — stretch goals only.

## 2. Functional requirements

### FR-1: Hydra-driven CLI

- MUST expose a CLI entry point `optimize` that takes a Hydra config and
  produces every artifact for one variant in one invocation.
- MUST support composition: `optimize model=resnet50 variant=int8_qat
  hardware=a100_80gb` produces one output set.
- MUST fail fast on missing required fields with a clear message.

### FR-2: Baseline runner

- MUST load both reference models from a pinned source (versioned).
- MUST evaluate each on its canonical validation set (IN-1k val, MNLI dev).
- MUST report top-1, top-5 (vision) or matched/mismatched (MNLI).
- MUST emit baseline latency distribution (P50/P90/P99/P99.9) and peak
  memory for batch sizes 1, 8, and 32.

### FR-3: FP16 path

- MUST support `model.half()` and `torch.autocast` paths separately.
- MUST verify no silent FP32 fallback by parsing a Nsight Systems trace.
- MUST report accuracy on the canonical validation set.

### FR-4: INT8 PTQ

- MUST implement at least 3 calibrators: entropy, percentile (99.9th),
  MSE-minimization.
- MUST support per-tensor AND per-channel quantization for weights.
- MUST persist calibration ranges to `calibration_<variant>.json`.
- MUST run a layer sensitivity sweep (skip each layer in turn, measure
  delta) and emit `sensitivity_<variant>.csv`.

### FR-5: INT8 QAT

- MUST use FX-graph mode quantization (`prepare_qat_fx`).
- MUST support custom `qconfig_mapping` (per-module overrides).
- MUST fine-tune for at least 3 epochs with a documented LR schedule.
- MUST handle BatchNorm folding correctly (no double-folding bug).

### FR-6: Structured pruning

- MUST implement L2-norm-based channel pruning.
- MUST be dependency-aware: when conv K is pruned, the next conv's input
  channels and any BN-affine parameters are also pruned.
- MUST support iterative pruning (configurable schedule).
- MUST emit a layer-by-layer report of channels removed and accuracy
  recovery per round.

### FR-7: Knowledge distillation

- MUST implement classical Hinton-style KL distillation with temperature.
- MUST support an intermediate-layer hint loss (FitNets) for at least 1
  layer pair, with a configurable projection module if dimensions differ.
- MUST report the alpha sweep (KD loss vs hard-label loss weighting).

### FR-8: TensorRT engine builder

- MUST use TensorRT 10.x Python API.
- MUST support `kFP16` and `kINT8` precision flags.
- MUST honor per-layer precision overrides loaded from YAML.
- MUST use an `IInt8EntropyCalibrator2` wired to the persisted calibration
  ranges from FR-4.
- MUST persist and reuse a timing cache to keep rebuilds under 60s.
- MUST emit engines for at least batch sizes 1, 8, 32 (or one engine with
  optimization profiles covering that range).

### FR-9: Benchmark runner

- MUST use **CUDA events**, not Python `time.perf_counter`, for timing.
- MUST do >= 50 warmup iterations and >= 500 measured iterations.
- MUST report P50, P90, P99, P99.9, mean, std-dev.
- MUST fail the run if std-dev / P50 > 5% (clocks aren't locked, or
  thermal throttling is active).
- MUST verify GPU clocks are locked (`nvidia-smi -lgc`).
- MUST capture energy via NVML where available.
- MUST run each variant against a fixed correctness oracle and reject if
  output diverges beyond the variant's allowed tolerance.

### FR-10: Profiling integration

- MUST emit at least one Nsight Systems `.nsys-rep` per variant.
- MUST emit at least one Nsight Compute `.ncu-rep` for the hottest kernel
  per variant (use `--launch-skip` and `--launch-count` to avoid 100GB
  reports).
- MUST produce a roofline plot per variant.

### FR-11: Manifest

- MUST emit `compression_manifest.yaml` with sha256 of every artifact,
  the source git SHA, the CUDA / cuDNN / TRT / torch versions, and the
  hardware identifier (`nvidia-smi -L` output).

### FR-12: Reproducibility

- MUST be invokable as a single command from a clean checkout
  (`make all` inside the project Dockerfile).
- MUST seed Python, NumPy, and PyTorch and document any non-deterministic
  cuDNN algorithms it relies on.

## 3. Performance requirements (hard gates)

These are repeated from the README and are evaluated on
**NVIDIA A100 80GB SXM** unless otherwise noted.

| ID | Metric | Target | Notes |
|----|--------|--------|-------|
| PR-1 | ResNet-50 bs=1 latency speedup vs FP32 | >= 3.0x | Final TRT engine |
| PR-2 | BERT-base bs=1 latency speedup vs FP32 | >= 3.0x | Final TRT engine |
| PR-3 | ResNet-50 bs=32 throughput speedup | >= 3.5x | Images/sec |
| PR-4 | ResNet-50 ImageNet top-1 drop | <= 2.0 pp | vs FP32 baseline |
| PR-5 | BERT-base MNLI-m accuracy drop | <= 2.0 pp | vs FP32 baseline |
| PR-6 | Model size on disk | <= 0.25x | Bytes of weight payload |
| PR-7 | Pipeline E2E runtime (full sweep) | <= 90 min | Wall clock on A100 |
| PR-8 | Benchmark std-dev / P50 | <= 5% | At measurement time |
| PR-9 | Memory-bandwidth util on hottest kernel | >= 70% | Soft gate (rubric) |
| PR-10 | Build determinism | bit-exact engine across two consecutive builds with cached timings | Soft gate |

## 4. Non-functional requirements

### NFR-1: Code structure

- File size <= 800 LOC.
- Function size <= 50 LOC (excluding well-named long match statements).
- Type hints on every public function.
- `mypy --strict` passes on the `src/` tree.
- `ruff` and `black` pass.

### NFR-2: Testing

- >= 80% statement coverage on `src/`.
- Unit tests for: calibration math, pruning dependency tracker, QAT
  qconfig builder, ONNX export shape inference.
- Integration test that runs the full pipeline on a 2-layer toy model in
  under 30 seconds (CI-friendly).
- Smoke test that exercises the CLI on each variant for 5 iterations.

### NFR-3: Logging and observability

- Structured logs (JSON) to stderr in production mode; pretty logs in dev.
- Every benchmark run emits a JSON line with all measured numbers; the
  Markdown report is a render of those JSON lines.

### NFR-4: Containerization

- Dockerfile based on `nvcr.io/nvidia/pytorch:24.05-py3` or later.
- Image MUST build offline from the repository (no surprise network pulls
  during model load — pre-cache weights into the image).

### NFR-5: Documentation

- Every public function has a one-line docstring + an `Args:` and
  `Returns:` block.
- Architecture decisions captured as ADRs in `docs/adr/`.

### NFR-6: Security

- No secrets in code, configs, or env files.
- The Hugging Face token (if used for gated models) MUST be loaded from
  an env var with a clear error if missing.

## 5. Constraints

- **Framework versions are pinned.** Do not silently bump.
- **Hardware budget**: one GPU. Do not rely on multi-GPU; the pipeline
  must still be useful on a single A10G.
- **Time budget**: the full sweep must complete in <= 90 minutes on A100.
  If a stretch experiment (INT4, FP8) takes longer, it MUST be gated by
  an explicit `--include-stretch` flag.
- **Closed-loop**: every accuracy gate is checked at the end of every
  variant. A regressed variant is moved to `engines/quarantine/` with a
  diagnostic log, not silently emitted.

## 6. Assumptions

- The candidate has access to the canonical ImageNet-1k validation set or
  the WebDataset shard subset used for sensitivity sweeps.
- The candidate has access to the MNLI dev set (HuggingFace `glue` config).
- The candidate is running on Linux with NVIDIA driver >= 545.
- The GPU is in a system the candidate can put in MIG mode if desired.

## 7. Dependencies (external)

| Component | Version | Reason |
|-----------|---------|--------|
| CUDA Toolkit | 12.4+ | TRT 10 compatibility |
| cuDNN | 9.x | PyTorch 2.3 path |
| PyTorch | 2.3+ | FX quantization stable |
| TensorRT | 10.0+ | New builder API, INT8 caching |
| ONNX | 1.16+ | opset 17 needed |
| onnxruntime-gpu | 1.18+ | Cross-check inference |
| HuggingFace transformers | 4.41+ | BERT-base load |
| Nsight Systems | 2024.2+ | Trace format |
| Nsight Compute | 2024.2+ | Roofline metrics |
| pynvml | latest | Energy telemetry |

## 8. Acceptance test sketch

Given a clean checkout on an A100 80GB host:

```
git clone <repo>
cd project-01-model-optimization
docker build -t opt-pipe .
docker run --gpus all -v $PWD:/work opt-pipe make all
```

Acceptance is granted if:

1. Exit code is 0.
2. `engines/` contains FP16, INT8 PTQ, INT8 QAT, pruned, distilled, TRT
   engines for ResNet-50 and BERT-base.
3. `reports/benchmark_summary.md` exists and every PR-* hard gate in
   Section 3 is met.
4. `compression_manifest.yaml` validates against its schema.
5. The `make verify` target re-runs benchmarks and produces numbers
   within 5% of those in the manifest.

## 9. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Clock unlocking causes flaky bench | High | High | Hard gate PR-8; abort with diagnostic if unlocked |
| QAT NaNs from missed BN folding | Medium | High | Unit test for BN folding; assert finite weights post-fold |
| TRT INT8 calibration drift across runs | Medium | Medium | Persist calibration cache; gate accuracy at build time |
| ONNX export breaks for dynamic shapes | Medium | High | Test export at bs=1, 8, 32 explicitly |
| TRT timing cache invalidation surprise | Low | Medium | Print cache hit/miss summary at build time |
| Distillation collapses (mode collapse) | Low | Medium | Track per-class accuracy; fail if any class drops below 50% relative recall |

## 10. Glossary

- **PTQ**: Post-Training Quantization. No retraining.
- **QAT**: Quantization-Aware Training. Fake-quant ops in training graph.
- **KD**: Knowledge Distillation. Train smaller model to mimic larger one.
- **TRT**: NVIDIA TensorRT.
- **Calibrator**: Algorithm that picks quantization ranges from observed
  activations.
- **Roofline**: Performance model relating arithmetic intensity to
  achievable throughput.
- **Timing cache**: TRT's persistent cache of tactic timings.
- **Optimization profile**: TRT's mechanism for handling dynamic shapes.
