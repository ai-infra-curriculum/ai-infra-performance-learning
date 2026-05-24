# Rubric — Production Model Optimization Pipeline

This rubric defines how submissions are scored. Reviewers apply each
dimension independently and report a 1-5 level. **Pass** requires all
hard performance gates from `requirements.md` Section 3 AND >= 3/5 in
every dimension. **Distinction** requires >= 4/5 in at least 6 of 8
dimensions.

Hard gates fail the project regardless of rubric score:

- PR-1..PR-7 hard targets missed.
- Any accuracy regression beyond the documented tolerance not declared
  in `compression_manifest.yaml` as quarantined.
- `make all` fails on a clean checkout inside the provided Docker image.

## 1. Scoring scale

| Level | Meaning |
|-------|---------|
| 1 | Missing or fundamentally broken. |
| 2 | Partial / hand-wavy / works only in the happy path. |
| 3 | Meets requirements, defensible. (Pass bar.) |
| 4 | Exceeds requirements with clear engineering judgment. |
| 5 | Reference-quality; the work would be merged into a production repo. |

## 2. Dimensions

### D1. Correctness (weight: highest)

Accuracy gates per variant respected; no silent regressions.

| Level | Evidence |
|-------|----------|
| 1 | Variants exceed allowed accuracy drop with no quarantine. |
| 2 | One variant ships outside spec; partial quarantine handling. |
| 3 | All variants hit accuracy gates; quarantine flow works. |
| 4 | Sensitivity sweep used to recover layers; per-layer overrides documented with measured deltas. |
| 5 | Calibrator comparison (entropy vs percentile vs MSE) presented with picks justified per model; layer-precision overrides have ablation numbers. |

### D2. Throughput (bs=32)

ResNet-50 images/sec speedup vs FP32 baseline.

| Level | Throughput speedup |
|-------|---------------------|
| 1 | < 1.5x |
| 2 | 1.5-2.5x |
| 3 | 2.5-3.5x (meets PR-3) |
| 4 | 3.5-4.5x |
| 5 | > 4.5x with breakdown of fusion / precision / layout contributions |

### D3. Latency (bs=1)

P99 latency speedup on the worst of the two models.

| Level | Latency speedup |
|-------|------------------|
| 1 | < 1.5x |
| 2 | 1.5-2.5x |
| 3 | 3.0x (meets PR-1/PR-2) |
| 4 | 3.5x with CUDA Graphs ablation |
| 5 | > 4x with CUDA Graphs + tactic-cache reuse + cold/warm comparison |

### D4. Memory and footprint

Peak GPU memory + model size on disk.

| Level | Evidence |
|-------|----------|
| 1 | Model size > 0.5x baseline; peak mem regressed. |
| 2 | Model size 0.25-0.5x baseline. |
| 3 | Model size <= 0.25x; peak mem documented (PR-6). |
| 4 | Per-variant memory chart in report; activation memory tracked. |
| 5 | Working-set chart from Nsight Compute Memory Workload Analysis; activation recompute trade-off discussed. |

### D5. Code quality

Type hints, modularity, no dead code, lint clean.

| Level | Evidence |
|-------|----------|
| 1 | Notebook-only or untyped. |
| 2 | Some structure, `mypy --strict` fails. |
| 3 | Module layout matches `architecture.md`; `mypy --strict`, `ruff`, `black` pass. |
| 4 | Public API documented; ADRs for non-obvious decisions. |
| 5 | Reviewer can extend with a new stage in < 30 minutes following the `Stage` protocol. |

### D6. Reproducibility

`make all` works; deterministic seeds; pinned versions.

| Level | Evidence |
|-------|----------|
| 1 | Cannot reproduce on a clean checkout. |
| 2 | Reproduces with manual steps. |
| 3 | `make all` inside Docker reproduces all artifacts (PR-12). |
| 4 | Two builds with identical inputs produce bit-identical TRT engines (modulo timestamps) when timing cache is warm. |
| 5 | Manifest includes `build_inputs_sha256`; reproducibility test runs in CI. |

### D7. Profiling depth

Roofline + Nsight Compute + defensible analysis.

| Level | Evidence |
|-------|----------|
| 1 | No profiling artifacts. |
| 2 | Nsight Systems trace only; no kernel-level analysis. |
| 3 | One Nsight Compute report per variant; one roofline plot per variant; hottest kernel classified as memory-bound or compute-bound. |
| 4 | Memory bandwidth utilization >= 70% on the hottest kernel of the final engine (PR-9 soft gate met); arithmetic intensity computed from measured FLOPs and bytes, not just SOL%. |
| 5 | Roofline shows the kernel walking up the ridge after each compression stage; commentary explains *why* each stage moved it. |

### D8. Engineering judgment

Right tool for the job; clean trade-offs; honest negative results.

| Level | Evidence |
|-------|----------|
| 1 | Cargo-culted defaults; no trade-off discussion. |
| 2 | One technique applied uniformly; no comparison. |
| 3 | PTQ vs QAT chosen per model with stated reason; pruning schedule justified. |
| 4 | Negative results reported (e.g. "MSE calibrator lost 0.4pp on BERT, dropped"); per-layer overrides explained. |
| 5 | Stretch path (FP8 / AWQ / CUDA Graphs) shipped with measured numbers and clear "when to use" guidance. |

## 3. Bonus rubric (distinction only)

These do not count toward Pass but contribute to Distinction:

- **B1**: FP8 (NVIDIA TransformerEngine) path on H100 with measured speedup.
- **B2**: AWQ 4-bit ablation on BERT-base with accuracy and speedup numbers.
- **B3**: CUDA Graphs capture on bs=1 path with cold/warm latency comparison.
- **B4**: 2:4 structured sparsity ablation on Ampere/Hopper.
- **B5**: Streamlit dashboard reading `benchmark_*.json`.

## 4. Anti-patterns (auto-deduction)

- **Hidden FP32 fallback in "FP16" path**: -1 level on D1 and D7.
- **Reporting Python `time.perf_counter` instead of CUDA events**: -1 on D2/D3.
- **No warmup or warmup < 50 iterations**: -1 on D2/D3.
- **Calibration data leaks from train into eval**: -2 on D1 (correctness violation).
- **Pruning without dependency awareness leading to shape mismatch at inference**: -2 on D1 and D5.
- **`make all` requires manual intervention**: -2 on D6.
- **No quarantine flow; failed variants silently dropped**: -1 on D1 and D5.

## 5. Reviewer checklist

```
[ ] Hard gates PR-1..PR-7 met? (else fail)
[ ] make all reproduces inside Docker? (else fail)
[ ] D1 Correctness:        _/5
[ ] D2 Throughput:         _/5
[ ] D3 Latency:            _/5
[ ] D4 Memory:             _/5
[ ] D5 Code quality:       _/5
[ ] D6 Reproducibility:    _/5
[ ] D7 Profiling depth:    _/5
[ ] D8 Engineering judgment: _/5

Total: __/40
Pass:        all >= 3 and hard gates met
Distinction: >= 6 dimensions at 4+
```

## 6. How to defend your score

For each dimension, the reviewer expects ONE artifact + ONE sentence:

| Dimension | Artifact | Sentence |
|-----------|----------|----------|
| D1 | `reports/accuracy_summary.md` | "Variant X lost Y pp because Z; recovered by overriding layer L." |
| D2 | `reports/benchmark_summary.md` (bs=32 row) | "Throughput won W from precision, V from fusion, U from NHWC layout." |
| D3 | `reports/benchmark_summary.md` (bs=1 row) | "P99 dropped from A ms to B ms; CUDA Graphs added C ms more." |
| D4 | `reports/memory_breakdown.md` | "Peak mem went from M GB to N GB; activation footprint dominates." |
| D5 | `src/` tree | "New variants are 1 file in `pipeline/` and 1 YAML in `configs/variant/`." |
| D6 | `make all` log | "Re-run produced identical sha256 for every engine." |
| D7 | `reports/roofline_*.png` + `profiles/*.ncu-rep` | "Kernel moved from memory-bound at AI=2 to compute-bound at AI=20." |
| D8 | `docs/adr/*.md` | "Picked QAT for BERT FFN, PTQ for ResNet stem; here is the data." |
