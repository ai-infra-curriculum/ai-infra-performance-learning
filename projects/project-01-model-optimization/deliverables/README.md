# Deliverables — Production Model Optimization Pipeline

This directory is the submission target. A reviewer should be able to
unzip a submission tarball into this directory, run `make verify` from
the project root, and reproduce every number in the report.

## 1. Required submission inventory

```
deliverables/
  src/                          # Importable Python package
  cli/                          # Hydra-driven entry points
    optimize.py
    benchmark.py
    report.py
  configs/                      # Hydra YAMLs (model / variant / hardware)
  engines/                      # Built TRT engines + ONNX + timing cache
    resnet50_fp32.onnx
    resnet50_fp16.engine
    resnet50_int8_ptq.engine
    resnet50_int8_qat.engine
    resnet50_pruned_int8.engine
    resnet50_distilled_int8.engine
    bert_base_fp32.onnx
    bert_base_fp16.engine
    bert_base_int8_ptq.engine
    bert_base_int8_qat.engine
    timing_cache.bin
    quarantine/                 # Variants that failed accuracy gates
  reports/
    benchmark_summary.md        # Single human-readable summary table
    benchmark_<variant>.json    # Machine-readable, one per variant
    accuracy_summary.md
    memory_breakdown.md
    sensitivity_<variant>.csv   # PTQ layer-skip sensitivity sweep
    pruning_report_<variant>.json
    roofline_<variant>.png      # One per variant (>= 1)
    roofline_<variant>.csv      # Underlying data for the PNG
    raw/                        # JSONL of every measurement
  profiles/
    nsys_<variant>.nsys-rep     # One Nsight Systems trace per variant
    ncu_<variant>.ncu-rep       # One Nsight Compute report per variant
  calibration/
    calibration_<variant>.json  # Per-tensor ranges
  compression_manifest.yaml     # Signed (sha256) artifact manifest
  Makefile                      # `make all`, `make verify`, `make clean`
  Dockerfile                    # Reproducible environment
  pyproject.toml
  tests/                        # pytest tree, >= 80% coverage on src/
  docs/
    adr/                        # ADRs for non-obvious decisions
  README_submission.md          # Submitter-written notes (deviations, etc.)
```

## 2. `compression_manifest.yaml` schema

Every artifact MUST be listed with sha256, size, source git SHA, and
per-variant accuracy + perf summary. Example:

```yaml
build:
  git_sha: 1a2b3c4d
  built_at_utc: 2026-04-01T12:00:00Z
  builder: candidate@hostname
  hardware: NVIDIA A100 80GB SXM
  driver: 545.29.06
  cuda: 12.4
  cudnn: 9.1.0
  tensorrt: 10.0.1
  torch: 2.3.1
  build_inputs_sha256: "abc...def"   # hash of (state_dict, calib data, config)

variants:
  - name: resnet50_int8_qat_trt
    artifact: engines/resnet50_int8_qat.engine
    sha256: "..."
    size_bytes: 27341823
    accuracy:
      metric: imagenet_top1
      baseline: 0.8014
      observed: 0.7912
      delta_pp: -1.02
      gate: pass            # <= 2.0 pp drop
    performance:
      hardware: a100_80gb
      batch_size: 1
      latency_p50_ms: 0.83
      latency_p99_ms: 0.91
      speedup_vs_fp32: 3.41
      gate: pass            # >= 3.0x
    profiling:
      nsys: profiles/nsys_resnet50_int8_qat.nsys-rep
      ncu: profiles/ncu_resnet50_int8_qat.ncu-rep
      roofline: reports/roofline_resnet50_int8_qat.png
      hottest_kernel: sm90_xmma_fprop_implicit_gemm_indexed_int8_..."
      mem_bw_util_pct: 78.2
      compute_util_pct: 62.5

quarantine:
  - name: bert_base_int4_awq
    reason: accuracy gate failed
    delta_pp: -3.4
    artifact: engines/quarantine/bert_base_int4_awq/
```

## 3. Reports — what each one must contain

### `benchmark_summary.md`

One table the SRE reads in 30 seconds:

```
| Model       | Variant          | bs | P50 (ms) | P99 (ms) | Throughput (img/s or seq/s) | Size (MB) | Acc delta (pp) | Gate |
```

### `accuracy_summary.md`

Per variant: baseline accuracy, observed accuracy, delta in percentage
points, calibrator used, layers with per-layer overrides.

### `memory_breakdown.md`

Per variant: peak GPU mem, weights mem, activations mem, workspace mem.
At least one chart.

### `roofline_<variant>.png`

X-axis: arithmetic intensity (FLOPs / byte). Y-axis: achieved
GFLOPs/sec. Hardware ridge plotted (A100: 19.5 TFLOPs FP16 ridge at
AI = 12.5 for HBM2e). At least 5 hottest kernels plotted.

## 4. Profiles — capture command reference

```bash
# Nsight Systems trace (one per variant; 5-10 second capture is enough)
nsys profile -o profiles/nsys_${VARIANT} \
  --trace=cuda,cudnn,cublas,nvtx \
  --capture-range=cudaProfilerApi \
  python -m cli.benchmark variant=${VARIANT} iters=200

# Nsight Compute report — hottest kernel only (avoid 100GB reports)
ncu -o profiles/ncu_${VARIANT} \
  --launch-skip 50 --launch-count 1 \
  --set full \
  --target-processes all \
  python -m cli.benchmark variant=${VARIANT} iters=60
```

## 5. Tests

```
tests/
  unit/
    test_calibrator_math.py
    test_pruning_dependency.py
    test_qat_qconfig.py
    test_bn_folding.py
    test_onnx_shape_inference.py
  integration/
    test_pipeline_toy_model.py   # 2-layer model, <30s
  smoke/
    test_cli_each_variant.py     # 5 iters per variant
```

Coverage must be >= 80% on `src/`. `pytest --cov=src --cov-report=term`
output goes into `reports/coverage.txt`.

## 6. Reproducibility check

```bash
make all          # Full pipeline
make verify       # Re-runs benchmarks, compares to manifest within 5%
```

`make verify` MUST pass on the same hardware within 5% of the numbers
recorded in `compression_manifest.yaml`. If the reviewer is on different
hardware, they use `make verify HARDWARE=a10g` and the comparison
window widens to 15%.

## 7. What NOT to submit

- Training checkpoints from intermediate KD or QAT epochs (only the
  final accepted weights).
- Raw datasets (link in `README_submission.md`).
- Editor cache, `__pycache__`, `.pytest_cache`, `wandb/`, `outputs/`.
- Anything > 1 GB without justification in `README_submission.md`.

## 8. Submission checklist

- [ ] All files in Section 1 present.
- [ ] `compression_manifest.yaml` validates against schema (Section 2).
- [ ] `make all` clean run completes in <= 90 minutes on A100.
- [ ] `make verify` numbers within 5% of manifest.
- [ ] No secrets committed.
- [ ] No data > 1 GB without justification.
- [ ] `README_submission.md` declares any deviations from spec.
