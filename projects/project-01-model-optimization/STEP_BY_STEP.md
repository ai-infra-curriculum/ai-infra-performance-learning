# Step-by-Step Build Guide — Production Model Optimization Pipeline

This is the canonical walkthrough. Do the steps in order; do not skip a
gate check. Each phase ends with a **gate** — if it fails, stop and fix
before continuing. Every code snippet here is illustrative; the real
implementation lives in `src/`.

The full build is six weeks at ~10h/week. Use this document as your
weekly checklist.

---

## Phase 0 — Environment, before week 1 (2-3h)

### 0.1 Verify hardware

```bash
nvidia-smi
nvidia-smi -L                       # confirm GPU SKU
nvidia-smi --query-gpu=name,driver_version,vbios_version --format=csv
ncu --version && nsys --version
```

Required: driver >= 545, CUDA >= 12.4, Nsight >= 2024.2.

### 0.2 Build the Docker image

```Dockerfile
FROM nvcr.io/nvidia/pytorch:24.05-py3
RUN pip install --no-cache-dir tensorrt==10.0.* onnx==1.16.* \
    onnxruntime-gpu==1.18.* transformers==4.41.* hydra-core==1.3.* \
    pynvml ruff black mypy pytest pytest-cov
WORKDIR /work
```

```bash
docker build -t opt-pipe:dev .
docker run --gpus all --rm -it -v $PWD:/work opt-pipe:dev nvidia-smi
```

### 0.3 Lock GPU clocks

```bash
sudo nvidia-smi --persistence-mode=1
sudo nvidia-smi --lock-gpu-clocks=1410,1410     # A100 base
sudo nvidia-smi --lock-memory-clocks=1593       # A100 HBM2e
```

**Gate 0**: `nvidia-smi -q -d CLOCK` shows clocks locked at the
configured values. If your environment does not allow clock locking,
note the deviation in `compression_manifest.yaml` so reviewers know.

### 0.4 Repo skeleton

```
project-01-model-optimization/
  README.md
  requirements.md
  architecture.md
  STEP_BY_STEP.md
  rubric.md
  Makefile
  Dockerfile
  pyproject.toml
  configs/
    model/
      resnet50.yaml
      bert_base.yaml
    variant/
      fp16.yaml
      int8_ptq.yaml
      int8_qat.yaml
      pruned.yaml
      distilled.yaml
      trt_final.yaml
    hardware/
      a100_80gb.yaml
      a10g.yaml
  src/
    pipeline/
      __init__.py
      loader.py
      calibrator.py
      fp16.py
      ptq.py
      qat.py
      pruning.py
      distillation.py
      onnx_exporter.py
      trt_builder.py
      validator.py
    bench/
      runner.py
      roofline.py
      nsys.py
      ncu.py
    cli/
      optimize.py
      benchmark.py
      report.py
  tests/
  deliverables/
```

---

## Phase 1 — Baseline and benchmark scaffolding (week 1, 8-10h)

### 1.1 Implement `ModelLoader`

```python
# src/pipeline/loader.py
from dataclasses import dataclass
from typing import Callable
import torch

@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    sample_input: torch.Tensor
    accuracy_oracle: Callable[[torch.nn.Module], "AccuracyResult"]
    canonical_name: str
    fp32_baseline_accuracy: float

def load_resnet50() -> ModelBundle:
    import torchvision.models as M
    model = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2).cuda()
    model.eval()
    sample = torch.randn(1, 3, 224, 224, device="cuda")
    # ... oracle wired to IN-1k val subset
    return ModelBundle(model, sample, ...)
```

### 1.2 Implement `BenchRunner` (the most important class in the project)

```python
# src/bench/runner.py
import torch
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PerfResult:
    p50_ms: float
    p90_ms: float
    p99_ms: float
    p999_ms: float
    mean_ms: float
    std_ms: float
    samples: list[float]

def bench_module(fn, sample_input, warmup: int = 50, iters: int = 500):
    # Warmup
    for _ in range(warmup):
        fn(sample_input)
    torch.cuda.synchronize()

    # Measured
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(sample_input)
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return PerfResult(
        p50_ms=float(np.percentile(times_ms, 50)),
        p90_ms=float(np.percentile(times_ms, 90)),
        p99_ms=float(np.percentile(times_ms, 99)),
        p999_ms=float(np.percentile(times_ms, 99.9)),
        mean_ms=float(times_ms.mean()),
        std_ms=float(times_ms.std()),
        samples=times_ms.tolist(),
    )
```

### 1.3 Guardrails

Before reporting, assert:

- `std_ms / p50_ms < 0.05` else `raise ClockStateError("clocks unlocked or throttling")`
- `nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader` returns `0x0`.

### 1.4 Baseline run

```bash
python -m cli.optimize model=resnet50 variant=baseline_fp32 hardware=a100_80gb
python -m cli.optimize model=bert_base variant=baseline_fp32 hardware=a100_80gb
```

### Gate 1

- [ ] Both baselines produce a `reports/benchmark_baseline_*.json`.
- [ ] Top-1 (ResNet) and MNLI-m (BERT) match published numbers within 0.1pp.
- [ ] `std_ms / p50_ms < 5%` on both.

### Gotchas

- **Python `time.perf_counter` is wrong here.** It measures CPU launch
  time, not GPU execution. CUDA events are mandatory.
- **First-time cuDNN benchmark** picks algorithms. Discard first 50
  iterations entirely or you'll measure heuristic overhead.
- **Different GPU IDs** with MIG enabled produce different perf. Pin
  with `CUDA_VISIBLE_DEVICES=0`.

---

## Phase 2 — FP16 and INT8 PTQ (week 2, 10-12h)

### 2.1 FP16

```python
# src/pipeline/fp16.py
def to_fp16_pure(bundle):
    m = bundle.model.half()
    # ResNet-50 final classifier: keep FP32 for numerical stability
    for name, child in m.named_children():
        if name == "fc":
            child.float()
    return bundle.replace(model=m, sample_input=bundle.sample_input.half())

def to_fp16_autocast(bundle):
    # Wrap forward in autocast at runtime
    ...
```

### 2.2 Verify no FP32 fallback

Run a short nsys trace:

```bash
nsys profile --trace=cuda,nvtx --stats=true -o profiles/fp16_resnet \
    python -m cli.optimize model=resnet50 variant=fp16_pure
nsys stats --format csv,column profiles/fp16_resnet.nsys-rep > profiles/fp16_resnet_stats.csv
grep -i "sgemm\|cudnn_fp32" profiles/fp16_resnet_stats.csv
```

If you see `sgemm` or `*_fp32_*` kernels in the hot path, you have a
silent fallback. Fix by:

- Casting inputs to `.half()` explicitly.
- Confirming no `.float()` calls leak into the model body.

### 2.3 INT8 PTQ scaffolding

```python
# src/pipeline/ptq.py
import torch.ao.quantization as Q

def ptq_calibrate(model, calib_loader, calibrator="entropy"):
    model.eval()
    qconfig = Q.get_default_qconfig("x86")
    # For TRT we don't actually use the PT INT8 kernels - we want
    # observers to capture ranges that we serialize to JSON for the
    # TRT builder.
    model = Q.prepare(model, inplace=False, qconfig_mapping=...)
    with torch.no_grad():
        for batch in calib_loader:
            model(batch.cuda())
    return extract_ranges(model)  # dict[str, (min, max)]
```

### 2.4 Sensitivity sweep

For each quantizable layer, create a variant where only that layer is
INT8 and the rest is FP32, measure accuracy delta, sort layers by their
contribution to accuracy loss.

Output `sensitivity_<variant>.csv`:

```
layer_name,baseline_acc,ablated_acc,delta_pp
layer1.0.conv1,76.13,76.10,-0.03
layer1.0.conv2,76.13,76.05,-0.08
...
layer4.2.conv3,76.13,75.42,-0.71
```

Use this to choose which layers stay FP16 vs go INT8 in the mixed-
precision TRT engine.

### Gate 2

- [ ] FP16 accuracy delta < 0.2pp on both models.
- [ ] FP16 nsys trace shows zero hot-path FP32 kernels.
- [ ] INT8 PTQ accuracy delta < 1.5pp on both models.
- [ ] `sensitivity_<variant>.csv` written and reviewed.

### Gotchas

- **BatchNorm in eval mode** still uses running stats, which means
  PTQ observers will see post-BN distributions. That's fine — but if you
  toggle eval/train you'll get different ranges.
- **HistogramObserver bin count**: too few bins underestimates tails;
  default 2048 is fine.
- **Class imbalance in calibration data**: if your 1024-sample subset
  misses rare classes, you'll bias quantization ranges. Stratify.

---

## Phase 3 — QAT and structured pruning (week 3, 10-12h)

### 3.1 QAT with FX graph mode

```python
# src/pipeline/qat.py
from torch.ao.quantization import (
    QConfigMapping, get_default_qat_qconfig
)
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

def to_qat(bundle, epochs=3, lr_scale=0.1):
    qconfig = get_default_qat_qconfig("x86")
    qmap = QConfigMapping().set_global(qconfig)
    # Per-module overrides for accuracy-sensitive layers
    qmap = qmap.set_module_name("fc", None)   # keep classifier FP32
    example_inputs = (bundle.sample_input.float(),)
    prepared = prepare_qat_fx(bundle.model.train(), qmap, example_inputs)
    # Fine-tune
    train_loop(prepared, epochs=epochs, lr=base_lr * lr_scale)
    return prepared
```

### 3.2 BN folding sanity check

Add a unit test:

```python
def test_bn_folding_preserves_output():
    m_before = build_toy_conv_bn()
    m_before.eval()
    x = torch.randn(2, 3, 8, 8)
    y_before = m_before(x)
    m_after = fold_bn(m_before)
    y_after = m_after(x)
    assert torch.allclose(y_before, y_after, atol=1e-5)
```

A failed BN fold is the most common QAT bug. Catch it in CI.

### 3.3 Structured pruning, dependency-aware

```python
# src/pipeline/pruning.py
import torch.fx

def build_dep_graph(model):
    gm = torch.fx.symbolic_trace(model)
    # Walk nodes, link conv -> bn -> next_conv with shared channel dim
    ...
    return DepGraph(nodes=..., edges=...)

def prune_channels(model, sparsity=0.3):
    g = build_dep_graph(model)
    for group in g.coupled_groups():
        scores = l2_norm_per_channel(group.shared_param)
        k = int(scores.numel() * sparsity)
        keep_idx = scores.topk(scores.numel() - k).indices.sort().values
        for layer in group.layers:
            slice_channels_in_place(layer, keep_idx, dim=group.dim_for(layer))
    return model
```

### 3.4 Iterative schedule

10% -> fine-tune 1 epoch -> 20% -> fine-tune 1 epoch -> 30% -> fine-tune
2 epochs. Track accuracy per round; if any round drops > 1pp from the
previous, stop and report.

### Gate 3

- [ ] QAT variant within 0.5pp of FP32 accuracy.
- [ ] Pruned variant at 30% sparsity within 1.0pp.
- [ ] `pruning_report.json` shows layer-by-layer channel removals and
      monotone accuracy recovery across fine-tuning rounds.

### Gotchas

- **NaN under fp16 during QAT fine-tune**: clip activations or fall back
  to BF16 for the training loop. Inference still goes through INT8.
- **Pruning the classifier head**: don't. Add it to the no-prune list.
- **FX trace failure on BERT**: HuggingFace BERT is not FX-friendly out
  of the box. Either (a) use the HuggingFace `BertModel` with
  `torch.fx.wrap` annotations on attention, or (b) fall back to
  module-name-based pruning for BERT.

---

## Phase 4 — Knowledge distillation (week 4, 8-10h)

### 4.1 Loss

```python
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, T=4.0):
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)

def total_loss(student_logits, teacher_logits, labels, alpha=0.7, T=4.0):
    soft = kd_loss(student_logits, teacher_logits, T)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * soft + (1 - alpha) * hard
```

### 4.2 Intermediate hint loss (FitNets-style)

```python
class HintLoss(torch.nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.proj = torch.nn.Linear(student_dim, teacher_dim, bias=False)
    def forward(self, s, t):
        return F.mse_loss(self.proj(s), t.detach())
```

Pick one student layer (e.g. mid-network feature map) and one teacher
layer with matching spatial dims (or insert a 1x1 conv to match).

### 4.3 Alpha sweep

Run with alpha in {0.3, 0.5, 0.7, 0.9}. Report a table.

### Gate 4

- [ ] Student model within 2.5pp of teacher pre-quantization.
- [ ] Alpha sweep table in `reports/distillation_alpha_sweep.md`.
- [ ] No mode collapse (every class has > 50% relative recall vs teacher).

---

## Phase 5 — TensorRT engine build (week 5, 10-12h)

### 5.1 ONNX export

```python
# src/pipeline/onnx_exporter.py
def export_onnx(model, sample, out_path, opset=17):
    dyn = {"input": {0: "batch"}, "output": {0: "batch"}}
    model.eval()
    torch.onnx.export(
        model, (sample,), out_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes=dyn, opset_version=opset, do_constant_folding=True,
    )
    import onnx
    m = onnx.load(out_path)
    onnx.checker.check_model(m)
    m = onnx.shape_inference.infer_shapes(m)
    onnx.save(m, out_path)
```

### 5.2 TRT builder

```python
# src/pipeline/trt_builder.py
import tensorrt as trt, os

def build(onnx_path, engine_path, fp16=True, int8=False,
          calibrator=None, timing_cache="engines/timing_cache.bin",
          profiles=((1, 8, 32),)):
    LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, LOGGER)
    with open(onnx_path, "rb") as f:
        assert parser.parse(f.read()), parser.get_error(0)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator

    # Optimization profile for dynamic batch
    for mn, opt, mx in profiles:
        prof = builder.create_optimization_profile()
        prof.set_shape("input", (mn, 3, 224, 224),
                       (opt, 3, 224, 224), (mx, 3, 224, 224))
        config.add_optimization_profile(prof)

    # Timing cache
    cache_blob = b""
    if os.path.exists(timing_cache):
        cache_blob = open(timing_cache, "rb").read()
    cache = config.create_timing_cache(cache_blob)
    config.set_timing_cache(cache, ignore_mismatch=False)

    engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(engine)
    with open(timing_cache, "wb") as f:
        f.write(memoryview(config.get_timing_cache().serialize()))
```

### 5.3 INT8 calibrator wired to phase-2 ranges

```python
class EntropyCalibratorFromJSON(trt.IInt8EntropyCalibrator2):
    def __init__(self, ranges_json, batches):
        super().__init__()
        self.batches = batches
        self.iter = iter(batches)
        self.device_buf = cuda.mem_alloc(batches[0].nbytes)
        self._cache = build_cache_blob_from_ranges(ranges_json)

    def get_batch_size(self):
        return self.batches[0].shape[0]

    def get_batch(self, names):
        try:
            b = next(self.iter)
        except StopIteration:
            return None
        cuda.memcpy_htod(self.device_buf, b)
        return [int(self.device_buf)]

    def read_calibration_cache(self):
        return self._cache

    def write_calibration_cache(self, cache):
        pass  # we always read from JSON
```

### 5.4 Build and verify

```bash
python -m cli.optimize model=resnet50 variant=trt_int8 hardware=a100_80gb
python -m cli.benchmark engine=engines/resnet50_trt_int8.engine
```

### Gate 5

- [ ] Engines built for all variants and at least batch sizes 1, 8, 32.
- [ ] Final TRT INT8 engine: latency speedup >= 3x at bs=1, throughput
      >= 3.5x at bs=32 (PR-1, PR-3).
- [ ] Accuracy gates (PR-4, PR-5) still pass.
- [ ] Timing cache present; second build under 60s.

### Gotchas

- **TRT silently dropping precision flags**: if your network has any
  layer the builder can't satisfy in INT8/FP16, it falls back to FP32
  unless you set `BuilderFlag.OBEY_PRECISION_CONSTRAINTS`. Then it errors
  loudly. Use that flag in CI.
- **Calibration cache mismatch**: TRT verifies the cache hash against
  the network signature. If you change the network and reuse the cache,
  you'll get a warning and a silent rebuild. Delete on graph change.
- **Workspace too small**: under 256MB you'll lose tactic options. Use
  >= 1GB for these models.

---

## Phase 6 — Bench, profile, report (week 6, 8-10h)

### 6.1 Full benchmark sweep

```bash
python -m cli.benchmark sweep=full
```

Should produce one row per (model, variant, batch_size). At minimum:

```
| model     | variant   | bs |  p50 |  p99 | speedup | mem_mb | size_mb |
|-----------|-----------|----|------|------|---------|--------|---------|
| resnet50  | fp32      |  1 | 4.20 | 4.40 |   1.00x |  142   |  102    |
| resnet50  | fp16      |  1 | 2.10 | 2.20 |   2.00x |  104   |   51    |
| resnet50  | int8_ptq  |  1 | 1.65 | 1.74 |   2.55x |   98   |   26    |
| resnet50  | trt_int8  |  1 | 1.20 | 1.27 |   3.50x |   84   |   26    |
| ...                                                                    |
```

### 6.2 Nsight Systems

```bash
nsys profile --trace=cuda,nvtx,osrt --stats=true \
    -o profiles/trt_int8_resnet50 \
    python -m cli.benchmark engine=engines/resnet50_trt_int8.engine \
        --warmup 5 --iters 20
```

Open the `.nsys-rep` in Nsight Systems UI to inspect kernel sequence,
launch overhead, and any CPU sync stalls.

### 6.3 Nsight Compute on hottest kernel

```bash
ncu --launch-skip 50 --launch-count 1 \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section SchedulerStats \
    --section Occupancy \
    --section SourceCounters \
    -o profiles/trt_int8_resnet50_hottest \
    python -m cli.benchmark engine=engines/resnet50_trt_int8.engine \
        --iters 100
```

Read out:

- `dram__throughput.avg.pct_of_peak_sustained_elapsed` (memory bandwidth %)
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` (compute %)
- `sm__warps_active.avg.pct_of_peak_sustained_active` (occupancy)

### 6.4 Roofline plot

```python
# src/bench/roofline.py
import matplotlib.pyplot as plt

def plot_roofline(kernels, peak_flops_tflops=312.0, peak_bw_tbps=2.039,
                  out_path="reports/roofline_resnet50_trt_int8.png"):
    # Hardware: A100 80GB peak INT8 312 TOPS, HBM2e 2.039 TB/s
    xs = [k.bytes / 1e9 for k in kernels]
    ys = [k.flops / k.time_s / 1e12 for k in kernels]
    ais = [k.flops / k.bytes for k in kernels]
    # Plot roofline...
```

### 6.5 Write the summary

`reports/benchmark_summary.md` MUST have:

1. The single table from 6.1.
2. A bullet list confirming each PR-* gate (pass/fail).
3. Pointers to Nsight reports.
4. A short paragraph per variant explaining the speedup source
   (precision, fusion, layout).

### 6.6 Sign the manifest

```python
# scripts/sign_manifest.py
import hashlib, yaml, pathlib

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

artifacts = {str(p): sha256(p) for p in pathlib.Path("engines").rglob("*.engine")}
manifest = {
    "git_sha": ...,
    "cuda": ..., "cudnn": ..., "trt": ..., "torch": ...,
    "hardware": ...,
    "artifacts": artifacts,
    "perf": load_perf_summary(),
    "accuracy": load_accuracy_summary(),
}
yaml.safe_dump(manifest, open("compression_manifest.yaml", "w"))
```

### Gate 6 (final)

- [ ] All hard PRs met.
- [ ] All deliverables in `deliverables/` per its README.
- [ ] `make all` succeeds end-to-end on a clean checkout.
- [ ] `make verify` reproduces numbers within 5%.

---

## Cross-phase gotcha catalog

### NaN under FP16

- Symptom: training loss explodes after a few hundred steps.
- Common causes: pre-softmax logits overflow; LayerNorm variance
  underflow; loss accumulation in FP16.
- Fix: keep loss in FP32, use `GradScaler`, BF16 for training where
  hardware supports it (Ampere+).

### OOM during calibration

- Symptom: `RuntimeError: CUDA out of memory` on the first calibration
  batch.
- Common causes: observer activation tensors held alive across forward
  passes; batch size set for inference but observers add per-channel
  histograms.
- Fix: smaller calibration batch size; use streaming observers;
  `torch.cuda.empty_cache()` between calibration epochs.

### Pruning broke inference shapes

- Symptom: `RuntimeError: size mismatch` after a pruning round.
- Common causes: didn't propagate channel reduction to downstream conv;
  forgot to slice BN affine params.
- Fix: dependency graph walker (section 3.3); unit test asserts forward
  pass on a held-out batch after every pruning round.

### TRT engine slower than baseline

- Symptom: engine builds but P50 is worse than FP32 baseline.
- Common causes: silent precision fallback (see 5.4 gotcha); workspace
  too small; layer-precision overrides forcing FP32 on the hot path.
- Fix: re-run with `BuilderFlag.OBEY_PRECISION_CONSTRAINTS`, enlarge
  workspace, drop overrides one by one.

### Kernel launch overhead dominates at bs=1

- Symptom: P50 floor around 0.5ms regardless of model size.
- Cause: each Python -> C -> CUDA launch is ~30us; small kernels are
  launch-bound.
- Fix: capture the engine in a CUDA Graph and replay
  (`torch.cuda.CUDAGraph` for PyTorch path, or TRT's execute_async_v3
  with persistent streams). This is a stretch goal but expected for
  rubric distinction.

### `nvidia-smi` says clocks locked but bench still flaky

- Cause: thermal throttling at >85C.
- Fix: improve cooling, lower clock target, or accept and document the
  derate. `clocks_throttle_reasons.hw_thermal_slowdown` is the field
  to check.

---

## Profiling cheat sheet

| Question | Command |
|----------|---------|
| Where is time spent overall? | `nsys profile --stats=true ...` |
| What kernels are hot? | `nsys stats --report cudaapisum,gpukernsum` |
| Is this kernel memory- or compute-bound? | `ncu --section MemoryWorkloadAnalysis,ComputeWorkloadAnalysis` |
| What's my occupancy? | `ncu --section Occupancy` |
| Why is this kernel slow? | `ncu --section SourceCounters` |
| Is FP32 sneaking in? | `nsys stats --report gpukernsum \| grep -i sgemm` |
| Are memory copies serializing? | `nsys stats --report cudaapisum \| grep memcpy` |

---

## Definition of done

You are done with this project when:

1. Every gate above is checked.
2. `make all` from clean checkout produces `compression_manifest.yaml`.
3. A reviewer can read `reports/benchmark_summary.md` and answer
   "does this hit the targets?" in under one minute.
4. A reviewer can re-run `make verify` and reproduce within 5%.
5. You can defend every trade-off in `architecture.md` in a 10-minute
   conversation.
