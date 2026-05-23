# Lecture 01: nsys + Nsight Compute

## Two profilers

| Tool | Granularity | Use for |
|---|---|---|
| `nsys` (Nsight Systems) | timeline view; what happens when | "where is the time going across the full pipeline?" |
| `ncu` (Nsight Compute) | per-kernel deep dive | "why is this specific kernel slow?" |

Start with nsys to find hotspots; drop into ncu to optimize them.

## nsys workflow

```bash
nsys profile -o trace.nsys-rep python train.py
nsys-ui trace.nsys-rep    # GUI
nsys stats trace.nsys-rep # text summary
```

Key views:
- **CUDA API**: time spent in malloc/free/memcpy/launch
- **CUDA kernels**: per-kernel total time + invocation count
- **NVTX ranges**: your code-defined regions ("forward", "backward", "optimizer step")

## ncu workflow

```bash
ncu --set full --kernel-id ::matmul_kernel:1 python train.py
ncu -i report.ncu-rep --section MemoryWorkloadAnalysis
```

Key sections:
- **GPU Speed of Light**: roofline placement
- **Memory Workload Analysis**: bandwidth + cache hit rates
- **Compute Workload Analysis**: throughput per pipeline
- **Source/SASS view**: per-instruction hotspots

## PyTorch profiler

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
) as prof:
    for _ in range(10):
        train_step()
        prof.step()
```

Then `tensorboard --logdir=log`. Easier than nsys for PyTorch-only workloads;
nsys when you need to see beyond PyTorch.

## Companion

[engineer-solutions/mod-107 ex-07 (gpu-memory-profiling)](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-107-gpu-computing/exercise-07-gpu-memory-profiling).
