# Lecture 01: CUDA Graphs + Streams

## Streams

CUDA streams = ordered queue of GPU work. Operations in the same stream run
sequentially; operations in different streams may run in parallel.

```python
import torch
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
with torch.cuda.stream(s1):
    out1 = expensive_op(x)
with torch.cuda.stream(s2):
    out2 = another_op(y)
torch.cuda.synchronize()
```

Use for: overlapping compute with data transfer (e.g., next batch loading while current trains).

## CUDA Graphs

Capture a sequence of CUDA ops once; replay many times. Eliminates per-launch overhead.

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = model(x)
# Replay
g.replay()
```

Wins ~5-20% for inference workloads dominated by small kernels.
`torch.compile(mode="reduce-overhead")` enables this automatically.

## Caveats

- Captured graph is fixed-shape; new input shape = recapture
- Allocations during capture are pinned for replay (no malloc/free)
- Debugging is harder (errors surface at replay, not capture)
