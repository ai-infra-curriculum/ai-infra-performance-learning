# Ex 01: CUDA Graphs

Take a small transformer inference path and capture it as a CUDA graph, then
benchmark graph replay against naive eager execution. CUDA graphs collapse the
per-kernel launch overhead into a single replay, so the win shows up most on
launch-bound, small-batch decode steps.

## Tasks

1. Build a fixed-shape inference step (e.g., one decode step of a GPT-2-sized
   model) at batch size 1. Static shapes are mandatory; CUDA graphs do not
   tolerate changing tensor sizes between replays.
2. Establish the eager baseline: warm up, then time the step with proper
   `torch.cuda.synchronize()` around the measured region (or CUDA events).
3. Capture the step with `torch.cuda.graph` (or `make_graphed_callables`).
   Pre-allocate static input/output buffers, copy new inputs into the static
   buffers each iteration, then call `graph.replay()`.
4. Verify correctness: replay output must match eager output within fp
   tolerance (`torch.allclose`, e.g., `atol=1e-3`).
5. Profile both paths with Nsight Systems (`nsys`) and confirm the launch gaps
   between kernels shrink in the graph version.

## Acceptance criteria

- Replay output matches eager within tolerance.
- Latency improvement of 5-15% on the decode step (more if it is heavily
  launch-bound; less for large-batch compute-bound steps — explain which case
  you hit).
- Nsight timeline shows reduced kernel-launch overhead under graph replay.

Document baseline vs replay (p50/p99 latency), the captured shapes, and the
`nsys` before/after observation in `CUDA_GRAPHS_REPORT.md`.

Reference: [PyTorch CUDA graphs
guide](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).
