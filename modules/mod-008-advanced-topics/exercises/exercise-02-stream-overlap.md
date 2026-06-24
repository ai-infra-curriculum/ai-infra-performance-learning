# Ex 02: Stream Overlap

Train a model where host-to-device data movement is the bottleneck, then use a
separate CUDA stream to overlap the next batch's transfer with the current
batch's compute. The default stream serializes copy and compute; a dedicated
copy stream plus pinned memory lets them run concurrently.

## Tasks

1. Build a training loop that is input-bound: a cheap model with large per-step
   tensor transfers (e.g., big batch of high-res images on a fast GPU). Confirm
   the bottleneck with `nsys` — the timeline should show compute stalling on
   `Memcpy HtoD`.
2. Allocate host inputs in pinned memory (`pin_memory=True` in the DataLoader,
   or `tensor.pin_memory()`) so transfers can be asynchronous.
3. Issue the next batch's transfer on a side stream with
   `non_blocking=True` while the current batch computes on the default stream.
   Use `torch.cuda.Stream` and synchronize with events (`stream.wait_event`)
   so compute never reads a half-copied buffer.
4. Double-buffer the staging tensors so the copy stream and compute stream are
   never writing/reading the same buffer in the same step.
5. Re-profile and measure step time before vs after overlap.

## Acceptance criteria

- Nsight timeline shows copy and compute kernels overlapping (copy on the side
  stream concurrent with compute on the default stream).
- Per-step time improves measurably — target at least a 1.2x speedup on the
  input-bound loop, or explain why the transfer was already hidden.
- Training numerics are unchanged: loss curve over the first N steps matches
  the non-overlapped baseline (no torn reads from missing synchronization).

Document the bottleneck evidence, before/after step time, and the
synchronization scheme in `STREAM_OVERLAP_REPORT.md`.

Reference: [CUDA streams and concurrency
docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution).
