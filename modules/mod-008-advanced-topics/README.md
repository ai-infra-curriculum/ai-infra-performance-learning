# Module 8 — Advanced Topics

> **Track:** AI/ML Performance Engineer
> **Time budget:** ~20 hours (2 weeks part-time)
> **Module status:** first build, 2026-11-06

Module 07 took a model from "trained" to "in production" — a signed
container, a Helm chart, a canary with an auto-revert gate, cost-tagged
resources, and an on-call runbook. Module 08 is the last mile of the
curriculum: the frontier techniques that sit *underneath* every
production stack built so far, and that start to matter once the
easy wins from Modules 03–07 are gone. Launch overhead stops being a
footnote and becomes the dominant term on short-sequence decode
(CUDA Graphs). The default stream's serialisation of copy and compute
turns into a measurable loss (custom streams). An inter-node collective
over the wrong fabric costs more wall-clock than a kernel rewrite ever
earned back (NVLink/InfiniBand/GPUDirect). A single tenant leaves most
of an H100 idle when seven partitioned tenants would saturate it (MIG).
And fp8 — on the right hardware with the right scaling discipline — is
a free 30–50% on top of an already-tuned bf16 pipeline.

The thread running through this module is **the frontier techniques are
not general-purpose speedups; they are regime-specific wins with a
narrow envelope of when they pay off**. CUDA Graphs do nothing for a
compute-bound kernel; they win on launch-bound decode. A side-stream
copy overlap is a no-op if your transfers are already hidden behind
compute; it is a 1.5× speedup when the default stream was serialising
them. NVLink makes tensor parallelism practical inside a node and makes
it a performance trap outside one. MIG trades peak throughput for
isolation and multi-tenant utilisation; it is the right answer for a
notebook cluster and the wrong answer for a single large training job.
FP8 on H100 with Transformer Engine is the standard; fp8 anywhere else,
or without TE's per-tensor scaling, is a research project. Every topic
in this module comes with its own "when it wins" and "when it is
overhead," and the learning objectives are organised around recognising
which regime you are in before you reach for the knob.

The posture this module trains is **profile first, pick the technique
second**. The Module 03 measurement discipline — warmup, synchronize,
median of several runs, before/after Nsight Systems timelines — is the
only thing that keeps the frontier techniques honest. A CUDA Graph
rollout that does not include an `nsys` timeline showing the shrunk
kernel-launch gaps is not a result. An fp8 training run that does not
include a bf16 accuracy baseline on the same eval is not a conclusion.

## Learning objectives

By the end of this module, you will be able to:

1. **Explain**, in one minute, what a CUDA Graph is — a captured
   sequence of GPU operations that replays as one launch, amortising
   per-kernel launch overhead — and identify the regime where it wins
   (launch-bound, small-batch, fixed-shape decode steps) versus where
   it is neutral (large-batch compute-bound kernels). Source:
   [PyTorch CUDA graphs blog](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
   and the [CUDA C Programming Guide §CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs).
2. **Capture** a transformer decode step with
   `torch.cuda.CUDAGraph` (or `torch.compile(mode="reduce-overhead")`),
   verify replay output matches eager within fp tolerance, and report
   the latency delta with an Nsight Systems timeline that shows the
   closed-up launch gaps. State the three capture-time constraints:
   static shapes, no allocations during capture, errors surface at
   replay rather than capture.
3. **Use** CUDA streams to overlap independent operations — a side
   stream for the next batch's H2D copy while the default stream
   computes the current batch — with pinned host memory and event-based
   synchronisation (`stream.wait_event`) so compute never reads a
   half-copied buffer. Source: [CUDA C Programming Guide §Asynchronous
   Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution).
4. **Distinguish** the three layers of GPU-to-GPU communication —
   NVLink/NVSwitch intra-node (H100 NVLink Gen4: ~900 GB/s aggregate
   per GPU), InfiniBand inter-node (HDR 200 Gb/s, NDR 400 Gb/s per
   port), and PCIe as the fallback (Gen4 x16: ~64 GB/s) — and explain
   why NVLink is effectively required for in-node tensor parallelism
   and why inter-node PP wants IB + GPUDirect RDMA rather than staging
   through CPU memory. Sources: [NVIDIA NVLink / NVSwitch
   page](https://www.nvidia.com/en-us/data-center/nvlink/),
   [NVIDIA InfiniBand
   page](https://www.nvidia.com/en-us/networking/infiniband/),
   [GPUDirect RDMA
   docs](https://docs.nvidia.com/cuda/gpudirect-rdma/).
5. **Run** `nccl-tests` (`all_reduce_perf`) on a cluster, record
   intra-node (NVLink-only) and inter-node (NVLink + IB) bandwidth at
   representative message sizes, and diagnose a slow-fabric result
   from the topology (`nvidia-smi topo -m`) and NCCL's own debug logs
   (`NCCL_DEBUG=INFO`). Sources: [NVIDIA NCCL
   repository](https://github.com/NVIDIA/nccl),
   [nccl-tests](https://github.com/NVIDIA/nccl-tests).
6. **Partition** an A100 or H100 with MIG, deploy multiple independent
   serving instances on the resulting profiles (e.g. `1g.5gb`,
   `2g.10gb`, `3g.20gb` on A100), and explain the two regimes where
   MIG is the right choice (multi-tenant isolation; many small models
   on one node) and the one where it is wrong (a single large training
   job that wants the whole GPU). Source: [NVIDIA MIG User
   Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/).
7. **Train** a transformer in fp8 with NVIDIA Transformer Engine on
   H100, compare against a bf16 baseline on the same model and eval,
   and report wall-clock speedup, memory savings, and accuracy delta.
   Explain in one sentence why rolling your own fp8 — without TE's
   per-tensor scaling for the E4M3 and E5M2 formats — is a research
   project and not a production path. Source: [Transformer Engine User
   Guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/).

Every exercise targets one or more of these; the quiz checks the
underlying concepts.

## Prerequisites

You should be comfortable with:

- **Module 03** — `nsys`, `ncu`, and the measure → change → re-measure
  loop. Every exercise in this module is scored with a before/after
  pair of numbers plus the Nsight timeline that explains them. A CUDA
  Graph "feels faster" is not a result; the closed-up launch gaps on
  the timeline are.
- **Module 04** — the vLLM serving stack and the intuition for
  where decode steps are launch-bound. Exercise 01 captures a single
  decode step; without the Module 04 picture of "what a decode step
  is actually doing," the acceptance criterion is hard to reason
  about.
- **Module 05** — quantization, 2:4 sparsity, and LoRA. The fp8
  discussion in Lecture 03 treats the E4M3/E5M2 formats as a cousin
  of the int8/int4 story from Module 05; the compose-with-what
  mindset from that module is what keeps "fp8 on top of quantized
  weights" from turning into a numerics horror show.
- **Module 06** — tensor parallelism, pipeline parallelism, NCCL. The
  interconnect lecture in this module is the *hardware floor* under
  the TP/PP discussion from Module 06; if you cannot state the
  per-layer all-reduce traffic term for TP, re-read Module 06
  Chapter 01 before starting Exercise 03 here.
- **Multi-GPU hardware with NVLink and, ideally, a second node with
  InfiniBand** — Exercise 02 (stream overlap) runs on a single GPU;
  Exercise 03 (nccl-tests) wants at least one NVLink-connected 2-GPU
  node for the intra-node number and a second node with IB for the
  inter-node number. If multi-node hardware is not available, run the
  intra-node half and read the inter-node result from a published
  benchmark rather than inventing one.
- **An H100 (or newer) for Exercise 05** — the fp8 Tensor Cores
  shipped first on Hopper; A100 has no fp8 hardware and will not
  produce a meaningful speedup. If H100 time is not available, the
  exercise degrades to "read the Transformer Engine quick-start and
  summarise the E4M3/E5M2 trade-off"; keep the write-up and skip the
  measured numbers.
- **An A100 or H100 for Exercise 04** — MIG is a Tesla-class feature
  (A100, H100, GH200, H200). Consumer GPUs do not support it.

If you do not have the right hardware locally, a short-lived cloud
instance (8×H100 for a few hours) covers every exercise in this module
and is cheaper than owning the hardware for a curriculum unit.

## Module structure

The lecture is split across three chapter files in the
[`lecture-notes/`](lecture-notes/) directory. Read in order:

| Chapter | What's in it |
|---|---|
| [`lecture-notes/01-cuda-graphs-streams.md`](lecture-notes/01-cuda-graphs-streams.md) | CUDA streams as ordered queues of GPU work, and how independent operations in different streams run concurrently; CUDA Graphs as capture-once / replay-many to amortise per-launch overhead; the three capture-time constraints (fixed shapes, no allocations, errors surface at replay); the regime where graphs win (small-batch launch-bound decode, ~5–20% on inference) and where they do not (compute-bound training steps). |
| [`lecture-notes/02-interconnect.md`](lecture-notes/02-interconnect.md) | NVLink / NVSwitch as the ~900 GB/s intra-node fabric that makes TP practical, vs ~64 GB/s PCIe Gen4 x16 as the fallback that makes it a trap; InfiniBand HDR/NDR as the inter-node fabric (RDMA, multi-rail); GPUDirect RDMA for zero-copy GPU-to-GPU across nodes; the three diagnostic tools (`ib_send_bw`, `nccl-tests`, `nvidia-smi topo -m`) and when each is the right first probe. |
| [`lecture-notes/03-mig-fp8.md`](lecture-notes/03-mig-fp8.md) | MIG partitioning an A100/H100 into up to seven hardware-isolated instances with their own memory and compute; the multi-tenant and many-small-models regimes where MIG wins and the single-large-training-job regime where it does not; fp8 on H100 via Transformer Engine, the E4M3 vs E5M2 range/precision trade-off, per-tensor scaling as the thing that keeps fp8 numerically stable, and the 30–50% training speedup / 1.5–2× inference speedup bands reported on transformer workloads. |

| Exercise | What it covers |
|---|---|
| [`exercises/exercise-01-cuda-graphs.md`](exercises/exercise-01-cuda-graphs.md) | Capture a single fixed-shape transformer decode step with `torch.cuda.CUDAGraph`, verify replay output matches eager within fp tolerance, and report p50/p99 latency before and after. Confirm the closed-up kernel-launch gaps on an Nsight Systems timeline. |
| [`exercises/exercise-02-stream-overlap.md`](exercises/exercise-02-stream-overlap.md) | Build an input-bound training loop, pin host memory, and overlap the next batch's H2D copy on a side stream with the current batch's compute on the default stream. Double-buffer, synchronise with events, and show the copy/compute overlap on the Nsight timeline plus a measurable step-time speedup. |
| [`exercises/exercise-03-nccl-tests.md`](exercises/exercise-03-nccl-tests.md) | Run `all_reduce_perf` from `nccl-tests` intra-node (NVLink only) and inter-node (NVLink + IB). Record bus bandwidth at a 1B-element message, compare against the theoretical peak for the cluster topology, and diagnose any anomaly from `nvidia-smi topo -m` plus `NCCL_DEBUG=INFO`. |
| [`exercises/exercise-04-mig-partition.md`](exercises/exercise-04-mig-partition.md) | Enable MIG on an A100 or H100. Create a representative mix of profiles (e.g. `1g.5gb`, `2g.10gb`, `3g.20gb` on A100). Deploy an independent vLLM instance on each partition and verify hardware isolation — one partition's workload must not affect another's latency. |
| [`exercises/exercise-05-fp8-training.md`](exercises/exercise-05-fp8-training.md) | Train a small transformer on H100 with NVIDIA Transformer Engine in fp8, compare against a bf16 baseline on the same model and eval, and report wall-clock speedup, memory savings, and accuracy delta. Use TE's per-tensor scaling; do not roll your own. |

Plus:

| File | What's in it |
|---|---|
| [`quiz.md`](quiz.md) / [`quiz-answers.md`](quiz-answers.md) | 10-question multiple-choice check on the learning objectives. 70% to pass. |
| [`resources.md`](resources.md) | Primary sources — CUDA Graphs docs, NCCL + nccl-tests, NVLink / InfiniBand / GPUDirect RDMA pages, MIG user guide, Transformer Engine docs. |

<!-- needs-research: confirm the exact engineer-solutions path shape for the MIG/GPU-sharing companion (currently referenced as `engineer-solutions/mod-107 ex-10` in resources.md); cross-check the public layout before linking from course-facing material. -->

## How to work through this module

1. Read the three chapter files in `lecture-notes/` in order. Plan for
   roughly 3–4 hours of reading plus small code-alongs. Chapter 01
   (CUDA Graphs + streams) is the one most students under-rate — the
   launch-overhead story sounds minor on paper and turns out to be
   the dominant term on small-batch decode in practice, so slow down
   on the "when graphs win" regime discussion.
2. Do the five exercises in order where hardware allows. Exercise 01
   (CUDA Graphs) is single-GPU and the cheapest to run; Exercise 02
   (stream overlap) is also single-GPU and makes the pinned-memory +
   side-stream pattern concrete. Exercise 03 (nccl-tests) wants a
   multi-GPU node for the intra-node number and a second node with
   InfiniBand for the inter-node number. Exercise 04 (MIG) needs an
   A100 or H100; Exercise 05 (fp8) needs an H100 or newer.
3. Profile every change. "CUDA Graphs made decode faster" is not what
   this module is training — the goal is before/after p50/p99 plus the
   Nsight timeline that shows the launch gaps shrinking (Exercise 01)
   or the copy/compute streams overlapping (Exercise 02). Save the
   `nsys` outputs alongside the per-exercise report; one without the
   other is not a result.
4. Verify correctness before you celebrate speed. Every speedup in
   this module has a numerics failure mode that produces fast nonsense:
   a CUDA Graph can replay with stale inputs if the static buffers
   were not updated between iterations; a side-stream copy can race if
   the event sync is missing; fp8 training diverges silently when the
   per-tensor scaling is wrong. Every exercise lists a correctness
   check — run it before the benchmark numbers, not after.
5. Measure fabric before you blame the model. If a distributed job is
   slow, Exercise 03 (nccl-tests) is the right *first* probe — it
   rules the fabric in or out before you spend an afternoon reading
   model code. The same `nccl-tests` number is also the ceiling any
   TP/PP configuration from Module 06 can reach, so a low bus
   bandwidth result caps every multi-GPU exercise above it.
6. Treat fp8 as a measured decision, not a default. Exercise 05 is
   worth running only on H100 (or newer) with Transformer Engine; on
   older hardware the exercise degrades to a reading assignment and
   the "fp8 is a free 30–50%" claim is not something you have earned.
   Pair the speedup number with an accuracy delta on the same eval —
   without the accuracy number, the speedup is not reportable.
7. Take the quiz **without** scrolling to `quiz-answers.md`. If you
   miss more than three questions, reread the chapter that covers
   them before moving on.

## Assessment criteria for this module

You are ready to call Module 08 complete when you can, on a blank page:

- State the CUDA Graphs one-line claim (capture a sequence of GPU
  ops once; replay many times as one launch to amortise per-kernel
  launch overhead) and the one regime where it wins (small-batch,
  fixed-shape, launch-bound decode steps, ~5–20% on inference) and
  the one regime where it does not (large-batch compute-bound
  training steps). Name the three capture-time constraints: static
  shapes, no allocations during capture, errors surface at replay.
- Sketch the stream-overlap pattern — pinned host memory + side
  stream + event-based synchronisation + double-buffered staging —
  and explain in one sentence why the default stream alone
  serialises copy and compute and therefore cannot overlap them.
- Name the three layers of GPU-to-GPU communication (NVLink/NVSwitch
  intra-node at ~900 GB/s on H100; InfiniBand inter-node at
  200–400 Gb/s per port; PCIe Gen4 x16 as the ~64 GB/s fallback) and
  the two rules that follow: TP goes inside the NVLink boundary, PP
  can cross it over IB + GPUDirect RDMA.
- Give the right first diagnostic for "slow distributed training" —
  run `nccl-tests` (`all_reduce_perf`) to rule the fabric in or out
  before touching model code — and name the two tools that tell you
  *why* the fabric is slow when it is (`nvidia-smi topo -m` for the
  topology, `NCCL_DEBUG=INFO` for the NCCL side).
- State what MIG does (hardware-partition an A100/H100 into up to
  seven isolated instances, each with its own memory and compute) and
  the two regimes where it is the right call (multi-tenant isolation;
  many small models co-located on one node) versus the one where it
  is not (a single large training job that wants the whole GPU).
- Explain the fp8 trade-off: E4M3 has finer precision and a smaller
  range (~[-448, 448]), E5M2 has a wider range (~[-57344, 57344]) at
  the cost of precision; the narrow dynamic range of fp8 is what
  makes per-tensor scaling mandatory, and Transformer Engine is the
  library that manages it. Give the rough speedup bands: 30–50% on
  H100 training vs bf16; 1.5–2× on H100 inference vs fp16/bf16.
- Explain in one minute when **not** to reach for any of the
  techniques in this module: CUDA Graphs are overhead on
  compute-bound kernels; a side-stream copy overlap is a no-op when
  transfers are already hidden; NVLink does not help a single-GPU
  workload; MIG costs peak throughput and is wrong for a single big
  job; fp8 without Transformer Engine on non-Hopper hardware is a
  research project. Every technique here is regime-specific — know
  the regime before you turn the knob.

Module 08 is the end of the core curriculum. From here the honest next
steps are not another module but the production discipline the whole
track has been building toward: ship a change, measure it on your own
workload with the Module 03 tools, and compare the numbers against
what the paper or the vendor doc claimed. The frontier moves faster
than any course can — new kernels, new fabrics, new number formats
arrive every quarter — and the thing that keeps an engineer current is
not the next lecture but the habit of applying the measure →
change → re-measure loop to whatever paper, kernel, or hardware
generation lands on the stack next. The
[`resources.md`](resources.md) in each module is the pointer back to
the primary sources when a technique needs re-reading; the
[`projects/`](../../projects/) directory in the repo root holds the
capstones that stitch several modules' worth of technique together.
