# Module 1 — Resources

> Primary sources only. The rule is: if a number appears in
> `lecture-notes.md` or in an exercise's check, it traces back to one
> of these documents.

## NVIDIA primary sources

**CUDA C Programming Guide** — the authoritative reference for the
execution model, the memory model, and per-compute-capability limits.
The "Compute Capabilities" appendix (Appendix H in current revisions)
is what you reach for when you need the resource limits for a specific
GPU generation.
<https://docs.nvidia.com/cuda/cuda-c-programming-guide/>

**NVIDIA H100 Tensor Core GPU Datasheet** — H100 SXM and PCIe peak
throughputs (FP64, FP32, TF32, BF16, FP16, FP8, INT8), memory
bandwidth (3.35 TB/s SXM, 2 TB/s PCIe), TDP, NVLink bandwidth.
*The "with sparsity" asterisk is important*: published tensor-core
TFLOPS are the 2× sparsity number; divide by 2 for dense.
<https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306>

**NVIDIA A100 Tensor Core GPU Datasheet** — A100 SXM4 and PCIe specs
including the 80GB and 40GB variants. The 80GB SXM4 has 2039 GB/s
HBM2e bandwidth (vs 1555 GB/s for the 40GB variant) — they are
not interchangeable.
<https://www.nvidia.com/en-us/data-center/a100/>

**NVIDIA Hopper Architecture Whitepaper** — the deepest source for
the SM micro-architecture: fourth-gen Tensor Cores, the Transformer
Engine, distributed shared memory, tensor memory accelerator.
<https://resources.nvidia.com/en-us-tensor-core>

**NVIDIA Ampere Architecture Whitepaper** — same for A100. Third-gen
Tensor Cores, TF32, sparsity acceleration, MIG.
<https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf>

**NVIDIA Nsight Compute documentation** — kernel-level profiler. The
metrics dictionary (`smsp__*`, `dram__*`, `sm__*`) is dense; the
"Profiling Guide" is the right entry point.
<https://docs.nvidia.com/nsight-compute/>

**NVIDIA Nsight Systems documentation** — system-level / multi-process
timeline profiler. Use when the question is "is the GPU even busy?"
rather than "why is this kernel slow?"
<https://docs.nvidia.com/nsight-systems/>

## Roofline model

**S. Williams, A. Waterman, D. Patterson.** "Roofline: An Insightful
Visual Performance Model for Multicore Architectures."
*Communications of the ACM*, April 2009. The original paper that
defined the model used in `exercises/exercise-04-roofline-analysis/`.
<https://dl.acm.org/doi/10.1145/1498765.1498785>

**LBNL Roofline tutorial** — the practical companion to the Williams
paper. Shows how to actually measure the points the model predicts.
<https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/>

## Occupancy beyond the basics

**V. Volkov.** "Better Performance at Lower Occupancy."
*GTC 2010.* The canonical counterexample to "always maximize
occupancy." Volkov shows kernels that beat their high-occupancy
versions by exploiting instruction-level parallelism. Required
reading once you have done Exercise 3.
<https://www.cs.berkeley.edu/~volkov/volkov10-GTC.pdf>

## Optional but high-value

**NVIDIA CUDA C++ Best Practices Guide** — the "what to do" companion
to the Programming Guide's "how it works." Coalescing rules, shared
memory patterns, kernel launch overhead.
<https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>

**Mark Harris — "Optimizing Parallel Reduction in CUDA"** (NVIDIA
slides, 2007). A worked example of taking one kernel through seven
stages of optimization. Still the cleanest demonstration of the
methodology in Module 1.
<https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf>

---

*Last revised: 2026-05-22 — initial build.*
