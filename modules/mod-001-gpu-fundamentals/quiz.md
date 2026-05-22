# Module 1 quiz — GPU fundamentals

> Twelve questions. Do these without scrolling to `quiz-answers.md`.
> Target: ≥ 10 / 12. Below 8 / 12, reread the relevant lesson before
> moving on to Module 2.

## Section A — SIMT and warp execution (LO 1)

**1.** A CUDA programmer writes a kernel that branches based on
`threadIdx.x % 2`. Inside a single warp, what happens at the branch?

- (a) The 32 threads divide into two warps of 16; each takes one branch.
- (b) The warp executes both sides serially with predication; half the
  threads idle on each side.
- (c) The warp scheduler picks the more-common branch and discards the
  other 16 threads' work.
- (d) The compiler statically rewrites the branch into an `else`-free
  form; there is no runtime cost.

**2.** Why is the warp size of 32 a particularly important number?

- (a) It equals the number of CUDA cores per SM on every NVIDIA GPU.
- (b) It is the unit of scheduling on the GPU; one instruction is
  issued per warp per cycle, and warp-level primitives operate on
  exactly 32 lanes.
- (c) It is the L1 cache line size in 32-bit words.
- (d) It is the maximum number of threads per block.

## Section B — Performance metrics (LOs 2, 4)

**3.** Compute the peak FP32 throughput of an NVIDIA A100 SXM4 80GB
(108 SMs, 64 FP32 CUDA cores per SM, boost ≈ 1.41 GHz).

- (a) 4.9 TFLOPS
- (b) 9.7 TFLOPS
- (c) 19.5 TFLOPS
- (d) 39.0 TFLOPS

**4.** A kernel achieves an arithmetic intensity of 0.5 FLOP/byte on an
A100. Is it memory-bound or compute-bound? (A100 ridge point ≈ 9.6
FLOP/byte.)

- (a) Compute-bound — the FLOPS ceiling will be hit before the
  bandwidth ceiling.
- (b) Memory-bound — the bandwidth ceiling will be hit first, so adding
  more FLOP throughput cannot help.
- (c) Neither — at 0.5 FLOP/byte the kernel is in the "balanced"
  region.
- (d) Cannot be determined without knowing the kernel's launch
  configuration.

**5.** What is the arithmetic intensity of an FP32 vector add
(`c[i] = a[i] + b[i]`)?

- (a) 0.083 FLOP/byte
- (b) 0.5 FLOP/byte
- (c) 1.0 FLOP/byte
- (d) 8.0 FLOP/byte

**6.** A kernel running on an A100 hits 80% of peak HBM bandwidth
(≈ 1631 GB/s) at an arithmetic intensity of 0.25 FLOP/byte. What is
its achieved throughput?

- (a) ≈ 410 GFLOPS (memory-bound)
- (b) ≈ 19.5 TFLOPS (compute-bound at peak)
- (c) ≈ 4.0 TFLOPS
- (d) Cannot be computed without occupancy data.

## Section C — Memory hierarchy (LO 6)

**7.** Order these GPU memory levels from lowest latency (1) to highest
latency (5):

```
___ Global HBM
___ L1 cache / shared memory
___ Registers
___ L2 cache
___ PCIe host memory (over Gen4 x16)
```

**8.** A warp issues a load where the 32 threads access 32 contiguous
4-byte words starting at an aligned 128-byte boundary. Approximately
how many global-memory transactions does the hardware issue?

- (a) 1 (fully coalesced).
- (b) 4 (one per 32-byte segment).
- (c) 8 (one per 16-byte segment).
- (d) 32 (one per thread).

**9.** Which of these scenarios most likely benefits from using
*shared memory* instead of going to global memory each access?

- (a) A vector-add kernel where each thread reads two inputs and
  writes one output, with no reuse.
- (b) A matrix multiply where threads in a block repeatedly read the
  same row of A and column of B.
- (c) A pointer-chasing kernel where each thread follows a
  data-dependent linked-list traversal.
- (d) A reduction kernel that processes 1 GB of input once and writes
  out a single scalar.

## Section D — Occupancy (LO 5)

**10.** A kernel uses 128 threads per block and 80 registers per
thread. On an Ampere SM (65,536 32-bit registers, max 64 warps), how
many concurrent blocks can fit on one SM (register-limit only)?

- (a) 1
- (b) 4
- (c) 6
- (d) 16

**11.** Which statement is most accurate?

- (a) Maximizing occupancy is always the goal; a kernel at 100%
  occupancy will outperform the same kernel at 50% occupancy.
- (b) Occupancy is a tool for hiding latency. A low-occupancy kernel
  with high instruction-level parallelism can outperform a
  high-occupancy kernel.
- (c) Occupancy is a function of the launch configuration only and
  does not depend on per-thread register usage.
- (d) Occupancy is irrelevant once tensor cores are used.

## Section E — Putting it together (LO 4)

**12.** You are handed a kernel that, after profiling, shows 92% of peak
HBM bandwidth and 8% of peak FP32 throughput. Which of these is the
*next* optimization to try?

- (a) Increase block size to raise occupancy.
- (b) Reduce per-thread register usage to fit more warps per SM.
- (c) Find a way to reduce bytes moved per FLOP (e.g., fuse with a
  neighboring kernel, use shared memory tiling, switch to a lower
  precision).
- (d) Inline more arithmetic to bring FP32 utilization closer to peak.
