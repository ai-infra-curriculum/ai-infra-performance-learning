# Module 1 quiz — answer key

> Self-grade. ≥ 10 / 12 to proceed; below 8 / 12, reread the relevant
> lesson(s) listed against each missed question.

---

**1. (b)** — Predicated execution. The warp executes both sides of the
branch serially; threads on the inactive side sit idle. *Lesson 1.5.*

**2. (b)** — Warp size 32 is the unit of scheduling. One instruction
is issued per warp per cycle (per SM sub-partition), and warp-level
primitives (shuffles, votes) operate on exactly 32 lanes. (a) is
wrong: SMs typically have 64 or 128 FP32 cores. *Lesson 1.1.*

**3. (c)** — 108 × 64 × 1.41 × 2 / 1000 ≈ 19.49 TFLOPS. *Lesson 1.6,
Exercise 2.*

**4. (b)** — 0.5 FLOP/byte is far below the ~9.6 FLOP/byte ridge
point. The kernel is memory-bound; FLOP-side optimization cannot
help until the bytes-moved-per-FLOP is reduced or the data is
moved faster. *Lesson 1.6.*

**5. (a)** — 1 FLOP per element / 12 bytes per element (read 4 + read
4 + write 4 in FP32) = 0.083 FLOP/byte. *Lesson 1.6, Exercise 4.*

**6. (a)** — Throughput = 1631 GB/s × 0.25 FLOP/byte = 408 GFLOPS.
The kernel is memory-bound, so multiplying achieved bandwidth by
arithmetic intensity gives the achieved FLOP throughput. *Lesson 1.6.*

**7.** Latency order (1 = lowest, 5 = highest):

```
4  Global HBM             (~500 cycles)
2  L1 / shared memory     (~20-30 cycles)
1  Registers              (~1 cycle)
3  L2 cache               (~200 cycles)
5  PCIe host memory       (~thousands of cycles + transfer)
```

*Lesson 1.3.*

**8. (a)** — 32 contiguous, aligned 4-byte words is exactly one
128-byte memory transaction — the canonical coalesced case.
*Lesson 1.3.*

**9. (b)** — Matrix multiply: threads in a block repeatedly use the
same row of A and column of B, so loading them once into shared
memory and reusing pays for itself many times over. (a) has no reuse
(shared memory does not help); (c) is data-dependent (cannot
pre-stage); (d) is one-pass streaming (also no reuse).
*Lesson 1.3.*

**10. (c)** — Register cap = floor(65536 / (128 × 80)) =
floor(65536 / 10240) = 6 blocks per SM if registers are the only
constraint. (The other caps — block-count = 32, warp-cap = 64/4 = 16
— are looser here, so registers do bind in this configuration.)
*Lesson 1.6, Exercise 3.*

**11. (b)** — High occupancy is one tool for hiding latency, not a
target in itself. Volkov 2010 ("Better Performance at Lower
Occupancy") is the canonical counterexample. *Lesson 1.6.*

**12. (c)** — At 92% of peak HBM, the kernel is firmly
memory-bound. Adding more arithmetic, increasing occupancy, or
reducing registers cannot help — the bottleneck is bytes, not
FLOPS. Reducing bytes moved per FLOP (operator fusion, shared-memory
tiling, lower precision) is the only path to higher throughput.
*Lesson 1.6.*
