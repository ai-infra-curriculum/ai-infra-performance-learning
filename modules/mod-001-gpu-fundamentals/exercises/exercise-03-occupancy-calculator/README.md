# Exercise 3 — Build an occupancy calculator from scratch

> **Targets learning objectives:** 3, 5
> **Time:** ~60–75 min
> **Requires:** Python 3.10+. No GPU needed.
> **References:** lecture-notes/04-warps-and-occupancy.md (the three resource caps, worked occupancy calculation).

## What you'll do

You will reimplement NVIDIA's occupancy calculation. Given a kernel
launch configuration (block size, registers per thread, shared
memory per block) and an SM resource budget (max threads, max
blocks, register file size, shared memory size), your function
returns the *achievable* occupancy as a fraction of the theoretical
maximum.

You have used the NVIDIA Occupancy Calculator spreadsheet, and you
may have used `cudaOccupancyMaxActiveBlocksPerMultiprocessor` in
code. This exercise is the math behind both. After it, you will
be able to answer "is this kernel register-limited or
shared-memory-limited?" in your head from the launch line and the
resource sheet.

## Why this skill matters

Performance tuning on the GPU is, roughly half the time, "this
kernel is at X% occupancy; what should I do?" The CUDA Occupancy
Calculator spreadsheet exists because the answer requires
understanding *which resource* is the binding constraint.

A few real situations:

- **"Why does adding one variable to my kernel cut throughput in
  half?"** Because that one variable pushed register usage past
  a cliff where the chip can fit one fewer block per SM, and now
  occupancy dropped from 50% to 25%, killing latency hiding.
- **"Why does my shared-memory tile size change the kernel's
  speed?"** Because a larger tile lets you reuse more, but if it
  pushes shared usage per block past the per-SM ceiling, you fit
  one fewer block per SM. Sometimes more reuse < more occupancy.
- **"This kernel achieves 8% of peak. The profiler says 6% theoretical
  occupancy. Where's the limit?"** Occupancy calculator tells you
  in one shot.

The calculation is conceptually simple. Internalizing the
*reasoning* — "what's the binding resource?" — is what makes you
fast at debugging.

## The calculation

For one SM and one kernel:

```
warps_per_block       = ceil(threads_per_block / 32)

# Each of three resources caps how many blocks can run concurrently
# on one SM. Take the smallest cap; that is the actual blocks-per-SM.
warp_cap     = floor(max_warps_per_sm / warps_per_block)
block_cap    = max_blocks_per_sm
register_cap = floor(register_file_size / (threads_per_block * registers_per_thread))
shared_cap   = floor(shared_mem_per_sm  / shared_mem_per_block)

active_blocks_per_sm = min(warp_cap, block_cap, register_cap, shared_cap)
active_warps_per_sm  = active_blocks_per_sm * warps_per_block
occupancy            = active_warps_per_sm / max_warps_per_sm
```

A few edge cases to handle correctly:

- If `shared_mem_per_block == 0`, the shared-memory cap is
  effectively infinity (there is no shared-memory constraint).
- If `registers_per_thread == 0`, similarly the register cap is
  effectively infinity (a kernel using zero registers does not
  exist, but the math should handle it cleanly).
- A `threads_per_block` not a multiple of 32 still rounds *up* to
  a whole warp; the leftover lanes are wasted but a warp is the
  unit of scheduling.
- Returned occupancy is a fraction in `[0.0, 1.0]`.

Your function should also report the *binding* resource — the one
whose cap is the smallest — so the caller can answer "what should
I optimize?" without rerunning the calculation.

### Walk through a non-trivial case

A kernel uses:
- 256 threads per block (= 8 warps)
- 40 registers per thread
- 8 KB of shared memory per block

On an A100 SM:
- max_warps = 64
- max_blocks = 32
- register_file = 65,536 (32-bit)
- shared per SM = 164 KB

Computing each cap:

```
warps_per_block = ceil(256 / 32) = 8

warp_cap     = 64 // 8 = 8                       (8 blocks fit by warp count)
block_cap    = 32                                (hard cap)
register_cap = 65536 // (256 * 40) = 6           (6 blocks fit by registers)
shared_cap   = (164 * 1024) // (8 * 1024) = 20   (20 blocks fit by shared)

binding = min(8, 32, 6, 20) = 6 -> "registers"
active_blocks = 6
active_warps  = 6 * 8 = 48
occupancy     = 48 / 64 = 0.75 = 75%
```

The kernel is **register-bound** at 75% occupancy. To raise
occupancy, reduce registers per thread.

## Reference SM specs

The autograder uses an Ampere A100 SM:

| Resource | Value |
|---|---|
| max_warps_per_sm | 64 |
| max_blocks_per_sm | 32 |
| register_file_size (32-bit registers) | 65,536 |
| shared_mem_per_sm (configured high) | 163 KB (= 163 * 1024 bytes ≈ 164 KB rounded up to 164 in some docs) |

A real implementation might also handle other architectures
(Volta with 80 KB shared, Hopper with 228 KB shared), but the
autograder sticks to A100 for simplicity.

## What to submit

Edit `starter.py`. Implement:

```python
def occupancy(launch: LaunchConfig, sm: SMSpec) -> OccupancyResult:
    ...
```

The function returns an `OccupancyResult` with:
- `active_blocks_per_sm: int` — how many blocks fit
- `active_warps_per_sm: int` — total warps that fit
- `occupancy: float` — in [0.0, 1.0]
- `limiter: str` — one of `"warps"`, `"blocks"`, `"registers"`,
  `"shared_mem"` indicating which resource was binding

Then run:

```bash
python check.py
```

The autograder tests your implementation against a battery of
cases including the four shown in the lecture (FP32 vector add,
sgemm tile, register-heavy kernel, shared-memory-heavy kernel)
plus edge cases.

## Hints

### Implementation skeleton

```python
import math

def occupancy(launch: LaunchConfig, sm: SMSpec) -> OccupancyResult:
    warps_per_block = math.ceil(launch.threads_per_block / WARP_SIZE)
    if warps_per_block == 0:
        warps_per_block = 1   # edge case; shouldn't happen but be safe

    warp_cap = sm.max_warps_per_sm // warps_per_block
    block_cap = sm.max_blocks_per_sm

    if launch.registers_per_thread > 0:
        register_cap = sm.register_file_size // (
            launch.threads_per_block * launch.registers_per_thread
        )
    else:
        register_cap = math.inf

    if launch.shared_mem_per_block_bytes > 0:
        shared_cap = sm.shared_mem_per_sm_bytes // launch.shared_mem_per_block_bytes
    else:
        shared_cap = math.inf

    caps = {
        "warps": warp_cap,
        "blocks": block_cap,
        "registers": register_cap,
        "shared_mem": shared_cap,
    }
    binding_cap = min(caps.values())
    # Tie-breaking order matters: warps > blocks > registers > shared_mem.
    limiter = next(name for name, v in caps.items() if v == binding_cap)

    active_blocks = min(int(binding_cap), sm.max_blocks_per_sm)
    active_warps = active_blocks * warps_per_block
    occ = active_warps / sm.max_warps_per_sm
    return OccupancyResult(
        active_blocks_per_sm=active_blocks,
        active_warps_per_sm=active_warps,
        occupancy=occ,
        limiter=limiter,
    )
```

You'll figure out roughly that shape yourself; the autograder
doesn't care about your specific code, only the output.

### Why `math.inf` for empty resources

If `shared_mem_per_block == 0`, the shared-memory cap is
"unbounded." But `min(very_large_number, ...)` works correctly,
so you can use `math.inf`. Or use any sentinel value larger than
`max_blocks_per_sm`. The key insight: a resource that's not used
shouldn't constrain anything.

### The tie-breaking order

When two caps are equal, you have to pick which one is "binding."
The autograder expects the order **warps > blocks > registers >
shared_mem**: if `warp_cap == register_cap == 8` you should
report `"warps"` as the limiter, not `"registers"`. This
matches NVIDIA's spreadsheet convention.

A neat way to do it: build the dict in that order and use
`next(name for name, v in caps.items() if v == binding_cap)`.
Python preserves insertion order, so the first matching value
wins.

### The block_cap clamp

After taking the min over caps, you also clamp by
`sm.max_blocks_per_sm`. Why? Because some pathological case
might give `warps_per_block = 1` and `warp_cap = max_warps_per_sm
= 64`, but the hardware also has `max_blocks = 32`. The clamp
ensures `active_blocks` never exceeds the hardware limit.

## Test cases the autograder runs

These cases are in `check.py`:

| # | Description | Threads | Regs | Shared | Expected limiter | Expected occupancy |
|---|---|---|---|---|---|---|
| 1 | vector add (low resources) | 256 | 16 | 0 | warps | 100% |
| 2 | sgemm 16x16 tile (moderate) | 256 | 40 | 8 KB | registers | 75% |
| 3 | register-heavy kernel | 256 | 96 | 0 | registers | 25% |
| 4 | shared-memory-heavy kernel | 128 | 32 | 48 KB | shared_mem | 18.75% |
| 5 | tiny block (32 threads) | 32 | 8 | 0 | blocks | 50% |
| 6 | threads not multiple of 32 (97) | 97 | 32 | 0 | warps | 100% |
| 7 | low resources, normal block | 128 | 8 | 0 | warps | 100% |

You're welcome to manually compute each case before running the
autograder to confirm your math.

## Common pitfalls

### 1. Forgetting to round threads up to whole warps

```
WRONG: warps_per_block = threads_per_block / 32   # might be 8.5
RIGHT: warps_per_block = math.ceil(threads_per_block / 32)
```

A block of 97 threads needs 4 warps, not 3.0625.

### 2. Using `/` instead of `//`

`floor` division matters for the caps. `64 / 8 = 8.0` works
numerically but you should think about it as `8` blocks. If
register_cap is `65536 / (256 * 40) = 6.4`, the answer is **6**
blocks, not 6.4.

### 3. Confusing "max blocks" with "max blocks given resources"

`max_blocks_per_sm = 32` is a hardware ceiling. Your actual
blocks-per-SM is the min of all caps. Don't return `min(...,
max_blocks_per_sm)` after already incorporating block_cap into
the caps dict; that's redundant. (But the clamp after the min
*is* needed for the pathological case above.)

### 4. Reporting wrong limiter on ties

If `warp_cap == register_cap == 8`, the autograder expects
`"warps"`. If you use Python's `min(caps.items(), key=lambda
kv: kv[1])`, you'll get whatever Python's sort considers "first"
— which may not match the expected order. Build the dict in
order and iterate.

### 5. Forgetting `math.inf` for zero-resource cases

If shared_mem_per_block is 0, your kernel uses no shared memory,
so it shouldn't be shared-memory-limited. `0 // 0` raises
ZeroDivisionError; you need the `if` guard.

## Grading rubric

The autograder runs your `occupancy(...)` against 7 cases (see
table above) and a hidden reference implementation. For each case
it checks:

1. `active_blocks_per_sm` matches the reference exactly.
2. `active_warps_per_sm` matches the reference exactly.
3. `occupancy` matches the reference within 1e-6.
4. `limiter` is the expected string.

If all 7 cases pass, `check.py` exits 0 and prints `PASS`. If any
case fails, you'll see a per-case diagnostic showing what you
returned vs what was expected.

## What "right" looks like

Your function should agree with NVIDIA's occupancy calculator on
every test case the autograder runs. If it does, you can throw
away the spreadsheet — you have the math.

After this exercise you should be able to:

- Look at `<<<blocks, 256>>>` with 40 registers/thread, 8 KB
  shared, and immediately answer "register-bound at 75%
  occupancy."
- Predict the effect of swapping in a kernel that uses 48
  registers/thread instead.
- Explain to a colleague why their kernel's "occupancy is too
  low" complaint is or isn't the right thing to optimize.

## Extended worked examples

### Walking through every case the autograder runs

The 7 test cases in `check.py` deserve walking through manually
before you implement, so you know what "right" looks like.

**Case 1: vector add (low resources)**

```
Inputs: threads=256, regs=16, shared=0
warps_per_block = 8

warp_cap     = 64 // 8                = 8
block_cap    = 32
register_cap = 65536 // (256 * 16)    = 65536 // 4096 = 16
shared_cap   = inf (no shared used)

binding = min(8, 32, 16, inf) = 8 -> "warps"
active_blocks = 8
active_warps  = 8 * 8 = 64
occupancy     = 64 / 64 = 1.0
```

Full occupancy, warp-cap limited. The kernel is using so few
resources per block that you fit the maximum 8 blocks
(constrained by warps, since 8 blocks of 8 warps = 64 warps =
max).

**Case 2: sgemm 16x16 tile (moderate)**

```
Inputs: threads=256, regs=40, shared=8 KB
warps_per_block = 8

warp_cap     = 64 // 8                = 8
block_cap    = 32
register_cap = 65536 // (256 * 40)    = 65536 // 10240 = 6
shared_cap   = (164 * 1024) // (8 * 1024) = 20

binding = min(8, 32, 6, 20) = 6 -> "registers"
active_blocks = 6
active_warps  = 6 * 8 = 48
occupancy     = 48 / 64 = 0.75
```

75% occupancy, register-limited. To raise occupancy, you'd reduce
the per-thread register count. The lesson: even with a
"reasonable" 40 regs/thread, you're already losing one block.

**Case 3: register-heavy kernel (96 regs)**

```
Inputs: threads=256, regs=96, shared=0
warps_per_block = 8

warp_cap     = 8
block_cap    = 32
register_cap = 65536 // (256 * 96) = 65536 // 24576 = 2
shared_cap   = inf

binding = 2 -> "registers"
active_blocks = 2
active_warps  = 2 * 8 = 16
occupancy     = 16 / 64 = 0.25
```

25% occupancy. Heavy register pressure. The fix is to either
reduce per-thread state or use `--maxrregcount` (which spills, so
trade carefully).

**Case 4: shared-memory-heavy kernel (48 KB)**

```
Inputs: threads=128, regs=32, shared=48 KB
warps_per_block = 4

warp_cap     = 64 // 4               = 16
block_cap    = 32
register_cap = 65536 // (128 * 32)   = 65536 // 4096 = 16
shared_cap   = (164 * 1024) // (48 * 1024) = 3

binding = min(16, 32, 16, 3) = 3 -> "shared_mem"
active_blocks = 3
active_warps  = 3 * 4 = 12
occupancy     = 12 / 64 = 0.1875
```

18.75% occupancy. Shared memory limits it to 3 blocks per SM. The
fix is to reduce shared memory per block (smaller tile, more
register-side computation, or split the kernel into phases).

**Case 5: tiny block (32 threads)**

```
Inputs: threads=32, regs=8, shared=0
warps_per_block = 1

warp_cap     = 64 // 1               = 64
block_cap    = 32                    <-- THIS WINS
register_cap = 65536 // (32 * 8)     = 256
shared_cap   = inf

binding = min(64, 32, 256, inf) = 32 -> "blocks"
active_blocks = 32
active_warps  = 32 * 1 = 32
occupancy     = 32 / 64 = 0.5
```

50% occupancy, **blocks** is the limiter. This is a common
pathology with very small block sizes: you hit the per-SM block
count cap before you exhaust other resources. To raise occupancy:
make the block bigger so each block carries more warps.

**Case 6: threads not a multiple of 32**

```
Inputs: threads=97, regs=32, shared=0
warps_per_block = ceil(97 / 32) = 4

warp_cap     = 64 // 4 = 16
block_cap    = 32
register_cap = 65536 // (97 * 32) = 65536 // 3104 = 21
shared_cap   = inf

binding = min(16, 32, 21, inf) = 16 -> "warps"
active_blocks = 16
active_warps  = 16 * 4 = 64
occupancy     = 64 / 64 = 1.0
```

The 97-thread block uses 4 full warps but only 97 lanes do work
in the last warp (28 lanes are wasted permanently). Full warp
occupancy from the scheduler's perspective, but ~24% of arithmetic
goes to dead lanes.

**Case 7: low resources, normal block**

```
Inputs: threads=128, regs=8, shared=0
warps_per_block = 4

warp_cap     = 64 // 4               = 16
block_cap    = 32
register_cap = 65536 // (128 * 8)    = 65536 // 1024 = 64
shared_cap   = inf

binding = min(16, 32, 64, inf) = 16 -> "warps"
active_blocks = 16
active_warps  = 16 * 4 = 64
occupancy     = 64 / 64 = 1.0
```

Full occupancy, warp-cap limited. Lots of headroom in registers
and blocks; the binding constraint is just that 16 blocks of 4
warps fills the SM's 64-warp ceiling exactly.

### When to use which "fix"

Once you've identified the binding resource, here's the
optimization decision tree:

```
binding = "warps"        -> you're at full occupancy already.
                            Look at instruction-level parallelism,
                            memory access patterns, etc.

binding = "blocks"       -> block is too small. Increase
                            threads_per_block (multiples of 32) so
                            each block carries more warps.

binding = "registers"    -> too much per-thread state.
                            Options:
                              - Split kernel into phases
                              - Recompute instead of cache in regs
                              - Use --maxrregcount (with spill caveat)
                              - Reduce per-thread loop unrolling

binding = "shared_mem"   -> too much shared memory per block.
                            Options:
                              - Smaller tile size
                              - Move some shared-mem data to L2-friendly
                                global access (cached)
                              - Use dynamic shared memory only as needed
```

The diagnosis matters because the *fixes* are completely different.
A register fix doesn't help a shared-mem-bound kernel and vice
versa. Get the diagnosis right and the fix follows.

## Why we don't return Python `math.inf` directly

A reasonable instinct is "for unused resources, return
math.inf and let min() do the work." That works for the cap
calculation but breaks if you return `math.inf` to the caller —
no real chip has infinite blocks per SM. The fix is to clamp
at the end:

```python
active_blocks = min(int(binding_cap), sm.max_blocks_per_sm)
```

For the *normal* case (where binding_cap is a finite resource
cap), `min(int(binding_cap), max_blocks_per_sm)` returns
`binding_cap`. For the pathological case where every resource
is unconstrained (`shared_cap = inf, register_cap = inf,
warp_cap > max_blocks_per_sm`), this returns `max_blocks_per_sm`
as the answer — the right physical bound.

## A real-world calibration

The numbers above are theoretical occupancy — what the calculator
returns. The *achieved* occupancy in a running kernel is reported
by `Nsight Compute` and may be lower due to:

- Grid not large enough to keep all SMs full at once (especially
  at the end of the grid).
- Block-to-SM scheduling skew (some SMs finish blocks earlier).
- Tail effects on irregular workloads (the last wave has fewer
  blocks).

A 100% theoretical occupancy kernel typically achieves 85-95%
achieved occupancy in practice. The gap doesn't bother you for
diagnosis: theoretical occupancy is the ceiling and *that's*
what you tune to raise.

## Related material

- Lecture 1.4 — Warps and occupancy (the full derivation).
- NVIDIA Occupancy Calculator spreadsheet (linked in
  `resources.md`).
- Volkov 2010, "Better Performance at Lower Occupancy."
- CUDA Programming Guide §5.2.3 ("Maximize Utilization") — official
  documentation of these constraints.

## Next exercise

Exercise 4 — Roofline analysis. Given six concrete kernels,
compute arithmetic intensity, classify each as compute-bound or
memory-bound, and plot them on an A100 roofline.
