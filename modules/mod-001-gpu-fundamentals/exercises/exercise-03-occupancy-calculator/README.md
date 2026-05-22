# Exercise 3 — Build an occupancy calculator from scratch

> **Targets learning objectives:** 3, 5
> **Time:** ~60 min
> **Requires:** Python 3.10+. No GPU needed.

## What you'll do

You will reimplement NVIDIA's occupancy calculation. Given a kernel
launch configuration (block size, registers per thread, shared memory
per block) and an SM resource budget (max threads, max blocks, register
file size, shared memory size), your function returns the *achievable*
occupancy as a fraction of the theoretical maximum.

You have used the NVIDIA Occupancy Calculator spreadsheet, and you may
have used `cudaOccupancyMaxActiveBlocksPerMultiprocessor` in code. This
exercise is the math behind both. After it, you will be able to answer
"is this kernel register-limited or shared-memory-limited?" in your
head from the launch line and the resource sheet.

## The calculation

For one SM and one kernel:

```
threads_per_block_warps     = ceil(threads_per_block / 32)

# Each of three resources caps how many blocks can run concurrently on
# one SM. Take the smallest cap; that is the actual blocks-per-SM.
warp_cap     = floor(max_warps_per_sm / threads_per_block_warps)
block_cap    = max_blocks_per_sm
register_cap = floor(register_file_size / (threads_per_block * registers_per_thread))
shared_cap   = floor(shared_mem_per_sm  / shared_mem_per_block)

active_blocks_per_sm = min(warp_cap, block_cap, register_cap, shared_cap)
active_warps_per_sm  = active_blocks_per_sm * threads_per_block_warps
occupancy            = active_warps_per_sm / max_warps_per_sm
```

A few edge cases:

- If `shared_mem_per_block == 0`, the shared-memory cap is infinity
  (there is no shared-memory constraint).
- If `registers_per_thread == 0`, similarly the register cap is
  infinity (a kernel using zero registers does not exist, but the
  math should handle it cleanly).
- A `threads_per_block` not a multiple of 32 still rounds *up* to a
  whole warp; the leftover lanes are wasted but a warp is the unit of
  scheduling.
- Returned occupancy is a fraction in `[0.0, 1.0]`.

Your function should also report the *binding* resource — the one whose
cap is the smallest — so the caller can answer "what should I optimize?"
without rerunning the calculation.

## What to submit

Edit `starter.py`. Implement:

```python
def occupancy(launch: LaunchConfig, sm: SMSpec) -> OccupancyResult:
    ...
```

Then run:

```bash
python check.py
```

The autograder tests your implementation against a battery of cases
including the four shown in the lecture (FP32 vector add, sgemm tile,
register-heavy kernel, shared-memory-heavy kernel) plus edge cases.

## Hints

- Write the three "caps" as separate local variables. Then take the
  min and report which one was binding (use a tiny dict mapping
  resource name to cap).
- The "infinity" for empty resources is cleanest as
  `math.inf` plus a manual cap of `max_blocks_per_sm` so you do not
  return infinity to the caller.
- Test your `ceil(threads / 32)` on `threads=33`: should be 2.

## What "right" looks like

Your function should agree with NVIDIA's occupancy calculator on every
test case the autograder runs. If it does, you can throw away the
spreadsheet — you have the math.
