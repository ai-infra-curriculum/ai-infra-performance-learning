# Exercise 5 — Warp divergence analyzer

> **Targets learning objectives:** 1, 5
> **Time:** ~45 min
> **Requires:** Python 3.10+. No GPU needed.
> **References:** lecture-notes/04-warps-and-occupancy.md ("Branch divergence and the warp tax" section).

## What you'll do

You will build a small analytical model that predicts the cost of
branch divergence inside a CUDA warp. Given a kernel structure
described as `(cost_of_branch_A, cost_of_branch_B, probability_lane_takes_A)`,
your model computes:

- The **expected cycles per warp** under the SIMT execution rule.
- The **effective throughput** as a fraction of the non-divergent
  case.
- The **break-even probability** at which "compute both branches
  unconditionally" beats "branch and diverge."

This is exactly the analysis you'd do when deciding whether to
restructure a kernel that has data-dependent control flow.

## Why this skill matters

Branch divergence is the second-most-common reason GPU kernels
underperform (the first is bad memory access patterns). When you
profile and see "Warp Execution Efficiency: 50%," the question
is *what to do about it*, and the answer depends on the
arithmetic: is rewriting the branch into a predicate cheaper than
paying the divergence tax?

Concrete situations:

- **Histogram with skewed bins.** Some bins are far more popular
  than others. Lanes in the same warp that hit different bins
  diverge. Should you sort the data first, accept the
  divergence, or restructure into a different algorithm?
- **Skiplist / hashmap probe.** Each lane chases a different
  chain length. Heavy divergence. Often you switch to a
  parallel-friendly data structure entirely.
- **Conditional masking in attention.** Causal masks; padding
  masks. Lanes that handle "masked" positions take a different
  path. Almost always cheaper to compute everything and apply a
  multiplicative mask at the end — but the model is what tells
  you when.

The model in this exercise is a first-order approximation, but
it's the right *shape* of model. Production tuning uses the same
idea with more parameters (instruction-level parallelism, memory
stalls, etc.).

## The model

The SIMT execution rule (Lecture 4):

> A warp of 32 lanes executes one instruction at a time across
> all lanes. If lanes diverge on a branch, the warp executes
> *both* branches serially, with predication masking off lanes
> that aren't on the current side.

For a single divergent branch with two sides A and B:

```
prob_lane_takes_A: probability that a single lane goes into branch A
                   (the other 1 - p goes into branch B)

For each lane the warp encounters the branch and:
  - executes branch A (cost: cost_A cycles) - lanes on B sit idle
  - executes branch B (cost: cost_B cycles) - lanes on A sit idle

The warp's cost is the *sum*, not the max:
  cycles_diverged(p) = cost_A + cost_B    (if any lane takes A AND any lane takes B)

Or in the no-divergence case:
  cycles_no_div(p)   = cost_A             (if all 32 lanes take A; never B)
                     = cost_B             (if all 32 lanes take B)
```

The interesting cases:

- **All 32 lanes take A.** Probability: p^32. Cost: cost_A.
- **All 32 lanes take B.** Probability: (1-p)^32. Cost: cost_B.
- **Mixed: at least one A and at least one B.** Probability:
  1 - p^32 - (1-p)^32. Cost: cost_A + cost_B.

So the expected cost per warp is:

```
E[cycles] = p^32       * cost_A
          + (1-p)^32   * cost_B
          + (1 - p^32 - (1-p)^32) * (cost_A + cost_B)
```

For most "interesting" p (i.e., p not very close to 0 or 1), the
middle term dominates: the warp diverges with probability ~1,
and the cost is cost_A + cost_B.

### Effective throughput

The "non-divergent baseline" is the cost you'd pay if all lanes
always took the more expensive branch:

```
baseline_cycles = max(cost_A, cost_B)
```

Effective throughput is:

```
throughput(p) = baseline_cycles / E[cycles(p)]
```

A value of 1.0 means "no penalty." A value of 0.5 means "you're
spending twice as long per warp as the baseline" (typical for
a balanced 50/50 divergent branch with equal-cost sides).

### The "predicate instead of branch" break-even

If both branches are cheap and small, you can refactor:

```cpp
// Branched (diverges if p is mixed):
if (cond) {
    x = a_path();
} else {
    x = b_path();
}

// Predicated (no divergence, but always computes both):
x_a = a_path();
x_b = b_path();
x   = cond ? x_a : x_b;     // hardware-level predicate, no branch
```

The predicated version has fixed cost `cost_A + cost_B` always.
The branched version has expected cost from the formula above.

The branched version is cheaper *only when* p is close to 0 or 1
(so the warp usually takes only one side). The predicated
version is cheaper *anywhere in the middle* (because you don't
pay the "select" overhead twice, just once).

The break-even probability — where `E[cycles_branched(p)] ==
cost_A + cost_B` — can be solved numerically. For symmetric
costs (`cost_A == cost_B`) it's at p where `p^32 + (1-p)^32`
balances out the cost difference.

## What to submit

Edit `starter.py`. Implement three functions:

```python
def expected_cycles_per_warp(cost_a: float, cost_b: float, prob_a: float) -> float: ...
def throughput_ratio(cost_a: float, cost_b: float, prob_a: float) -> float: ...
def break_even_probability(cost_a: float, cost_b: float, tolerance: float = 1e-4) -> float: ...
```

Then run:

```bash
python check.py
```

The autograder runs 5 test cases with known answers (single
warp, balanced 50/50 split, heavily skewed, equal-cost branches,
unequal-cost branches).

## Hints

### Implementing `expected_cycles_per_warp`

```python
def expected_cycles_per_warp(cost_a, cost_b, prob_a):
    if not (0.0 <= prob_a <= 1.0):
        raise ValueError(f"prob_a must be in [0,1], got {prob_a}")
    if cost_a < 0 or cost_b < 0:
        raise ValueError("costs must be non-negative")

    all_a = prob_a ** 32
    all_b = (1 - prob_a) ** 32
    mixed = 1 - all_a - all_b

    return all_a * cost_a + all_b * cost_b + mixed * (cost_a + cost_b)
```

### Implementing `throughput_ratio`

```python
def throughput_ratio(cost_a, cost_b, prob_a):
    baseline = max(cost_a, cost_b)
    expected = expected_cycles_per_warp(cost_a, cost_b, prob_a)
    if expected == 0:
        return float("inf")
    return baseline / expected
```

### Implementing `break_even_probability`

Use a binary search between 0 and 0.5 (by symmetry the break-even
is the same on both sides; we find one). The function we're
solving is:

```
expected_cycles(p) - (cost_a + cost_b) = 0
```

When p is very small or very large, expected_cycles is close to
max(cost_a, cost_b) (the warp mostly takes one side). When p is
moderate (say 0.05–0.5), expected_cycles is close to
cost_a + cost_b (the warp almost always diverges).

```python
def break_even_probability(cost_a, cost_b, tolerance=1e-4):
    target = cost_a + cost_b
    lo, hi = 0.0, 0.5
    # f(lo) = max(cost_a, cost_b) - target  < 0
    # f(hi) = ~ target - target            ≈ 0 from below
    # We're looking for the smallest p where E[cycles(p)] == target,
    # i.e. where divergence has just become guaranteed.
    while hi - lo > tolerance:
        mid = (lo + hi) / 2
        e = expected_cycles_per_warp(cost_a, cost_b, mid)
        if e < target - tolerance:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
```

Note: the autograder tolerates ±0.02 in the break-even probability,
because the function is very flat near the answer (lots of solutions
within a small window).

### Why the formula has `p^32`

A warp has 32 lanes. For all 32 to take the A branch, each must
take it independently — probability `p^32`. With p=0.9, that's
~0.038 (very rare!). With p=0.99, that's ~0.725 (likely!). The
exponent shapes the model dramatically: even small fractions of
"wrong" lanes guarantee divergence.

### A worked numerical example

Take `cost_A = 100, cost_B = 100, prob_a = 0.5` (perfectly
balanced):

```
all_a = 0.5^32 ≈ 2.33e-10  (essentially zero)
all_b = 0.5^32 ≈ 2.33e-10
mixed = ~1.0

E[cycles] ≈ 1.0 * (100 + 100) = 200

baseline = max(100, 100) = 100
throughput = 100 / 200 = 0.5    (50% efficiency)
```

So a 50/50 split with equal-cost branches gives the canonical "2×
slowdown" you read about everywhere. The model recovers it.

Now `prob_a = 0.99`:

```
all_a = 0.99^32 ≈ 0.7250
all_b = 0.01^32 ≈ 0 (essentially zero)
mixed = ~0.275

E[cycles] = 0.7250 * 100 + 0 + 0.275 * 200
          = 72.5 + 55
          = 127.5

throughput = 100 / 127.5 ≈ 0.78    (78% efficiency)
```

So even with 99% of lanes taking the "true" branch, you still
pay 27% in throughput from the 27.5% chance of divergence.

Now `prob_a = 0.999`:

```
all_a = 0.999^32 ≈ 0.9684
mixed = ~0.0316

E[cycles] = 0.9684 * 100 + 0.0316 * 200
          = 96.84 + 6.32
          ≈ 103.2

throughput = 100 / 103.2 ≈ 0.97    (97% efficiency)
```

Three nines is enough that the warp is almost always coherent.

## Common pitfalls

### 1. Computing per-lane instead of per-warp

The divergence rule is *per warp*, not per lane. If only one
lane takes a different branch, the *whole warp* pays the cost.
That's why `p^32` matters — you need every single lane to agree
for the warp to be coherent.

### 2. Confusing "cost" with "throughput"

Higher expected_cycles = lower throughput. Don't get the
direction wrong. baseline ÷ expected, not the other way.

### 3. Treating p=0 or p=1 as edge cases

`0^32 = 0` and `1^32 = 1` work fine in floating point. The model
returns cost_b at p=0 and cost_a at p=1, no special-casing needed.

### 4. Forgetting the mixed term

```
WRONG: E[cycles] = p * cost_A + (1-p) * cost_B   # treats lanes as independent
RIGHT: E[cycles] = p^32 * cost_A + (1-p)^32 * cost_B
                  + (1 - p^32 - (1-p)^32) * (cost_A + cost_B)
```

The wrong formula treats each lane as independent — which gives
you a "linear combination" that doesn't reflect the per-warp
divergence rule. The right formula accounts for the warp being
the SIMT unit.

### 5. Numerical issues near p ≈ 0.5

`0.5^32 = 2.328e-10`. Floating point handles this fine, but
don't be surprised that `all_a + all_b ≈ 4.6e-10`, essentially
zero. Don't try to "simplify" the formula by dropping these
terms — keep them for general p, even if they vanish at the
midpoint.

## Grading rubric

The autograder runs 5 cases:

| Case | cost_A | cost_B | prob_a | Expected throughput | Tolerance |
|---|---|---|---|---|---|
| All A (no divergence) | 100 | 100 | 1.0 | 1.0 | 0.02 |
| All B (no divergence) | 100 | 100 | 0.0 | 1.0 | 0.02 |
| Balanced 50/50 | 100 | 100 | 0.5 | 0.5 | 0.02 |
| Skewed 90/10 | 100 | 50 | 0.9 | ~0.50 to ~0.55 | 0.05 |
| Unequal costs | 200 | 50 | 0.3 | ~0.80 | 0.05 |

Plus one call to `break_even_probability(100, 100)` expecting a
result between 0.05 and 0.20 (anywhere in the "warp is
guaranteed to be divergent" tail).

If all checks pass, prints `PASS`.

## What "right" looks like

Your model should give results in line with the worked examples
above. After this exercise you should be able to:

- Quote "a balanced divergent branch is a 2× slowdown" with the
  math behind it.
- Predict the cost of "the masked positions in attention" without
  benchmarking.
- Decide between branching and predication for a specific kernel
  by computing the break-even probability.

## Extended worked examples

### Case A: causal attention mask

In a transformer's attention layer, the causal mask sets positions
above the diagonal to -infinity before softmax. The kernel
typically has:

```cuda
if (j > i) {  // future position
    score = -INFINITY;
} else {      // past or current
    score = q[i] . k[j];   // expensive dot product
}
```

For a sequence of length L:

- Within a warp, lanes typically span 32 consecutive `j` values
  for a given `i`. So the branch `j > i` holds true for *some*
  lanes and false for others — divergence within the warp.
- The "expensive" branch (compute the dot product) costs ~100
  cycles. The "cheap" branch (just write -INFINITY) costs ~1
  cycle.

```
cost_a = 1     (cheap: write -INFINITY)
cost_b = 100   (expensive: dot product)
prob_a = (L - i) / L   (varies across i)
```

For i = L/2 (middle of the sequence), prob_a = 0.5. Equal split,
both branches taken by some lanes:

```
E[cycles] ≈ 0 + 0 + 1.0 * (1 + 100) = 101 cycles
baseline = max(1, 100) = 100 cycles
throughput = 100 / 101 ≈ 99%
```

The cheap branch is so cheap that even when half the warp takes
it, the warp barely slows down. The expensive branch dominates
the time regardless.

For i = 0 (start of sequence), prob_a ≈ 1.0:

```
all_a ≈ 1, all_b ≈ 0, mixed ≈ 0
E[cycles] ≈ 1.0 * 1 + 0 + 0 = 1 cycle
baseline = 100
throughput = 100 / 1 = 100x (much *better* than baseline)
```

Wait — that's not divergence "cost," that's the warp getting away
with not doing the expensive branch at all because all lanes are
on the cheap side. The "baseline" of `max(cost_a, cost_b) = 100`
is the cost of always taking the expensive branch; the warp at
i=0 just doesn't do that work. Good for it.

The takeaway: causal-mask divergence is mostly free because the
masked-out branch is essentially zero cost. The model captures
this correctly.

### Case B: per-lane lookup in a small table

```cuda
__shared__ float table[16];   // initialized earlier

__global__ void k(const int* indices, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int t = indices[idx];   // varies per lane
        if (t == 0) {
            out[idx] = table[0] * 2.0f;
        } else if (t == 1) {
            out[idx] = table[1] * 3.0f;
        } else if (t == 2) {
            // ... 16 branches total ...
        }
    }
}
```

A 16-way switch inside the kernel. If `indices` is random, every
warp probably has lanes on all 16 branches — full divergence,
each branch executes serially.

For 16 branches each costing 10 cycles:

```
With divergence: warp executes all 16 sequentially = 16 * 10 = 160 cycles
Without divergence (one branch wins): 10 cycles
Penalty: 16x.
```

Same kernel, ideal data layout (indices sorted so all 32 lanes of
each warp see the same index):

```
0 cycles wasted on divergence; runs at 10 cycles per warp.
```

The 16x speedup from sorting the data is *huge*. Sometimes the
right thing is to sort or bucket data on the CPU before launching
the kernel.

This pattern is exactly what NVIDIA's "warp specialization"
techniques are about — assigning warps to specific roles so
within-warp divergence vanishes.

### Case C: bit-twiddling without divergence

You want each lane to compute `lane_id < 16 ? x : y`. Naive:

```cuda
float r;
if (threadIdx.x % 32 < 16) {
    r = x;
} else {
    r = y;
}
```

Divergent (half the warp takes each side). Predicated rewrite:

```cuda
float r = (threadIdx.x % 32 < 16) ? x : y;
```

The ternary operator typically compiles to a `selp` (select
predicate) instruction — zero divergence, one cycle. Same logical
result. This is the "compute both, mask the result" trick.

Even better, if `x` and `y` are themselves cheap to compute, the
predicated version is essentially free relative to the branched
version.

The model in this exercise predicts the **branched** cost. If
you're refactoring to use `selp` (predication), the cost is just
the union of both branches, no divergence overhead.

## When the simple model breaks down

The model in this exercise is first-order. Real kernels have
complications:

- **Nested divergence.** An `if` inside an `if` compounds. The
  model handles two branches; n-way nesting needs the recursive
  formula. (Practically: nesting beyond 2 is rare; restructure
  if you see it.)
- **Memory stalls during divergence.** If the divergent branches
  also issue memory loads, the warp scheduler can hide some
  latency by issuing other warps. The model assumes no latency
  hiding; real throughput is sometimes better.
- **Independent thread scheduling (Volta+).** Threads in a warp
  can technically run independent program counters; this rarely
  helps for typical divergent code but enables some patterns
  (especially with `__syncwarp`) that don't fit the simple
  model.
- **Warp specialization with `__shfl_sync`.** Some kernels
  intentionally split a warp into roles and use shuffles for
  coordination. The "divergence" looks high on the surface but
  the warp is doing structured work; the cost model doesn't
  fully apply.

For Module 1's purposes, the simple model is good enough. Real
production tuning uses the same intuition with more knobs.

## Related material

- Lecture 1.4 — Branch divergence and the warp tax.
- NVIDIA Nsight Compute "Warp State Statistics" — the profiler
  equivalent of this model.
- CUDA Programming Guide §5.3.1 — "Control Flow Instructions."
- "Independent Thread Scheduling" — Volta-and-later relaxation
  of the strict warp-lockstep model.

## Next exercise

Exercise 6 — Tensor-core throughput prediction. Given a GEMM
size and a target GPU, predict whether the kernel will be
compute-bound or memory-bound on tensor cores, and what the
realistic TFLOPS will be.
