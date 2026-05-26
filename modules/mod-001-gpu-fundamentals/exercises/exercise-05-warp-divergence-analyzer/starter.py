"""Exercise 5 starter — warp divergence cost analyzer.

Implement three functions modeling the cost of a divergent branch
inside a CUDA warp under the SIMT execution rule:

    expected_cycles_per_warp(cost_a, cost_b, prob_a)
        The expected number of cycles a warp spends on the branch,
        accounting for the per-warp divergence rule (if any lane
        takes A and any lane takes B, the warp pays cost_a + cost_b;
        otherwise it pays only the cost of the branch all lanes take).

    throughput_ratio(cost_a, cost_b, prob_a)
        The effective throughput relative to the non-divergent
        baseline (max(cost_a, cost_b)). 1.0 = no penalty;
        0.5 = 2x slowdown.

    break_even_probability(cost_a, cost_b, tolerance=1e-4)
        The smallest p (in [0, 0.5]) at which expected_cycles_per_warp
        equals cost_a + cost_b within `tolerance`. Below this p,
        branching might still beat predication. Above it, predication
        wins.

See README.md for the full derivation.

Run:

    python check.py
"""

from __future__ import annotations

WARP_SIZE = 32


def expected_cycles_per_warp(cost_a: float, cost_b: float, prob_a: float) -> float:
    """Expected per-warp cost of a two-sided divergent branch.

    Inputs:
      cost_a:  cycles to execute branch A (one lane taking A; warp
               pays this cost when at least one lane takes A).
      cost_b:  cycles to execute branch B.
      prob_a:  probability that a single lane takes branch A.

    Returns:
      Expected cycles per warp under SIMT divergence.

    See README.md "The model" for the formula.
    """
    raise NotImplementedError


def throughput_ratio(cost_a: float, cost_b: float, prob_a: float) -> float:
    """Effective throughput relative to non-divergent baseline.

    Baseline = max(cost_a, cost_b).
    Throughput = baseline / expected_cycles_per_warp(...).

    Returns a float in (0, 1]. 1.0 means no divergence penalty.
    """
    raise NotImplementedError


def break_even_probability(
    cost_a: float, cost_b: float, tolerance: float = 1e-4
) -> float:
    """Probability at which branching equals always-predicate cost.

    Solves expected_cycles_per_warp(cost_a, cost_b, p) == cost_a + cost_b
    for the smallest p in [0, 0.5] (within `tolerance`).

    Below this p, the warp is usually coherent and branching wins.
    Above this p, divergence is near-guaranteed and you should
    predicate (always pay cost_a + cost_b) instead.

    Use a binary search.
    """
    raise NotImplementedError
