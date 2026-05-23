"""Autograder for Exercise 5. Run: python check.py

Tests the warp-divergence analytical model on five throughput
cases and one break-even probability case. The reference
implementation is included so the autograder is self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import starter  # type: ignore[import-not-found]
except ImportError as e:
    print(f"FAIL: could not import starter.py ({e})")
    sys.exit(1)


WARP_SIZE = 32


# -----------------------------------------------------------------------------
# Reference implementation (matches what the README walks through).
# -----------------------------------------------------------------------------


def _ref_expected_cycles(cost_a: float, cost_b: float, prob_a: float) -> float:
    all_a = prob_a ** WARP_SIZE
    all_b = (1.0 - prob_a) ** WARP_SIZE
    mixed = 1.0 - all_a - all_b
    if mixed < 0:
        mixed = 0.0  # guard floating point
    return all_a * cost_a + all_b * cost_b + mixed * (cost_a + cost_b)


def _ref_throughput(cost_a: float, cost_b: float, prob_a: float) -> float:
    baseline = max(cost_a, cost_b)
    e = _ref_expected_cycles(cost_a, cost_b, prob_a)
    if e == 0:
        return float("inf")
    return baseline / e


# -----------------------------------------------------------------------------
# Cases.
# -----------------------------------------------------------------------------


# (description, cost_a, cost_b, prob_a, expected_throughput, tol)
THROUGHPUT_CASES = [
    ("all lanes take A (no divergence)", 100.0, 100.0, 1.0, 1.0, 0.02),
    ("all lanes take B (no divergence)", 100.0, 100.0, 0.0, 1.0, 0.02),
    ("balanced 50/50 (canonical 2x slowdown)", 100.0, 100.0, 0.5, 0.5, 0.02),
    ("skewed 90/10 (still mostly divergent)", 100.0, 50.0, 0.9, None, 0.05),
    ("unequal costs, skewed", 200.0, 50.0, 0.3, None, 0.05),
]


def _check_throughput_cases() -> list[str]:
    failures: list[str] = []
    for desc, cost_a, cost_b, prob_a, expected, tol in THROUGHPUT_CASES:
        if expected is None:
            expected = _ref_throughput(cost_a, cost_b, prob_a)
        try:
            got = starter.throughput_ratio(cost_a, cost_b, prob_a)
        except NotImplementedError:
            failures.append(f"  {desc}: throughput_ratio not implemented")
            continue
        except Exception as e:
            failures.append(f"  {desc}: raised {type(e).__name__}: {e}")
            continue
        if abs(got - expected) > tol:
            failures.append(
                f"  {desc}: throughput_ratio({cost_a}, {cost_b}, {prob_a}) "
                f"= {got:.4f}, expected ~{expected:.4f} (tol {tol})"
            )
    return failures


def _check_expected_cycles_directly() -> list[str]:
    """Spot-check expected_cycles_per_warp against the reference."""
    failures: list[str] = []
    cases = [
        (100.0, 100.0, 0.5),
        (100.0, 50.0, 0.9),
        (200.0, 50.0, 0.3),
        (100.0, 100.0, 1.0),
        (100.0, 100.0, 0.0),
    ]
    for cost_a, cost_b, prob_a in cases:
        try:
            got = starter.expected_cycles_per_warp(cost_a, cost_b, prob_a)
        except NotImplementedError:
            failures.append(
                f"  expected_cycles_per_warp({cost_a}, {cost_b}, {prob_a}): not implemented"
            )
            continue
        except Exception as e:
            failures.append(
                f"  expected_cycles_per_warp({cost_a}, {cost_b}, {prob_a}): raised "
                f"{type(e).__name__}: {e}"
            )
            continue
        expected = _ref_expected_cycles(cost_a, cost_b, prob_a)
        # Use relative tolerance because expected values vary widely.
        if expected == 0:
            ok = abs(got) < 1e-9
        else:
            ok = abs(got - expected) / max(abs(expected), 1.0) < 0.01
        if not ok:
            failures.append(
                f"  expected_cycles_per_warp({cost_a}, {cost_b}, {prob_a}) "
                f"= {got:.4f}, expected ~{expected:.4f}"
            )
    return failures


def _check_break_even() -> list[str]:
    """Break-even should fall in the 'guaranteed divergent' tail."""
    failures: list[str] = []
    try:
        p = starter.break_even_probability(100.0, 100.0)
    except NotImplementedError:
        return ["  break_even_probability: not implemented"]
    except Exception as e:
        return [f"  break_even_probability: raised {type(e).__name__}: {e}"]

    if not (0.0 < p <= 0.5):
        failures.append(
            f"  break_even_probability(100, 100) = {p:.4f}, must be in (0, 0.5]"
        )
        return failures

    # E(p) asymptotes from below to (cost_a + cost_b) very quickly — by p~0.15
    # it's already within 0.3% of the target. The break-even depends on what
    # `tolerance` the student picks, so we accept any p in a generous window
    # where E(p) is close to target.
    e = _ref_expected_cycles(100.0, 100.0, p)
    if not (e > 195.0):
        failures.append(
            f"  break_even_probability(100, 100) = {p:.4f} gives expected_cycles "
            f"{e:.4f}; the break-even should be in the regime where E(p) is "
            f"close to cost_a + cost_b = 200 (i.e. p large enough that "
            f"divergence is near-guaranteed)"
        )

    return failures


def grade() -> int:
    failures: list[str] = []
    failures += _check_expected_cycles_directly()
    failures += _check_throughput_cases()
    failures += _check_break_even()

    if failures:
        print("FAIL:")
        for f in failures:
            print(f)
        return 1

    print(
        "PASS — divergence model agrees with reference on 5 throughput cases "
        "+ direct expected-cycles spot checks + break-even probability."
    )
    return 0


if __name__ == "__main__":
    sys.exit(grade())
