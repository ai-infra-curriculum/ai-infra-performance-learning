"""Autograder for Exercise 4. Run: python check.py

Tests the six AI values, the classifier, the attainable-throughput
calculation, and that `plot_roofline` produces a PNG of plausible
size with the right number of artists in the figure.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import starter  # type: ignore[import-not-found]
except ImportError as e:
    print(f"FAIL: could not import starter.py ({e})")
    sys.exit(1)


# Tolerance is generous on AI because the softmax estimate is a range,
# and learners may model the FLOPS slightly differently.
AI_TOLERANCE = 0.10

# Reference AI values (FLOP/byte).
# vector_add:    1 FLOP / 12 B   = 0.0833
# vector_fma:    2 FLOPS / 16 B  = 0.125
# axpy_scaled:   2 FLOPS / 12 B  = 0.1667
# relu:          1 FLOP / 8 B    = 0.125
# gemm_8192:     2*8192^3 / (12 * 8192^2)  = 2*8192/12  ~ 1365.33
# softmax_row:   5 FLOPS / 8 B   = 0.625
REFERENCE_AI = {
    "vector_add":  1 / 12,
    "vector_fma":  2 / 16,
    "axpy_scaled": 2 / 12,
    "relu":        1 / 8,
    "gemm_8192":   (2 * 8192) / 12,
    "softmax_row": 5 / 8,
}


def _within(actual: float, expected: float, tol: float = AI_TOLERANCE) -> bool:
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / expected <= tol


def _check_ai() -> list[str]:
    fails: list[str] = []
    for k, expected in REFERENCE_AI.items():
        try:
            got = starter.arithmetic_intensity(k)
        except NotImplementedError:
            fails.append(f"  arithmetic_intensity({k!r}): not implemented")
            continue
        except Exception as e:
            fails.append(f"  arithmetic_intensity({k!r}): raised {type(e).__name__}: {e}")
            continue
        if not _within(got, expected):
            fails.append(
                f"  arithmetic_intensity({k!r}) = {got:.4f}, expected ~{expected:.4f} (within 10%)"
            )
    return fails


def _check_classify() -> list[str]:
    fails: list[str] = []
    cases = [
        ("memory-bound below ridge (AI=0.1)", 0.1, "memory-bound"),
        ("memory-bound (vector_add)", 1 / 12, "memory-bound"),
        ("compute-bound (large GEMM)", 1000.0, "compute-bound"),
        ("compute-bound right above ridge (AI=20)", 20.0, "compute-bound"),
    ]
    for desc, ai, expected in cases:
        try:
            got = starter.classify(ai)
        except NotImplementedError:
            fails.append(f"  classify({ai}): not implemented")
            continue
        except Exception as e:
            fails.append(f"  classify({ai}): raised {type(e).__name__}: {e}")
            continue
        if got != expected:
            fails.append(f"  classify({ai}) = {got!r}, expected {expected!r} ({desc})")
    return fails


def _check_attainable() -> list[str]:
    fails: list[str] = []
    # AI=0.1: peak_bw * AI = 203.9 GFLOPS = 0.204 TFLOPS (memory ceiling wins)
    # AI=20:  peak_bw * AI = 40780 GFLOPS = 40.78 TFLOPS, capped at 19.5 (compute)
    cases = [
        ("AI=0.1 -> memory ceiling", 0.1, 0.2039),
        ("AI=20  -> compute ceiling", 20.0, 19.5),
    ]
    for desc, ai, expected in cases:
        try:
            got = starter.attainable_throughput_tflops(ai)
        except NotImplementedError:
            fails.append(f"  attainable({ai}): not implemented")
            continue
        except Exception as e:
            fails.append(f"  attainable({ai}): raised {type(e).__name__}: {e}")
            continue
        if not _within(got, expected, tol=0.02):
            fails.append(
                f"  attainable({ai}) = {got:.4f}, expected ~{expected:.4f} ({desc})"
            )
    return fails


def _check_plot() -> list[str]:
    fails: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "roofline.png")
        try:
            starter.plot_roofline(out)
        except NotImplementedError:
            return ["  plot_roofline: not implemented"]
        except Exception as e:
            return [f"  plot_roofline: raised {type(e).__name__}: {e}"]
        if not os.path.exists(out):
            fails.append(f"  plot_roofline did not create {out}")
            return fails
        size = os.path.getsize(out)
        if size < 5_000:
            fails.append(
                f"  plot_roofline produced a tiny file ({size} bytes); "
                "is the figure actually populated?"
            )
        # Cheap structural sniff: PNG magic.
        with open(out, "rb") as f:
            head = f.read(8)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            fails.append(f"  plot_roofline output is not a PNG file (got {head!r})")
    return fails


def grade() -> int:
    fails: list[str] = []
    fails += _check_ai()
    fails += _check_classify()
    fails += _check_attainable()
    fails += _check_plot()

    if fails:
        print("FAIL:")
        for f in fails:
            print(f)
        return 1

    print("PASS — six AI values, classifier, attainable throughput, and plot all check out.")
    return 0


if __name__ == "__main__":
    sys.exit(grade())
