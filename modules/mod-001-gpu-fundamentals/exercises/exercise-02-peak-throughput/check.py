"""Autograder for Exercise 2. Run: python check.py

Checks (1) all spec fields are filled (not Ellipsis), (2) the two
formula functions are implemented, and (3) computed peaks land within
5% of NVIDIA's published reference values.
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


TOLERANCE = 0.05  # 5%


# Reference peak values from NVIDIA's published spec sheets.
# Sources cited in modules/mod-001-gpu-fundamentals/resources.md.
REFERENCE = {
    "A100 SXM4 80GB": {"fp32_tflops": 19.5, "bandwidth_gbs": 2039.0},
    "H100 SXM5 80GB": {"fp32_tflops": 67.0, "bandwidth_gbs": 3350.0},
    "RTX 4090": {"fp32_tflops": 82.6, "bandwidth_gbs": 1008.0},
}


def _has_ellipsis(obj: object) -> bool:
    """Recursively detect un-filled `...` values."""
    if obj is ...:
        return True
    if hasattr(obj, "__dataclass_fields__"):
        return any(_has_ellipsis(getattr(obj, f)) for f in obj.__dataclass_fields__)
    return False


def _within(actual: float, expected: float, tol: float = TOLERANCE) -> bool:
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / expected <= tol


def grade() -> int:
    failures: list[str] = []

    specs = {
        "A100 SXM4 80GB": getattr(starter, "A100_SXM4_80GB", None),
        "H100 SXM5 80GB": getattr(starter, "H100_SXM5_80GB", None),
        "RTX 4090": getattr(starter, "RTX_4090", None),
    }

    for name, spec in specs.items():
        if spec is None:
            failures.append(f"  {name}: spec object missing from starter.py")
            continue
        if _has_ellipsis(spec):
            failures.append(f"  {name}: spec has unfilled `...` fields")
            continue

        ref = REFERENCE[name]

        try:
            fp32 = starter.peak_fp32_tflops(spec)
        except NotImplementedError:
            failures.append(f"  {name}: peak_fp32_tflops not implemented")
            continue
        if not _within(fp32, ref["fp32_tflops"]):
            failures.append(
                f"  {name}: FP32 = {fp32:.1f} TFLOPS, expected ~{ref['fp32_tflops']:.1f} (within 5%)"
            )

        try:
            bw = starter.peak_memory_bandwidth_gbs(spec)
        except NotImplementedError:
            failures.append(f"  {name}: peak_memory_bandwidth_gbs not implemented")
            continue
        if not _within(bw, ref["bandwidth_gbs"]):
            failures.append(
                f"  {name}: BW = {bw:.0f} GB/s, expected ~{ref['bandwidth_gbs']:.0f} (within 5%)"
            )

    if failures:
        print("FAIL:")
        for f in failures:
            print(f)
        return 1

    print("PASS — all three GPUs match the published peak values within 5%.")
    return 0


if __name__ == "__main__":
    sys.exit(grade())
