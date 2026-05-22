"""Autograder for Exercise 1. Run: python check.py

Grades only the bucketed answer (gpu_helps + expected_speedup_bucket).
The reasoning prose is graded by the human reader against the
solutions repo's worked answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sibling starter.py importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent))

try:
    import starter  # type: ignore[import-not-found]
except ImportError as e:
    print(f"FAIL: could not import starter.py ({e})")
    sys.exit(1)


EXPECTED = {
    # Dense GEMM at 8192^3 has AI ~ 341 FLOP/byte, far above the A100
    # ridge point (~9.75). It is the canonical compute-bound GPU win.
    # Practical speedups for FP32 dense GEMM on cuBLAS vs MKL on a
    # modern server CPU land at >100x (8192^3 is ~1.1 TFLOP of work;
    # a 19.5 TFLOPS A100 does it in ~60 ms, vs several seconds on
    # most server CPUs).
    "dense_matmul": {"gpu_helps": True, "bucket": ">100x"},
    # JSON parsing is irregular control flow, tiny arithmetic per byte,
    # serial dependency from one nested element to the next. The GPU
    # has no path to win.
    "json_parse": {"gpu_helps": False, "bucket": "<1x"},
    # 4 MB tensor at ~2 TB/s HBM is ~2 microseconds of memory work.
    # Kernel launch alone is 5-20 us. The CPU does this in <1 ms with
    # warm caches and no kernel-launch tax; the GPU is bottlenecked by
    # launch overhead at this size. Most realistic: 1-2x (GPU might
    # barely win or barely lose). Either way, not a meaningful
    # speedup; the bucketed answer is 1-2x.
    "elementwise_relu": {"gpu_helps": True, "bucket": "1-2x"},
}

VALID_BUCKETS = {"<1x", "1-2x", "2-10x", "10-100x", ">100x"}


def grade() -> int:
    answers = getattr(starter, "ANSWERS", None)
    if not isinstance(answers, dict):
        print("FAIL: ANSWERS dict missing from starter.py")
        return 1

    failures: list[str] = []
    for name, expected in EXPECTED.items():
        a = answers.get(name)
        if not isinstance(a, dict):
            failures.append(f"  {name}: entry missing or not a dict")
            continue
        gh = a.get("gpu_helps")
        bk = a.get("expected_speedup_bucket")
        if gh is ... or bk is ... or gh is None or bk is None:
            failures.append(f"  {name}: not filled in (still ... )")
            continue
        if not isinstance(gh, bool):
            failures.append(f"  {name}: gpu_helps must be bool, got {type(gh).__name__}")
            continue
        if bk not in VALID_BUCKETS:
            failures.append(
                f"  {name}: expected_speedup_bucket must be one of {sorted(VALID_BUCKETS)}, got {bk!r}"
            )
            continue
        if gh != expected["gpu_helps"]:
            failures.append(
                f"  {name}: gpu_helps={gh} but expected {expected['gpu_helps']}"
            )
            continue
        if bk != expected["bucket"]:
            failures.append(
                f"  {name}: expected_speedup_bucket={bk!r} but expected {expected['bucket']!r}"
            )
            continue

    if failures:
        print("FAIL:")
        for f in failures:
            print(f)
        return 1

    print("PASS — all three workloads bucketed correctly.")
    print("(Reasoning prose is not autograded; compare against the solutions repo.)")
    return 0


if __name__ == "__main__":
    sys.exit(grade())
