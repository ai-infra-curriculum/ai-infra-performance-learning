"""Autograder for Exercise 6. Run: python check.py

Tests gemm_arithmetic_intensity, classify_gemm, and predict_tflops
against five GEMM cases that exercise both compute-bound and
memory-bound regimes on A100 and H100.
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


# Reference spec values (must match starter constants).
A100_TC = 312.0
A100_BW = 2039.0
H100_TC = 989.0
H100_BW = 3350.0
DEFAULT_UTIL = 0.85


def _ref_ai(M, N, K, bpe):
    return (2 * M * N * K) / (bpe * (M * K + K * N + M * N))


def _ref_predict(M, N, K, bpe, gpu, util=DEFAULT_UTIL):
    if gpu == "A100":
        peak_t, peak_bw = A100_TC, A100_BW
    elif gpu == "H100":
        peak_t, peak_bw = H100_TC, H100_BW
    else:
        raise ValueError(gpu)
    ai = _ref_ai(M, N, K, bpe)
    compute_gf = peak_t * 1000
    memory_gf = peak_bw * ai
    attain_gf = min(compute_gf, memory_gf)
    return util * attain_gf / 1000


def _ref_classify(M, N, K, bpe, gpu):
    if gpu == "A100":
        peak_t, peak_bw = A100_TC, A100_BW
    elif gpu == "H100":
        peak_t, peak_bw = H100_TC, H100_BW
    else:
        raise ValueError(gpu)
    ai = _ref_ai(M, N, K, bpe)
    ridge = (peak_t * 1000) / peak_bw
    return "memory-bound" if ai < ridge else "compute-bound"


# (description, M, N, K, bpe, gpu, expected_class, predict_tolerance)
CASES = [
    ("4096^3 FP16 on A100 (compute-bound)", 4096, 4096, 4096, 2, "A100", "compute-bound", 0.10),
    ("4096^3 FP16 on H100 (compute-bound)", 4096, 4096, 4096, 2, "H100", "compute-bound", 0.10),
    ("tiny 32^3 FP16 on A100 (memory-bound)", 32, 32, 32, 2, "A100", "memory-bound", 0.10),
    ("tiny 32^3 FP16 on H100 (memory-bound)", 32, 32, 32, 2, "H100", "memory-bound", 0.10),
    ("non-square (4096,4096,512) FP16 on H100", 4096, 4096, 512, 2, "H100", "compute-bound", 0.10),
]


def _check_ai_directly() -> list[str]:
    """Direct spot-checks on AI computation."""
    failures: list[str] = []
    spots = [
        (4096, 4096, 4096, 2),
        (4096, 4096, 4096, 4),    # FP32 same shape
        (32, 32, 32, 2),
        (4096, 4096, 512, 2),
        (1024, 1024, 1024, 2),
    ]
    for M, N, K, bpe in spots:
        try:
            got = starter.gemm_arithmetic_intensity(M, N, K, bpe)
        except NotImplementedError:
            failures.append(
                f"  gemm_arithmetic_intensity({M},{N},{K},{bpe}): not implemented"
            )
            continue
        except Exception as e:
            failures.append(
                f"  gemm_arithmetic_intensity({M},{N},{K},{bpe}): "
                f"raised {type(e).__name__}: {e}"
            )
            continue
        expected = _ref_ai(M, N, K, bpe)
        rel_err = abs(got - expected) / max(abs(expected), 1e-12)
        if rel_err > 0.01:
            failures.append(
                f"  gemm_arithmetic_intensity({M},{N},{K},{bpe}) = {got:.4f}, "
                f"expected {expected:.4f} (within 1%)"
            )
    return failures


def _check_classify() -> list[str]:
    failures: list[str] = []
    for desc, M, N, K, bpe, gpu, expected_cls, _ in CASES:
        try:
            got = starter.classify_gemm(M, N, K, bpe, gpu)
        except NotImplementedError:
            failures.append(f"  {desc}: classify_gemm not implemented")
            continue
        except Exception as e:
            failures.append(f"  {desc}: classify_gemm raised {type(e).__name__}: {e}")
            continue
        if got != expected_cls:
            failures.append(
                f"  {desc}: classify_gemm = {got!r}, expected {expected_cls!r}"
            )
    return failures


def _check_predict() -> list[str]:
    failures: list[str] = []
    for desc, M, N, K, bpe, gpu, _, tol in CASES:
        try:
            got = starter.predict_tflops(M, N, K, bpe, gpu)
        except NotImplementedError:
            failures.append(f"  {desc}: predict_tflops not implemented")
            continue
        except Exception as e:
            failures.append(
                f"  {desc}: predict_tflops raised {type(e).__name__}: {e}"
            )
            continue
        expected = _ref_predict(M, N, K, bpe, gpu)
        if expected == 0:
            ok = abs(got) < 1e-6
        else:
            ok = abs(got - expected) / abs(expected) <= tol
        if not ok:
            failures.append(
                f"  {desc}: predict_tflops = {got:.2f}, expected ~{expected:.2f} "
                f"(within {tol*100:.0f}%)"
            )
    return failures


def _check_invalid_gpu() -> list[str]:
    failures: list[str] = []
    for fn_name in ("classify_gemm", "predict_tflops"):
        fn = getattr(starter, fn_name, None)
        if fn is None:
            failures.append(f"  starter is missing {fn_name}")
            continue
        try:
            fn(4096, 4096, 4096, 2, "B200")
        except ValueError:
            pass  # expected
        except NotImplementedError:
            failures.append(f"  {fn_name}: not implemented")
        except Exception as e:
            failures.append(
                f"  {fn_name}('B200', ...) should raise ValueError; "
                f"got {type(e).__name__}: {e}"
            )
        else:
            failures.append(
                f"  {fn_name}('B200', ...) should raise ValueError; returned silently"
            )
    return failures


def grade() -> int:
    failures: list[str] = []
    failures += _check_ai_directly()
    failures += _check_classify()
    failures += _check_predict()
    failures += _check_invalid_gpu()

    if failures:
        print("FAIL:")
        for f in failures:
            print(f)
        return 1

    print(
        f"PASS — AI, classification, and TFLOPS prediction match reference "
        f"on all {len(CASES)} GEMM cases."
    )
    return 0


if __name__ == "__main__":
    sys.exit(grade())
