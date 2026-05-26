"""Exercise 6 starter — predict GEMM TFLOPS via roofline on tensor cores.

Implement three functions that use ONLY the verified A100 / H100
datasheet values below to predict GEMM behavior on the FP16
tensor-core path.

Run:

    python check.py
"""

from __future__ import annotations


# Verified NVIDIA datasheet values (see lecture-notes/06-gpu-generations.md):

A100_TC_FP16_TFLOPS = 312.0   # FP16 tensor-core peak, dense (no sparsity)
A100_BW_GBS = 2039.0          # HBM2e peak bandwidth

H100_TC_FP16_TFLOPS = 989.0   # FP16 tensor-core peak, dense
H100_BW_GBS = 3350.0          # HBM3 peak bandwidth


def gemm_arithmetic_intensity(M: int, N: int, K: int, bytes_per_elem: int) -> float:
    """Arithmetic intensity of an MxK times KxN GEMM at given precision.

    AI = total_flops / total_bytes_moved
       = 2*M*N*K / (bytes_per_elem * (M*K + K*N + M*N))

    Units: FLOP/byte.
    """
    raise NotImplementedError


def classify_gemm(M: int, N: int, K: int, bytes_per_elem: int, gpu: str) -> str:
    """Return 'memory-bound' or 'compute-bound' for a GEMM on `gpu`.

    `gpu` is 'A100' or 'H100'. Anything else: raise ValueError.

    The ridge point on the FP16 tensor-core path is:
        ridge_ai = (peak_tc_tflops * 1000) / peak_bw_gbs

    The GEMM is memory-bound if its AI is below this ridge.
    """
    raise NotImplementedError


def predict_tflops(
    M: int, N: int, K: int, bytes_per_elem: int, gpu: str, util: float = 0.85
) -> float:
    """Predict realistic TFLOPS for the GEMM on `gpu`'s tensor cores.

    The roofline-attainable peak:
        compute_ceiling = peak_tc_tflops * 1000      (GFLOPS)
        memory_ceiling  = peak_bw_gbs * AI            (GFLOPS)
        attainable      = min(both)

    Then apply a realistic utilization factor (default 0.85 to
    match well-tuned cuBLAS on compute-bound large GEMMs):

        realistic = util * attainable / 1000           (TFLOPS)
    """
    raise NotImplementedError
