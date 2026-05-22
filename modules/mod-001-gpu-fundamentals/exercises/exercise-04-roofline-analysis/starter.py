"""Exercise 4 starter — roofline analysis for an A100.

Implement the four functions below. The autograder verifies AI values,
classifications, attainable throughputs, and that the plot is written
with the right structure.

Run:

    python check.py
"""

from __future__ import annotations

from dataclasses import dataclass


# A100 SXM4 80GB. Values from NVIDIA datasheet; see resources.md.
A100_PEAK_FP32_TFLOPS = 19.5
A100_PEAK_BW_GBS = 2039.0


# The six kernels you analyze. The autograder checks AI for each of these
# names.
KERNEL_NAMES = (
    "vector_add",
    "vector_fma",
    "axpy_scaled",
    "relu",
    "gemm_8192",
    "softmax_row",
)


@dataclass(frozen=True)
class KernelStats:
    name: str
    flops: float       # total FLOPS for one invocation
    bytes_moved: float # total bytes moved between HBM and SM


def arithmetic_intensity(kernel: str) -> float:
    """Return arithmetic intensity (FLOP/byte) for `kernel`.

    Supported kernel names are in `KERNEL_NAMES`. Raise ValueError for
    anything else.
    """
    raise NotImplementedError


def classify(
    ai: float,
    peak_flops_tflops: float = A100_PEAK_FP32_TFLOPS,
    peak_bw_gbs: float = A100_PEAK_BW_GBS,
) -> str:
    """Return 'memory-bound' or 'compute-bound' for a given AI on a GPU.

    A kernel is memory-bound when its AI is below the ridge point, i.e.
    when peak bandwidth × AI is less than peak compute throughput.
    """
    raise NotImplementedError


def attainable_throughput_tflops(
    ai: float,
    peak_flops_tflops: float = A100_PEAK_FP32_TFLOPS,
    peak_bw_gbs: float = A100_PEAK_BW_GBS,
) -> float:
    """Return the roofline-bounded attainable throughput (TFLOPS) at this AI.

    attainable = min(peak_flops, peak_bw * ai)  -- in matching units.
    """
    raise NotImplementedError


def plot_roofline(out_path: str) -> None:
    """Write a roofline plot to `out_path` (a PNG file).

    The figure must contain:
      - the two ceilings (memory-bound line and compute-bound line)
      - one marker per kernel, placed at its AI on the attainable curve

    Use matplotlib's loglog scaling so the structure is legible.
    """
    raise NotImplementedError
