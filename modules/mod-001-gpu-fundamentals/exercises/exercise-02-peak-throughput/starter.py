"""Exercise 2 starter — compute peak throughput from spec-sheet values.

Fill in the three GPU spec dicts and the two formula functions, then run:

    python check.py

The autograder checks that your computed peak FP32 throughput and peak
memory bandwidth land within 5% of NVIDIA's published values for each
GPU.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    """Inputs you read off a spec sheet for one GPU SKU."""

    name: str
    sm_count: int
    cuda_cores_per_sm: int
    boost_clock_ghz: float

    # Effective per-pin memory data rate in Gbps.
    # GDDR6X: NVIDIA quotes this directly (e.g. 21 Gbps for RTX 4090).
    # HBM2e/HBM3: equals total_bandwidth_gbs * 8 / bus_width_bits.
    memory_pin_gbps: float
    memory_bus_width_bits: int


# ---------------------------------------------------------------------------
# (a) Fill in the three spec dicts.
# Replace each ... with the value from NVIDIA's spec sheet
# (see README.md hints and resources.md for source links).
# ---------------------------------------------------------------------------

A100_SXM4_80GB = GPUSpec(
    name="A100 SXM4 80GB",
    sm_count=...,                      # how many SMs on an A100?
    cuda_cores_per_sm=...,             # FP32 CUDA cores per SM (GA100)
    boost_clock_ghz=...,               # boost clock in GHz
    memory_pin_gbps=...,               # HBM2e effective per-pin Gbps
    memory_bus_width_bits=...,         # HBM2e total bus width
)

H100_SXM5_80GB = GPUSpec(
    name="H100 SXM5 80GB",
    sm_count=...,
    cuda_cores_per_sm=...,
    boost_clock_ghz=...,
    memory_pin_gbps=...,               # HBM3 effective per-pin Gbps
    memory_bus_width_bits=...,
)

RTX_4090 = GPUSpec(
    name="RTX 4090",
    sm_count=...,
    cuda_cores_per_sm=...,
    boost_clock_ghz=...,
    memory_pin_gbps=...,               # GDDR6X per-pin Gbps
    memory_bus_width_bits=...,
)


# ---------------------------------------------------------------------------
# (b) Fill in the two formula functions.
# ---------------------------------------------------------------------------


def peak_fp32_tflops(spec: GPUSpec) -> float:
    """Peak FP32 throughput in TFLOPS, dense, no tensor cores.

    peak = (sm_count * cuda_cores_per_sm * boost_clock_ghz * 2) / 1000

    The factor of 2 is the FMA — each FP32 CUDA core can issue one
    multiply-add per cycle, which counts as 2 FLOPS.
    """
    raise NotImplementedError


def peak_memory_bandwidth_gbs(spec: GPUSpec) -> float:
    """Peak memory bandwidth in GB/s.

    bandwidth = (memory_pin_gbps * memory_bus_width_bits) / 8

    Per-pin gigabits per second times bus width in bits divided by
    8 bits per byte gives gigabytes per second, using powers of 10
    (which is how NVIDIA quotes memory bandwidth).
    """
    raise NotImplementedError
