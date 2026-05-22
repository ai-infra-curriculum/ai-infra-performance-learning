"""Exercise 3 starter — reimplement the NVIDIA occupancy calculation.

Implement the `occupancy` function. The autograder runs your
implementation against a battery of cases (vector add, GEMM tile,
register-heavy kernel, shared-memory-heavy kernel, edge cases).

Run:

    python check.py
"""

from __future__ import annotations

from dataclasses import dataclass


WARP_SIZE = 32


@dataclass(frozen=True)
class LaunchConfig:
    """A kernel launch's per-thread / per-block resource usage."""

    threads_per_block: int
    registers_per_thread: int
    shared_mem_per_block_bytes: int


@dataclass(frozen=True)
class SMSpec:
    """One SM's resource budget."""

    max_warps_per_sm: int
    max_blocks_per_sm: int
    register_file_size: int          # 32-bit registers per SM
    shared_mem_per_sm_bytes: int


@dataclass(frozen=True)
class OccupancyResult:
    """What the calculation tells the caller."""

    active_blocks_per_sm: int
    active_warps_per_sm: int
    occupancy: float                 # in [0.0, 1.0]
    limiter: str                     # "warps" | "blocks" | "registers" | "shared_mem"


def occupancy(launch: LaunchConfig, sm: SMSpec) -> OccupancyResult:
    """Return how many blocks/warps can run concurrently on one SM."""
    raise NotImplementedError
