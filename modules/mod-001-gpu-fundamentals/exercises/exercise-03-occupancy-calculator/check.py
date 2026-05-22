"""Autograder for Exercise 3. Run: python check.py

Tests the student's `occupancy` against a reference implementation on a
battery of cases including the four lecture examples and three edge
cases. The reference is included in this file so the autograder is
self-contained (the canonical solution lives in the paired solutions
repo).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import starter  # type: ignore[import-not-found]
    from starter import LaunchConfig, SMSpec
except ImportError as e:
    print(f"FAIL: could not import starter.py ({e})")
    sys.exit(1)


WARP_SIZE = 32


# -----------------------------------------------------------------------------
# Reference SM specs.
# -----------------------------------------------------------------------------

# An Ampere (CC 8.0) SM, e.g. A100. Values from CUDA C Programming Guide
# Appendix H "Compute Capabilities" table.
AMPERE_A100_SM = SMSpec(
    max_warps_per_sm=64,
    max_blocks_per_sm=32,
    register_file_size=65_536,
    shared_mem_per_sm_bytes=164 * 1024,  # configurable; 164 KB max on A100
)


# -----------------------------------------------------------------------------
# Reference implementation.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ref:
    active_blocks_per_sm: int
    active_warps_per_sm: int
    occupancy: float
    limiter: str


def reference_occupancy(launch: LaunchConfig, sm: SMSpec) -> _Ref:
    warps_per_block = math.ceil(launch.threads_per_block / WARP_SIZE)
    warp_cap = sm.max_warps_per_sm // warps_per_block if warps_per_block else math.inf
    block_cap = sm.max_blocks_per_sm

    if launch.registers_per_thread > 0:
        register_cap = sm.register_file_size // (
            launch.threads_per_block * launch.registers_per_thread
        )
    else:
        register_cap = math.inf

    if launch.shared_mem_per_block_bytes > 0:
        shared_cap = sm.shared_mem_per_sm_bytes // launch.shared_mem_per_block_bytes
    else:
        shared_cap = math.inf

    caps = {
        "warps": warp_cap,
        "blocks": block_cap,
        "registers": register_cap,
        "shared_mem": shared_cap,
    }
    binding_cap = min(caps.values())
    # Stable order so ties resolve the same way the student's code is
    # expected to (warps > blocks > registers > shared_mem).
    limiter = next(name for name, v in caps.items() if v == binding_cap)
    active_blocks = min(int(binding_cap), sm.max_blocks_per_sm)
    active_warps = active_blocks * warps_per_block
    occ = active_warps / sm.max_warps_per_sm

    return _Ref(active_blocks, active_warps, occ, limiter)


# -----------------------------------------------------------------------------
# Cases.
# -----------------------------------------------------------------------------


CASES = [
    # (description, launch, sm, expected_limiter)
    (
        "vector add (low register, no shared) — block_cap-bound (full occupancy)",
        LaunchConfig(threads_per_block=256, registers_per_thread=16, shared_mem_per_block_bytes=0),
        AMPERE_A100_SM,
        "warps",
    ),
    (
        "sgemm 16x16 tile (moderate registers + shared) — register-bound",
        LaunchConfig(threads_per_block=256, registers_per_thread=40, shared_mem_per_block_bytes=8 * 1024),
        AMPERE_A100_SM,
        "registers",
    ),
    (
        "register-heavy kernel (96 regs per thread) — register-bound",
        LaunchConfig(threads_per_block=256, registers_per_thread=96, shared_mem_per_block_bytes=0),
        AMPERE_A100_SM,
        "registers",
    ),
    (
        "shared-memory-heavy kernel (48 KB per block) — shared-bound",
        LaunchConfig(threads_per_block=128, registers_per_thread=32, shared_mem_per_block_bytes=48 * 1024),
        AMPERE_A100_SM,
        "shared_mem",
    ),
    (
        "tiny block (32 threads, light resources) — block_cap-bound",
        LaunchConfig(threads_per_block=32, registers_per_thread=8, shared_mem_per_block_bytes=0),
        AMPERE_A100_SM,
        "blocks",
    ),
    (
        "threads not a multiple of 32 (97 -> 4 warps wasted)",
        LaunchConfig(threads_per_block=97, registers_per_thread=32, shared_mem_per_block_bytes=0),
        AMPERE_A100_SM,
        "warps",  # warps_per_block=4, warp_cap=16, blocks=32 -> warp-bound
    ),
    (
        "zero shared memory, zero-ish registers — block_cap-bound",
        LaunchConfig(threads_per_block=128, registers_per_thread=8, shared_mem_per_block_bytes=0),
        AMPERE_A100_SM,
        "warps",
    ),
]


def grade() -> int:
    failures: list[str] = []

    for desc, launch, sm, expected_limiter in CASES:
        try:
            got = starter.occupancy(launch, sm)
        except NotImplementedError:
            failures.append(f"  {desc}: occupancy() not implemented")
            continue
        except Exception as e:  # pragma: no cover - surface any error
            failures.append(f"  {desc}: raised {type(e).__name__}: {e}")
            continue

        ref = reference_occupancy(launch, sm)

        if got.active_blocks_per_sm != ref.active_blocks_per_sm:
            failures.append(
                f"  {desc}: active_blocks_per_sm={got.active_blocks_per_sm}, "
                f"expected {ref.active_blocks_per_sm}"
            )
            continue
        if got.active_warps_per_sm != ref.active_warps_per_sm:
            failures.append(
                f"  {desc}: active_warps_per_sm={got.active_warps_per_sm}, "
                f"expected {ref.active_warps_per_sm}"
            )
            continue
        if abs(got.occupancy - ref.occupancy) > 1e-6:
            failures.append(
                f"  {desc}: occupancy={got.occupancy:.4f}, expected {ref.occupancy:.4f}"
            )
            continue
        if got.limiter != expected_limiter:
            failures.append(
                f"  {desc}: limiter={got.limiter!r}, expected {expected_limiter!r}"
            )
            continue

    if failures:
        print("FAIL:")
        for f in failures:
            print(f)
        return 1

    print(f"PASS — all {len(CASES)} occupancy cases match the reference.")
    return 0


if __name__ == "__main__":
    sys.exit(grade())
