"""Exercise 1 starter — fill in your answers for the three workloads.

Each entry of `ANSWERS` is a dict with three keys:

    gpu_helps: bool
        True if you expect a modern data-center GPU (A100/H100 class) to
        outperform a comparable modern server CPU on this workload.

    expected_speedup_bucket: str
        One of: "<1x", "1-2x", "2-10x", "10-100x", ">100x".
        ">100x" means the GPU is more than 100 times faster.
        "<1x" means the CPU is faster.

    reasoning: str
        3-6 sentences of justification. The autograder does not grade
        the prose; the solutions repo has the worked answer.

Replace each `...` with your answer. Then run:

    python check.py
"""

ANSWERS = {
    "dense_matmul": {
        # Multiply two FP32 (8192, 8192) matrices, all data already on device.
        "gpu_helps": ...,
        "expected_speedup_bucket": ...,
        "reasoning": ...,
    },
    "json_parse": {
        # Parse a 1 GB deeply-nested JSON file.
        "gpu_helps": ...,
        "expected_speedup_bucket": ...,
        "reasoning": ...,
    },
    "elementwise_relu": {
        # y = max(0, x) on a single FP32 (1024, 1024) tensor,
        # data already on the appropriate device.
        "gpu_helps": ...,
        "expected_speedup_bucket": ...,
        "reasoning": ...,
    },
}
