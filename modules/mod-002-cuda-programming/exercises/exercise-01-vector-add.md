# Ex 01: Vector Add Kernel

Write `add_kernel.cu` doing element-wise vector add. Pair with a PyTorch
extension; benchmark vs `torch.add`. Target: within 10% of `torch.add` at
n=1M; within 2× at n=10K.

Companion: engineer-solutions/mod-107 ex-02.
