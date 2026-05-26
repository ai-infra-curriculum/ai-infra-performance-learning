# Ex 05: Write a Triton Kernel

Implement a fused LayerNorm in Triton. Benchmark vs `torch.nn.LayerNorm`.

Reference: [Triton LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html).

Target: within 20% of `torch.nn.LayerNorm` (which uses optimized cuBLAS internally) at hidden_dim=4096.
