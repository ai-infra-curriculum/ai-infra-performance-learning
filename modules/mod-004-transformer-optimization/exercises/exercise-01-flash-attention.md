# Ex 01: FlashAttention Adoption

Take a vanilla transformer training loop. Replace its attention with
`F.scaled_dot_product_attention`. Benchmark seq_len=512, 2048, 8192.
Verify accuracy unchanged. Report speedup + memory savings.
