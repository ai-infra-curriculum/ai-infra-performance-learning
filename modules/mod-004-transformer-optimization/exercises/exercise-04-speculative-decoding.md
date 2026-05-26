# Ex 04: Speculative Decoding

Serve Llama-70B with a Llama-7B draft model via vLLM speculative decoding.
Benchmark vs the 70B alone. Expect 2-3× speedup on long generation; possibly
slower on short generation. Verify quality unchanged with a held-out set.
