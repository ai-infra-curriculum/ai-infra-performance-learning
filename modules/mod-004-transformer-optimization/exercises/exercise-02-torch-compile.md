# Ex 02: torch.compile

Apply `torch.compile(model, mode="reduce-overhead")` to a model from a HF
hub. Benchmark before/after on at least 3 sequence lengths.

Note: first call recompiles; warm up before measuring.
