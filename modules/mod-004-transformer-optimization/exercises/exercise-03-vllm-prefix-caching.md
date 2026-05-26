# Ex 03: vLLM + Prefix Caching

Launch vLLM with `--enable-prefix-caching`. Build a workload where 1000
requests share a 500-token system prompt + a 50-token user query. Measure
TTFT + throughput vs no prefix caching.

Companion: engineer-solutions/mod-110 ex-03.
