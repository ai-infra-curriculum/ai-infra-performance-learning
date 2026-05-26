# Ex 04: Prefix-Aware Routing

Build a thin router (FastAPI) in front of N vLLM replicas. Use consistent
hashing on the request's system prompt hash → route same system prompt to
same replica → prefix cache hit rate maximized.

Compare cache hit rate + throughput to round-robin LB.
