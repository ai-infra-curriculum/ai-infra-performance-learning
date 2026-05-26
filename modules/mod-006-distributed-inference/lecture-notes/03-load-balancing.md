# Lecture 03: Load Balancing for LLM Inference

## Why naive LB hurts

Round-robin LB distributes requests evenly by count. For LLM serving, a
request that generates 10 tokens is much cheaper than one generating 2000
tokens. Even-by-count → uneven-by-load → some replicas saturate while others
idle.

## Better strategies

- **Least-loaded**: route to the replica with the lowest current queue depth (vLLM exposes this)
- **Power-of-two-choices**: sample 2 random replicas, pick the less loaded
- **Token-aware**: send long-prompt requests to replicas with more KV-cache headroom

vLLM has experimental support for last two; production deployments often
front vLLM with Envoy + a custom LB plugin.

## Routing for prefix caching

If you have prefix caching (mod-004), routing matters more — sending a
request to a replica that already has its system prompt cached is 2-3× faster.

Use consistent hashing on the prefix: same system prompt → same replica → cache hit.

## Failure modes

- Sticky session to a dead replica: TCP timeouts retry; LB should detect + remove
- Slow-loris: one client tying up a slot for hours; per-tenant concurrency limits
- Thundering herd on cold start: new replica gets 100% of traffic before warmup
  → use slow-start in the LB
