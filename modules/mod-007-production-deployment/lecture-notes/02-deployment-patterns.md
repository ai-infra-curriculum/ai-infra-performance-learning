# Lecture 02: Deployment Patterns

## Rolling, blue-green, canary, shadow

(See engineer-solutions/mod-106 ex-08 for fully-implemented YAML manifests.)

Pick by risk profile:
- Low risk (re-trained version of same architecture): **rolling**
- Schema change or major version bump: **blue-green**
- Risky model change with measurable success metrics: **canary** (Argo Rollouts)
- New model architecture: **shadow** first, then canary

## Auto-revert gates

Argo Rollouts can auto-revert based on:
- 5xx rate during canary
- Latency p95
- Custom metric (e.g., model accuracy delta vs A/B control)

## Companion

- engineer-solutions/mod-106 ex-08 for all 4 strategies with manifests
- engineer-solutions/mod-110 ex-07 for multi-tenant LLM platform
