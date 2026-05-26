# Ex 04: Multi-Tier Cost Routing

Build a router that selects small / medium / large model per request:
- Small (Mistral-7B local) for simple QA
- Medium (gpt-4o-mini) for code
- Large (gpt-4o) for complex reasoning

Use a heuristic classifier + confidence-based escalation. Measure cost +
quality vs always-large baseline.

Companion: engineer-solutions/mod-110 ex-08.
