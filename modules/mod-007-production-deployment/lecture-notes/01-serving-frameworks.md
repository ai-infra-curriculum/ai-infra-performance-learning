# Lecture 01: Serving Frameworks

## The landscape

| Framework | Best for |
|---|---|
| FastAPI + custom | small models, control over every request |
| TorchServe | PyTorch models, modest scale |
| Triton Inference Server | multi-model, multi-framework, NVIDIA-shop |
| TensorFlow Serving | TF models |
| vLLM | LLMs (autoregressive generation) |
| TGI (HuggingFace) | LLMs, simpler ops than vLLM |
| BentoML | model-ops mature; pythonic |
| Ray Serve | streaming + composition heavy |

## Decision tree

- LLM inference (autoregressive) → vLLM or TGI
- Classification / vision / embeddings → Triton or TorchServe
- Custom logic per request → FastAPI
- Need composition (chain of model calls) → Ray Serve / BentoML
- Already in TF ecosystem → TF Serving

## Anti-patterns

- One serving framework per model → fragmented ops
- vLLM for classification (the wrong tool)
- FastAPI alone for high-QPS LLM (no continuous batching)
- TorchServe for the highest-traffic LLM (vLLM is much faster)
