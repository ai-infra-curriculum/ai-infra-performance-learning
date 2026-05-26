# Lecture 03: Speculative Decoding + Other Inference Tricks

## Speculative decoding

A small draft model proposes N tokens. The large target model verifies them
in one forward pass. If all N are correct, you got N tokens for the cost of
~1. If only K are correct, you got K tokens; throw out the rest.

Typical speedup: 2-3× on long completions for a well-matched draft model.

vLLM supports it out of the box:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-70b-chat-hf --tensor-parallel-size 4 \
  --speculative-model meta-llama/Llama-2-7b-chat-hf --num-speculative-tokens 5
```

## Continuous batching

Naive serving: pad all requests in a batch to the longest; small requests waste compute.

Continuous batching: each step adds new requests + drops finished ones, regardless of length.

Speedup: 5-10× for typical chat workloads (highly variable request lengths).

vLLM, TGI, TensorRT-LLM all do this.

## Prefix caching

Multiple requests share a system prompt. Cache its KV state; reuse across
requests. Speedup: 2-4× for system-prompt-heavy workloads (chat, agents, RAG).

## Compounding

```
Baseline → +FlashAttn (2x) → +continuous batching (4x) → +prefix cache (2x) → +spec decode (2x)
       1×              2×                       8×                   16×              32×
```

Real measured: ~30× from baseline HuggingFace `generate()` to optimized vLLM
on the same model.

## Companion

[engineer-solutions/mod-110 (all 14 LLM exercises)](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-110-llm-infrastructure).
