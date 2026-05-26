# Lecture 01: Tensor + Pipeline Parallelism

## Tensor parallel (TP)

Split each layer's weights across N GPUs. Each GPU computes its slice; an
all-reduce combines results.

- **Pros**: low latency (no pipeline bubbles)
- **Cons**: requires fast interconnect (NVLink); communication scales with batch × seq

Use for: 7B-70B models on a single node (8 GPUs).

## Pipeline parallel (PP)

Split model into N sequential stages, each on a different GPU. Process a
mini-batch as several "micro-batches" so all stages stay busy.

- **Pros**: communication is small (only activations between stages)
- **Cons**: bubbles at start + end; latency adds up

Use for: 100B+ models across multiple nodes.

## Combined

Frontier models use 3D parallelism: TP within a node, PP across nodes, plus
data parallel for replicas.

vLLM's `--tensor-parallel-size 8 --pipeline-parallel-size 2` covers most
real deployments.

## When NOT to parallelize

A model that fits on one GPU (or one quantized GPU) is faster + cheaper to
serve replicated. Don't tensor-parallel a 7B model across 2 GPUs to brag.
