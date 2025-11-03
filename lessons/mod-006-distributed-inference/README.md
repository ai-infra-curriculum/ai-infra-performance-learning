# Module 6: Distributed Inference

## Overview

Master techniques for serving large models across multiple GPUs with tensor parallelism, pipeline parallelism, and efficient batching.

**Duration**: 4-5 weeks (20-25 hours)

## Learning Objectives

- Implement tensor parallelism for model sharding
- Deploy pipeline parallelism for memory efficiency
- Optimize multi-GPU communication
- Build high-throughput serving systems

## Key Topics

### 1. Tensor Parallelism
- Model sharding strategies
- Attention and FFN parallelization
- Communication optimization
- Megatron-LM techniques

### 2. Pipeline Parallelism
- Layer distribution across GPUs
- Micro-batching and bubble optimization
- GPipe and PipeDream algorithms

### 3. Inference Optimization
- Dynamic batching
- Request scheduling
- Load balancing
- Fault tolerance

### 4. Production Systems
- Ray Serve integration
- vLLM deployment
- Multi-node scaling
- Cost optimization

## Expected Outcomes

- **Throughput**: 5-10x with multi-GPU
- **Latency**: <200ms p95 for large models
- **Scalability**: Linear scaling to 8+ GPUs

See `lecture-notes/` for detailed content and `exercises/` for hands-on labs.
