# Module 02: CUDA Programming

**Duration**: ~30 hours
**Prerequisites**: Module 01 (GPU fundamentals); C++ basics; Linux + nvcc setup

## Overview

Mod-001 explained GPU architecture conceptually. Module 02 takes you from
"understands warps + occupancy" to "writes a custom CUDA kernel + integrates
it into PyTorch." We cover the C++ side (kernels, memory, launch config)
+ the PyTorch C++ extension build chain.

## Objectives

1. Write a CUDA kernel that beats `cudaMemcpy` for memory-bound ops.
2. Use shared memory for tile-based matmul; reach 70%+ of cuBLAS perf.
3. Pack a custom kernel as a PyTorch C++ extension; benchmark vs `torch.add`.
4. Read PTX + SASS; identify register spills + bank conflicts.
5. Tune launch config (block size, grid size) for compute-bound + memory-bound kernels.

```
mod-002-cuda-programming/
├── README.md
├── lecture-notes/  (3 files)
├── exercises/      (5)
├── quiz.md, quiz-answers.md
└── resources.md
```

Companion: [engineer-solutions/mod-107 ex-02 (cuda-kernel)](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-107-gpu-computing/exercise-02-cuda-kernel).
