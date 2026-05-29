# AI/ML Performance Engineer - Learning Repository

> **Specialized Track**: Performance Engineering & Optimization for AI/ML Systems
>
> Master GPU optimization, CUDA programming, model compression, and high-performance inference systems for production AI/ML workloads.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Course Level](https://img.shields.io/badge/Level-Advanced-red.svg)]()
[![Estimated Time](https://img.shields.io/badge/Time-200--250_hours-blue.svg)]()

## Table of Contents

- [Overview](#overview)
- [Learning Path](#learning-path)
- [Prerequisites](#prerequisites)
- [Course Structure](#course-structure)
- [Projects](#projects)
- [Setup Instructions](#setup-instructions)
- [Learning Resources](#learning-resources)
- [Assessment & Certification](#assessment--certification)
- [Community & Support](#community--support)
- [Career Path](#career-path)

## Overview

This learning repository is designed to transform senior AI infrastructure engineers into specialized **AI/ML Performance Engineers**—experts who optimize deep learning models and infrastructure for production deployment at scale.

### What You'll Learn

- **GPU Architecture & CUDA Programming**: Master low-level GPU optimization and custom kernel development
- **Performance Profiling**: Use NVIDIA Nsight, PyTorch Profiler, and other tools to identify bottlenecks
- **Model Compression**: Implement quantization, pruning, knowledge distillation, and TensorRT conversion
- **Transformer Optimization**: Build custom CUDA kernels for Flash Attention, RoPE, and LayerNorm
- **High-Performance Inference**: Design LLM serving systems with continuous batching and PagedAttention
- **Distributed Optimization**: Optimize multi-GPU training and inference pipelines
- **Production Deployment**: Deploy optimized models with monitoring and cost optimization

### Who This Is For

This course is designed for:

- **Senior AI Infrastructure Engineers** looking to specialize in performance optimization
- **ML Platform Engineers** who need deep GPU and optimization expertise
- **Performance Engineers** transitioning to AI/ML workloads
- **Research Engineers** deploying models to production at scale

### Prerequisites

**Required Skills**:
- Strong Python programming (3+ years)
- PyTorch or TensorFlow experience (2+ years)
- Linux/Unix system administration
- Git version control
- Docker and Kubernetes basics
- Understanding of transformer architectures (GPT, BERT, LLaMA)

**Recommended Background**:
- Computer architecture fundamentals
- C++ programming
- Distributed systems concepts
- Production ML experience

**Hardware Requirements**:
- NVIDIA GPU (minimum: RTX 3090, A10G, or cloud GPU instance)
  - Recommended: A100, A10G, or H100
- 64GB+ RAM
- 500GB+ SSD storage
- Ubuntu 20.04/22.04 or similar Linux distribution

**Software Prerequisites**:
- CUDA Toolkit 12.0+
- Python 3.10+
- PyTorch 2.1+
- Docker
- Git

## Learning Path

```
Prerequisites ──> GPU Fundamentals ──> CUDA Programming ──> Performance Profiling
                                                                      │
                                                                      ▼
Production Deployment <── Distributed Inference <── Model Compression
       │                          │                         │
       ▼                          ▼                         ▼
   Project 3              Advanced Topics        Transformer Optimization
(LLM Inference)                                          │
                                                         ▼
                                                    Project 1 & 2
                                              (Compression & CUDA Kernels)
```

### Estimated Timeline

- **Total Duration**: 200-250 hours (10-12 weeks full-time, 20-25 weeks part-time)
- **Lessons**: 8 modules, 2-3 weeks each
- **Projects**: 3 major projects, 40-80 hours each
- **Assessments**: Weekly quizzes + 3 practical exams

## Course Structure

### Module 1: GPU Fundamentals (20 hours)

**Learning Objectives**:
- Understand GPU architecture (CUDA cores, Tensor Cores, memory hierarchy)
- Master GPU memory management (global, shared, registers, L1/L2 cache)
- Learn CUDA execution model (grids, blocks, threads, warps)
- Understand memory bandwidth and compute-bound operations

**Topics**:
- NVIDIA GPU architecture evolution (Pascal → Ampere → Hopper)
- CUDA programming model fundamentals
- Memory hierarchy and bandwidth optimization
- Warp-level operations and thread divergence
- Occupancy and resource utilization

**Deliverables**:
- Quiz: GPU architecture and CUDA model
- Exercise: Memory bandwidth analysis
- Lab: Simple CUDA kernel profiling

### Module 2: CUDA Programming (30 hours)

**Learning Objectives**:
- Write efficient CUDA kernels from scratch
- Optimize memory access patterns (coalescing, alignment)
- Use shared memory and warp primitives
- Integrate CUDA with PyTorch (C++ extensions)

**Topics**:
- CUDA kernel syntax and launch configurations
- Memory coalescing and alignment
- Shared memory optimization and bank conflicts
- Warp-level primitives (`__shfl`, reductions)
- PyTorch C++ extensions with pybind11
- Autograd integration for custom operators

**Deliverables**:
- Quiz: CUDA programming concepts
- Exercise: Implement vectorized operations
- Lab: Build PyTorch CUDA extension

### Module 3: Performance Profiling (25 hours)

**Learning Objectives**:
- Profile GPU applications with NVIDIA Nsight Compute
- Analyze system-wide performance with Nsight Systems
- Perform roofline analysis
- Identify memory vs compute bottlenecks

**Topics**:
- NVIDIA Nsight Compute deep dive
- NVIDIA Nsight Systems for end-to-end profiling
- PyTorch Profiler and TensorBoard integration
- Roofline model and performance analysis
- Memory bandwidth vs compute utilization
- Kernel optimization strategies

**Deliverables**:
- Quiz: Profiling tools and metrics
- Exercise: Roofline analysis case study
- Lab: Profile and optimize transformer model

### Module 4: Transformer Optimization (40 hours)

**Learning Objectives**:
- Understand transformer architecture bottlenecks
- Implement Flash Attention algorithm
- Build custom CUDA kernels for RoPE, LayerNorm, GELU
- Optimize attention memory usage

**Topics**:
- Transformer architecture deep dive
- Attention mechanism bottlenecks
- Flash Attention algorithm and implementation
- Rotary Position Embeddings (RoPE) optimization
- Fused kernel design (LayerNorm + GELU)
- Memory-efficient attention patterns

**Deliverables**:
- Quiz: Transformer optimization techniques
- Exercise: Flash Attention analysis
- **Project 2**: Custom CUDA Kernels for Transformers (60 hours)

### Module 5: Model Compression (35 hours)

**Learning Objectives**:
- Implement post-training quantization (PTQ) and QAT
- Apply structured pruning techniques
- Implement knowledge distillation
- Convert models to TensorRT

**Topics**:
- Quantization: INT8, FP16, mixed precision
- PyTorch quantization APIs
- Pruning: magnitude-based, structured, iterative
- Knowledge distillation frameworks
- TensorRT conversion and optimization
- Calibration strategies for quantization

**Deliverables**:
- Quiz: Compression techniques
- Exercise: Quantization sensitivity analysis
- **Project 1**: Automated Model Compression Pipeline (40 hours)

### Module 6: Distributed Inference (30 hours)

**Learning Objectives**:
- Implement tensor parallelism for large models
- Design efficient multi-GPU serving systems
- Optimize cross-GPU communication
- Build load balancing systems

**Topics**:
- Tensor parallelism fundamentals
- Pipeline parallelism for inference
- NCCL and inter-GPU communication
- Load balancing strategies
- Multi-GPU memory management
- Scaling efficiency analysis

**Deliverables**:
- Quiz: Distributed inference
- Exercise: Tensor parallelism implementation
- Lab: Multi-GPU serving system

### Module 7: Production Deployment (25 hours)

**Learning Objectives**:
- Design high-throughput inference APIs
- Implement continuous batching
- Build monitoring and observability systems
- Optimize cost per inference

**Topics**:
- REST and gRPC APIs for inference
- Continuous batching and request scheduling
- Prometheus and Grafana monitoring
- SLA management and autoscaling
- Cost optimization strategies
- Deployment with Docker and Kubernetes

**Deliverables**:
- Quiz: Production deployment
- Exercise: Design serving architecture
- **Project 3**: High-Performance LLM Inference System (80 hours)

### Module 8: Advanced Topics (20 hours)

**Learning Objectives**:
- Implement speculative decoding
- Use PagedAttention for memory efficiency
- Explore INT4 quantization
- Learn latest optimization techniques

**Topics**:
- PagedAttention implementation
- Speculative decoding algorithms
- INT4 and sub-byte quantization
- Continuous batching advanced patterns
- Flash Decoding for inference
- Latest research in LLM optimization

**Deliverables**:
- Quiz: Advanced optimization
- Exercise: PagedAttention analysis
- Lab: Implement speculative decoding

## Projects

### Project 1: Automated Model Compression Pipeline (40 hours)

**Complexity**: Intermediate+

Build a production-ready compression pipeline that applies quantization, pruning, knowledge distillation, and TensorRT conversion to reduce model size by 75% and increase inference speed by 3x while maintaining 98%+ accuracy.

**Technologies**: PyTorch, TensorRT, ONNX, Neural Compressor

**Performance Targets**:
- 3x inference speedup
- 75% model size reduction
- <2% accuracy degradation

**Key Features**:
- Post-training quantization (INT8/FP16)
- Quantization-aware training
- Structured pruning with fine-tuning
- TensorRT engine building
- Automated benchmarking

📁 Project 1 — _planned (project repo not yet scaffolded; see modules 03-05 for the underlying techniques)_

### Project 2: Custom CUDA Kernels for Transformer Optimization (60 hours)

**Complexity**: Advanced

Develop custom CUDA kernels to optimize critical transformer operations, achieving 3x+ speedup over standard PyTorch implementations through Flash Attention, fused RoPE, optimized LayerNorm, and GELU.

**Technologies**: CUDA, C++, PyTorch C++ Extensions, Triton, Nsight

**Performance Targets**:
- Flash Attention: 3x speedup
- Fused kernels: 3.5x speedup
- 80%+ memory bandwidth utilization
- 70%+ compute utilization

**Key Features**:
- Flash Attention v2 implementation
- Fused RoPE kernel
- Welford-based LayerNorm
- Vectorized GELU
- PyTorch integration

📁 Project 2 — _planned (project repo not yet scaffolded; see module 08 for advanced GPU topics)_

### Project 3: High-Performance LLM Inference System (80 hours)

**Complexity**: Advanced+

Build a production-grade LLM serving system capable of 1000+ requests/second with P99 latency <100ms using continuous batching, PagedAttention, and advanced scheduling.

**Technologies**: PyTorch, vLLM, FastAPI, FlashAttention, Triton

**Performance Targets**:
- 1000+ req/sec throughput
- <100ms P99 latency
- 85%+ GPU utilization
- 70% memory savings with PagedAttention

**Key Features**:
- Continuous batching engine
- PagedAttention implementation
- Dynamic request scheduling
- Streaming inference support
- Prometheus monitoring

📁 Project 3 — _planned (project repo not yet scaffolded; see module 06 for distributed inference techniques)_

## Setup Instructions

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/ai-infra-curriculum/ai-infra-performance-learning.git
cd ai-infra-performance-learning

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify CUDA installation
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
```

### 2. CUDA Toolkit Installation

```bash
# Ubuntu 22.04 (adjust for your distribution)
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

### 3. NVIDIA Nsight Tools

```bash
# Nsight Compute
sudo apt-get install nvidia-nsight-compute

# Nsight Systems
sudo apt-get install nvidia-nsight-systems

# Verify
ncu --version
nsys --version
```

### 4. Development Tools

```bash
# CMake (for CUDA compilation)
sudo apt-get install cmake

# Build essentials
sudo apt-get install build-essential

# pybind11 for PyTorch extensions
pip install pybind11
```

### 5. Project-Specific Setup

Per-project setup will live alongside each project once scaffolded.
For now, follow the module-level setup instructions in [modules/](modules/) — each module is independently runnable.

## Learning Resources

### Essential Reading

**GPU & CUDA**:
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) (NVIDIA)
- "Programming Massively Parallel Processors" by Hwu, Kirk, Hajj
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) (NVIDIA)

**Model Optimization**:
- "A White Paper on Neural Network Quantization" (Qualcomm)
- "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (Google)
- "Learning both Weights and Connections for Efficient Neural Networks" (Han et al.)

**Transformer Optimization**:
- "Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022)
- "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (Dao, 2023)
- "Attention is All You Need" (Vaswani et al., 2017)

**LLM Serving**:
- "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM paper)
- "Orca: A Distributed Serving System for Transformer-Based Generative Models"

### Tools & Libraries

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [TensorRT](https://developer.nvidia.com/tensorrt) - NVIDIA inference optimization
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform inference
- [vLLM](https://github.com/vllm-project/vllm) - LLM serving reference
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Optimized attention
- [Triton](https://github.com/openai/triton) - GPU programming language
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - Optimization library

### Video Courses

- NVIDIA Deep Learning Institute - GPU Programming
- NVIDIA DLI - Optimizing Deep Learning Models
- Coursera - GPU Programming Specialization
- YouTube: CUDA Programming tutorials by NVIDIA

### Community Resources

- NVIDIA Developer Forums
- PyTorch Discussion Forums
- r/MachineLearning Performance threads
- MLPerf benchmarking community

## Assessment & Certification

### Quiz System

Each module includes:
- **Pre-quiz**: Assess baseline knowledge
- **Mid-module checkpoints**: Verify understanding
- **Post-quiz**: Comprehensive module assessment

**Passing Score**: 80% or higher

### Practical Examinations

Three major practical exams aligned with projects:

1. **Compression Exam** (Module 5): Compress a given model to meet performance targets
2. **CUDA Exam** (Module 4): Implement custom CUDA kernel from specification
3. **Inference Exam** (Module 7): Deploy serving system meeting SLA requirements

**Passing Criteria**: Meet all performance targets

### Final Certification

**Requirements**:
- Complete all 8 modules with 80%+ quiz scores
- Submit all 3 projects with passing grades
- Pass all 3 practical examinations

**Certificate**: "AI/ML Performance Engineer - Advanced Specialization"

## Community & Support

### Getting Help

- **Discussion Forum**: [GitHub Discussions](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/discussions) — the supported community surface today.
- **Issues**: File [an issue](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/issues) for content corrections or runtime-validation problems.
- **Dedicated chat / office hours**: Not currently scheduled — coordinate via Discussions if you'd like to organize one with other learners.

### Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas for contribution:
- Additional exercises and labs
- Bug fixes and improvements
- Performance benchmarks
- Documentation enhancements
- New optimization techniques

### Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please read and adhere to it in all interactions.

## Career Path

### Role Progression

```
AI Infrastructure Engineer (Level 2)
            ↓
AI/ML Performance Engineer (Level 2.5D) ← YOU ARE HERE
            ↓
    ┌──────────────┬──────────────┐
    ↓              ↓              ↓
Senior Performance  Principal     Performance
Engineer (L3)      Architect     Team Lead
```

### Skills Matrix

| Skill | Entry | After Course | Expert |
|-------|-------|-------------|--------|
| GPU Programming | Basic | Advanced | ⭐⭐⭐⭐⭐ |
| CUDA Kernels | None | Intermediate | ⭐⭐⭐⭐ |
| Model Compression | Basic | Advanced | ⭐⭐⭐⭐⭐ |
| Performance Profiling | Basic | Advanced | ⭐⭐⭐⭐⭐ |
| LLM Serving | None | Advanced | ⭐⭐⭐⭐ |
| Production Deployment | Intermediate | Advanced | ⭐⭐⭐⭐⭐ |

### Salary Expectations

Based on industry data (2024 US market):

- **Entry Performance Engineer**: $140K - $180K
- **Senior Performance Engineer**: $180K - $240K
- **Principal Performance Engineer**: $240K - $350K+

Specialized AI/ML Performance Engineers command 20-30% premium over general infrastructure roles.

### Next Steps After Completion

1. **Senior AI Infrastructure Architect** track
2. **Principal AI Infrastructure Engineer** (technical leadership)
3. **AI Infrastructure Team Lead** (people management)
4. Specialized roles: MLOps, ML Platform, AI Security

## Project Timeline

### Recommended Schedule (Full-Time)

| Week | Module | Activities | Hours |
|------|--------|-----------|-------|
| 1-2 | Module 1-2 | GPU Fundamentals + CUDA | 50 |
| 3 | Module 3 | Performance Profiling | 25 |
| 4-5 | Project 1 | Model Compression Pipeline | 40 |
| 6-7 | Module 4 | Transformer Optimization | 40 |
| 8-10 | Project 2 | Custom CUDA Kernels | 60 |
| 11-12 | Module 5-6 | Compression + Distributed | 65 |
| 13-16 | Project 3 | LLM Inference System | 80 |
| 17-18 | Module 7-8 | Production + Advanced | 45 |

**Total**: ~18 weeks (full-time) or 36 weeks (part-time, 20 hrs/week)

## License

This learning repository is licensed under the [MIT License](LICENSE).

Course materials, code examples, and projects are freely available for educational purposes.

## Contact

- **GitHub**: [@ai-infra-curriculum](https://github.com/ai-infra-curriculum)
- **Email**: ai-infra-curriculum@joshua-ferguson.com
- **Website**: [ai-infra-curriculum.com](https://ai-infra-curriculum.com)

## Acknowledgments

This curriculum was developed with input from:
- Senior ML Infrastructure Engineers at major tech companies
- NVIDIA Developer Relations team
- Academic researchers in GPU optimization
- Production ML teams deploying LLMs at scale

---

**Ready to become an AI/ML Performance Engineering expert?**

Start with [Module 1: GPU Fundamentals →](modules/mod-001-gpu-fundamentals/README.md)
*(Module 1 ships with 6 learning objectives, lecture notes, four
autograded CPU-only exercises, and a 12-question quiz. Modules 2+
are scheduled.)*

> The longer-form curriculum spec (8 modules + 3 projects) lives in
> [`CURRICULUM.md`](CURRICULUM.md). Modules will be promoted from
> spec to implementation incrementally.


---

<!-- aicg:maintained-by -->
Maintained by [VeriSwarm.ai](https://veriswarm.ai)
