# AI/ML Performance Engineer - Comprehensive Curriculum

## Course Overview

This curriculum transforms experienced AI infrastructure engineers into specialized **AI/ML Performance Engineers** through hands-on projects, advanced GPU programming, and production optimization techniques.

### Learning Philosophy

1. **Performance-First**: Every lesson focuses on measurable performance improvements
2. **Hands-On**: Learn by building real optimization systems
3. **Production-Ready**: All projects use production-grade tools and practices
4. **Measurement-Driven**: Profile, benchmark, and validate all optimizations

### Prerequisites Assessment

Before starting, you should be able to:

- [ ] Write production Python code with type hints and testing
- [ ] Train transformer models with PyTorch
- [ ] Deploy Docker containers to cloud infrastructure
- [ ] Read and understand research papers
- [ ] Debug performance issues with basic profiling tools
- [ ] Use Linux command line proficiently

**Self-Assessment**: If you can confidently check the boxes above, you're ready. Otherwise, work through [ai-infra-engineer-learning](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning) first.

## Detailed Module Breakdown

---

## Module 1: GPU Fundamentals (20 hours)

### Learning Outcomes

By the end of this module, you will:

- Explain GPU architecture from CUDA cores to memory hierarchy
- Calculate theoretical peak performance (FLOPS and bandwidth)
- Understand warp execution and thread divergence
- Analyze occupancy and resource utilization
- Apply memory coalescing principles

### Week 1: GPU Architecture Deep Dive

**Lesson 1.1: Introduction to GPU Computing** (3 hours)
- CPU vs GPU architecture comparison
- SIMT (Single Instruction, Multiple Thread) execution model
- Use cases for GPU acceleration in ML
- Understanding when GPUs help (and when they don't)

**Lesson 1.2: NVIDIA GPU Architecture** (4 hours)
- Streaming Multiprocessors (SM) structure
- CUDA cores, Tensor Cores, RT cores
- Architecture evolution: Pascal → Volta → Ampere → Hopper
- Compute capability and feature sets
- **Lab**: Profile model on different GPU architectures

**Lesson 1.3: GPU Memory Hierarchy** (4 hours)
- Global memory (HBM/GDDR)
- L2 cache and L1 cache/shared memory
- Registers and local memory
- Constant and texture memory
- Memory bandwidth calculations
- **Lab**: Measure memory bandwidth with benchmarks

### Week 2: CUDA Execution Model

**Lesson 1.4: Threads, Blocks, and Grids** (3 hours)
- Thread hierarchy (thread → block → grid)
- Thread indices and ID calculations
- Block dimensions and constraints
- Occupancy fundamentals
- **Exercise**: Calculate optimal block dimensions

**Lesson 1.5: Warp Execution** (3 hours)
- Warp definition and warp size (32 threads)
- Thread divergence and branching
- Warp scheduling and latency hiding
- Active vs eligible warps
- **Lab**: Profile warp divergence impact

**Lesson 1.6: Performance Metrics** (3 hours)
- Compute bound vs memory bound
- Arithmetic intensity and roofline model
- Achieved vs theoretical performance
- GPU utilization metrics
- **Exercise**: Roofline analysis case study

### Hands-On Labs

1. **Lab 1.1**: GPU Memory Bandwidth Benchmarking
   - Measure achievable bandwidth with different access patterns
   - Compare global, shared, and register access speeds

2. **Lab 1.2**: Occupancy Calculator Usage
   - Use NVIDIA Occupancy Calculator
   - Optimize kernel launch configurations

3. **Lab 1.3**: Simple CUDA Kernel Profiling
   - Write basic vector addition kernel
   - Profile with Nsight Compute
   - Analyze metrics and bottlenecks

### Assessment

- **Quiz 1**: GPU architecture and CUDA model (20 questions, 80% to pass)
- **Practical**: Optimize simple kernel for maximum bandwidth utilization
- **Deliverable**: Performance analysis report

---

## Module 2: CUDA Programming (30 hours)

### Learning Outcomes

By the end of this module, you will:

- Write efficient CUDA kernels from scratch
- Optimize memory access patterns for coalescing
- Use shared memory effectively
- Implement warp-level reductions
- Integrate CUDA kernels with PyTorch

### Week 1: CUDA Basics

**Lesson 2.1: CUDA Kernel Syntax** (4 hours)
- `__global__`, `__device__`, `__host__` qualifiers
- Kernel launch syntax `<<<grid, block>>>`
- Thread index calculations
- Error handling in CUDA
- **Lab**: Write first CUDA kernel

**Lesson 2.2: Memory Management** (4 hours)
- Device memory allocation (`cudaMalloc`, `cudaFree`)
- Host-device transfers (`cudaMemcpy`)
- Unified memory and managed memory
- Memory pinning for faster transfers
- **Exercise**: Optimize transfer patterns

**Lesson 2.3: Memory Coalescing** (5 hours)
- Coalesced vs strided access patterns
- Structure of Arrays (SoA) vs Array of Structures (AoS)
- Alignment requirements
- Bank conflicts in shared memory
- **Lab**: Measure impact of access patterns

### Week 2: Advanced CUDA

**Lesson 2.4: Shared Memory** (5 hours)
- Shared memory allocation and usage
- Bank conflicts and how to avoid them
- Shared memory as cache
- Synchronization with `__syncthreads()`
- **Lab**: Implement matrix multiplication with shared memory

**Lesson 2.5: Warp-Level Primitives** (4 hours)
- Shuffle instructions (`__shfl_*`)
- Warp-level reductions
- Warp synchronization
- Ballot and vote functions
- **Exercise**: Implement fast reduction kernel

**Lesson 2.6: PyTorch C++ Extensions** (4 hours)
- pybind11 basics
- Writing PyTorch CUDA extensions
- Autograd integration
- Building with setuptools
- **Lab**: Create custom CUDA operator for PyTorch

### Week 3: Optimization Techniques

**Lesson 2.7: Kernel Optimization Strategies** (4 hours)
- Vectorized memory access (`float4`, `int2`)
- Loop unrolling
- Register optimization
- Instruction-level parallelism
- **Lab**: Optimize element-wise operations

### Hands-On Labs

1. **Lab 2.1**: Vector Operations with Coalescing
   - Implement vectorized add, multiply, fused operations
   - Benchmark against PyTorch

2. **Lab 2.2**: Matrix Multiplication Optimization
   - Naive implementation
   - Shared memory tiling
   - Register tiling
   - Compare with cuBLAS

3. **Lab 2.3**: PyTorch Custom Operator
   - Write CUDA kernel for custom activation
   - Integrate with PyTorch autograd
   - Benchmark speedup

### Assessment

- **Quiz 2**: CUDA programming concepts (25 questions, 80% to pass)
- **Coding Challenge**: Implement optimized reduction kernel
- **Project**: Build custom PyTorch CUDA extension

---

## Module 3: Performance Profiling (25 hours)

### Learning Outcomes

By the end of this module, you will:

- Profile GPU applications with NVIDIA Nsight Compute
- Analyze system-wide performance with Nsight Systems
- Perform roofline analysis
- Identify and fix performance bottlenecks
- Optimize based on profiling data

### Week 1: Profiling Tools

**Lesson 3.1: NVIDIA Nsight Compute** (5 hours)
- Nsight Compute overview and metrics
- Kernel replay and analysis
- Memory throughput analysis
- Compute utilization metrics
- **Lab**: Profile custom CUDA kernel

**Lesson 3.2: NVIDIA Nsight Systems** (4 hours)
- System-wide timeline profiling
- CPU-GPU interactions
- Data transfer bottlenecks
- Kernel launch overhead
- **Lab**: Profile PyTorch training loop

**Lesson 3.3: PyTorch Profiler** (3 hours)
- torch.profiler API
- TensorBoard integration
- Memory profiling
- Operator-level analysis
- **Lab**: Profile transformer model

### Week 2: Performance Analysis

**Lesson 3.4: Roofline Model** (5 hours)
- Roofline model theory
- Calculating arithmetic intensity
- Memory-bound vs compute-bound analysis
- Identifying optimization opportunities
- **Exercise**: Roofline analysis case studies

**Lesson 3.5: Bottleneck Identification** (4 hours)
- Memory bandwidth bottlenecks
- Compute utilization issues
- Occupancy problems
- Kernel launch overhead
- **Lab**: Systematic bottleneck analysis

**Lesson 3.6: Optimization Iteration** (4 hours)
- Iterative optimization workflow
- Measuring optimization impact
- Regression testing
- Performance tracking
- **Exercise**: Optimize real model based on profiling

### Hands-On Labs

1. **Lab 3.1**: Nsight Compute Deep Dive
   - Profile attention mechanism
   - Analyze memory bottlenecks
   - Identify optimization opportunities

2. **Lab 3.2**: End-to-End Model Profiling
   - Profile GPT-2 inference
   - Identify top bottlenecks
   - Create optimization roadmap

3. **Lab 3.3**: Roofline Analysis Practice
   - Calculate theoretical peaks
   - Measure achieved performance
   - Plot roofline charts

### Assessment

- **Quiz 3**: Profiling tools and metrics (20 questions, 80% to pass)
- **Lab Report**: Complete profiling analysis of provided model
- **Presentation**: Optimization recommendations based on profiling

---

## Module 4: Transformer Optimization (40 hours)

### Learning Outcomes

By the end of this module, you will:

- Understand transformer architecture bottlenecks
- Implement Flash Attention algorithm
- Build custom CUDA kernels for transformer operations
- Optimize memory usage in attention
- Achieve 3x+ speedup over baseline

### Week 1-2: Attention Optimization

**Lesson 4.1: Transformer Architecture Deep Dive** (4 hours)
- Multi-head attention mechanism
- Feed-forward networks
- Layer normalization
- Position encodings (absolute, relative, RoPE)
- **Exercise**: Analyze attention complexity

**Lesson 4.2: Attention Bottlenecks** (4 hours)
- O(N²) memory complexity
- KV cache management
- Softmax numerical stability
- Memory bandwidth challenges
- **Lab**: Profile standard attention

**Lesson 4.3: Flash Attention Algorithm** (8 hours)
- Flash Attention theory and math
- Block-sparse tiling strategy
- Online softmax algorithm
- IO-aware design
- **Lab**: Implement Flash Attention v1

**Lesson 4.4: Flash Attention v2** (8 hours)
- Improvements over v1
- Better parallelism
- Reduced shared memory usage
- Causal masking optimization
- **Lab**: Implement Flash Attention v2

### Week 3: Custom Kernels

**Lesson 4.5: Rotary Position Embeddings** (4 hours)
- RoPE mathematical formulation
- Fusion opportunities
- Memory access optimization
- **Lab**: Implement fused RoPE kernel

**Lesson 4.6: Fused Normalization and Activation** (6 hours)
- LayerNorm optimization with Welford algorithm
- Warp-level reductions
- GELU approximation and fusion
- **Lab**: Implement fused LayerNorm + GELU

**Lesson 4.7: Integration and Benchmarking** (6 hours)
- PyTorch integration
- Backward pass implementation
- Numerical correctness validation
- Performance benchmarking
- **Lab**: End-to-end transformer with custom kernels

### Hands-On Labs

1. **Lab 4.1**: Flash Attention Implementation
   - Implement tiling and blocking
   - Handle causal masking
   - Validate correctness
   - Benchmark speedup

2. **Lab 4.2**: Custom Transformer Kernels
   - RoPE kernel
   - LayerNorm kernel
   - GELU kernel
   - Integration testing

3. **Lab 4.3**: Complete Transformer Optimization
   - Replace all operations with custom kernels
   - Profile improvements
   - Validate end-to-end accuracy

### Major Project

**Project 2: Custom CUDA Kernels for Transformer Optimization** (60 hours)

Build complete custom CUDA kernel suite for transformers:
- Flash Attention v2
- Fused RoPE
- Optimized LayerNorm
- Fused GELU

**Deliverables**:
- Full CUDA implementation
- PyTorch integration
- Comprehensive benchmarks
- Nsight profiling reports

**Performance Targets**:
- Flash Attention: 3x speedup
- Fused kernels: 3.5x speedup
- 80%+ memory bandwidth utilization

_Full project spec — planned (see module 08 for the underlying techniques)._

### Assessment

- **Quiz 4**: Transformer optimization (25 questions, 80% to pass)
- **Project 2**: Complete custom kernel implementation
- **Performance Report**: Benchmarking and profiling analysis

---

## Module 5: Model Compression (35 hours)

### Learning Outcomes

By the end of this module, you will:

- Implement post-training quantization and QAT
- Apply structured pruning techniques
- Implement knowledge distillation
- Convert models to TensorRT
- Achieve 3x speedup with 75% size reduction

### Week 1: Quantization

**Lesson 5.1: Quantization Fundamentals** (4 hours)
- Fixed-point arithmetic
- Symmetric vs asymmetric quantization
- Per-tensor vs per-channel quantization
- Calibration strategies
- **Exercise**: Manual quantization walkthrough

**Lesson 5.2: Post-Training Quantization (PTQ)** (5 hours)
- PTQ algorithms and workflow
- PyTorch quantization API
- Calibration dataset selection
- Accuracy vs performance trade-offs
- **Lab**: Implement PTQ pipeline

**Lesson 5.3: Quantization-Aware Training (QAT)** (5 hours)
- QAT theory and benefits
- Fake quantization during training
- Batch normalization handling
- Fine-tuning strategies
- **Lab**: Implement QAT workflow

**Lesson 5.4: Advanced Quantization** (4 hours)
- Mixed-precision quantization
- Sensitivity analysis
- Layer-wise quantization
- INT4 and sub-byte quantization
- **Exercise**: Quantization sensitivity analysis

### Week 2: Pruning and Distillation

**Lesson 5.5: Pruning Techniques** (5 hours)
- Magnitude-based pruning
- Structured vs unstructured pruning
- Iterative pruning with fine-tuning
- Pruning schedules
- **Lab**: Implement structured pruning

**Lesson 5.6: Knowledge Distillation** (4 hours)
- Teacher-student framework
- Loss function design
- Temperature parameter tuning
- Intermediate layer distillation
- **Lab**: Implement distillation pipeline

**Lesson 5.7: TensorRT Conversion** (4 hours)
- TensorRT overview
- ONNX export workflow
- TensorRT builder configuration
- INT8 calibration for TensorRT
- **Lab**: Convert model to TensorRT

**Lesson 5.8: End-to-End Compression** (4 hours)
- Combined optimization strategies
- Pipeline orchestration
- Validation and rollback
- **Lab**: Automated compression pipeline

### Major Project

**Project 1: Automated Model Compression Pipeline** (40 hours)

Build production-ready compression pipeline:
- PTQ and QAT implementation
- Structured pruning with fine-tuning
- Knowledge distillation (optional)
- TensorRT conversion
- Automated benchmarking

**Performance Targets**:
- 3x inference speedup
- 75% model size reduction
- <2% accuracy degradation

_Full project spec — planned (see modules 03-05 for the underlying techniques)._

### Assessment

- **Quiz 5**: Model compression techniques (25 questions, 80% to pass)
- **Project 1**: Complete compression pipeline
- **Report**: Compression performance analysis

---

## Module 6: Distributed Inference (30 hours)

### Learning Outcomes

By the end of this module, you will:

- Implement tensor parallelism for large models
- Design multi-GPU serving systems
- Optimize inter-GPU communication
- Build load balancing systems
- Achieve 90%+ scaling efficiency

### Week 1: Parallelism Strategies

**Lesson 6.1: Parallelism Overview** (3 hours)
- Data parallelism vs model parallelism
- Tensor parallelism fundamentals
- Pipeline parallelism for inference
- When to use each strategy
- **Exercise**: Calculate memory requirements

**Lesson 6.2: Tensor Parallelism Implementation** (6 hours)
- Column and row parallelism
- Attention layer partitioning
- MLP layer partitioning
- Allreduce operations
- **Lab**: Implement 2-GPU tensor parallelism

**Lesson 6.3: Communication Optimization** (5 hours)
- NCCL library usage
- Reducing communication overhead
- Overlapping compute and communication
- NVLink vs PCIe bandwidth
- **Lab**: Optimize multi-GPU communication

### Week 2: Production Multi-GPU Systems

**Lesson 6.4: Load Balancing** (4 hours)
- Request routing strategies
- Dynamic load balancing
- GPU affinity and NUMA awareness
- **Exercise**: Design load balancer

**Lesson 6.5: Multi-GPU Serving** (6 hours)
- Replication vs partitioning
- Memory management across GPUs
- Failure handling
- **Lab**: Build multi-GPU inference server

**Lesson 6.6: Scaling Analysis** (4 hours)
- Measuring scaling efficiency
- Amdahl's law for inference
- Cost-performance analysis
- **Exercise**: Multi-GPU performance analysis

**Lesson 6.7: Production Deployment** (2 hours)
- Kubernetes multi-GPU pods
- GPU resource allocation
- Monitoring multi-GPU systems

### Hands-On Labs

1. **Lab 6.1**: Tensor Parallel Attention
   - Partition attention across 2 GPUs
   - Implement allreduce
   - Benchmark communication overhead

2. **Lab 6.2**: Multi-GPU Serving System
   - Build request router
   - Implement load balancing
   - Test failover scenarios

3. **Lab 6.3**: Scaling Efficiency Analysis
   - Benchmark 1, 2, 4, 8 GPU configurations
   - Measure scaling efficiency
   - Identify bottlenecks

### Assessment

- **Quiz 6**: Distributed inference (20 questions, 80% to pass)
- **Lab Report**: Multi-GPU scaling analysis
- **Implementation**: Multi-GPU serving system

---

## Module 7: Production Deployment (25 hours)

### Learning Outcomes

By the end of this module, you will:

- Design high-throughput inference APIs
- Implement continuous batching
- Build comprehensive monitoring systems
- Optimize cost per inference
- Deploy production-ready systems

### Week 1: Serving Architecture

**Lesson 7.1: Inference API Design** (4 hours)
- REST vs gRPC for inference
- Streaming responses (SSE, WebSocket)
- Request validation and error handling
- API versioning
- **Lab**: Build FastAPI inference server

**Lesson 7.2: Continuous Batching** (6 hours)
- Dynamic batching fundamentals
- Request scheduling algorithms
- Preemption and resumption
- Fair scheduling policies
- **Lab**: Implement continuous batching engine

**Lesson 7.3: Request Management** (4 hours)
- Request queuing strategies
- Timeout handling
- Priority queues
- Backpressure mechanisms
- **Lab**: Build request manager

### Week 2: Monitoring and Optimization

**Lesson 7.4: Monitoring and Observability** (5 hours)
- Prometheus metrics design
- Grafana dashboards
- OpenTelemetry tracing
- SLO tracking and alerting
- **Lab**: Set up monitoring stack

**Lesson 7.5: Cost Optimization** (4 hours)
- Cost per inference calculation
- GPU instance selection
- Autoscaling policies
- Cost-performance trade-offs
- **Exercise**: Cost optimization analysis

**Lesson 7.6: Production Best Practices** (2 hours)
- Health checks and readiness probes
- Graceful shutdown
- Circuit breakers
- Rate limiting

### Major Project

**Project 3: High-Performance LLM Inference System** (80 hours)

Build production LLM serving system:
- Continuous batching engine
- PagedAttention implementation
- Dynamic request scheduling
- Streaming support
- Prometheus monitoring

**Performance Targets**:
- 1000+ req/sec throughput
- <100ms P99 latency
- 85%+ GPU utilization
- 70% memory savings

_Full project spec — planned (see modules 06-07 for distributed inference + deployment techniques)._

### Assessment

- **Quiz 7**: Production deployment (20 questions, 80% to pass)
- **Project 3**: Complete LLM inference system
- **Load Test Report**: Performance under production load

---

## Module 8: Advanced Topics (20 hours)

### Learning Outcomes

By the end of this module, you will:

- Implement PagedAttention
- Apply speculative decoding
- Explore cutting-edge optimizations
- Stay current with latest research

### Week 1: Advanced Techniques

**Lesson 8.1: PagedAttention** (6 hours)
- Virtual memory for KV cache
- Page table management
- Copy-on-write for prefix sharing
- Memory pool allocation
- **Lab**: Implement PagedAttention

**Lesson 8.2: Speculative Decoding** (6 hours)
- Speculative execution theory
- Draft model selection
- Verification and acceptance
- Multi-token generation
- **Lab**: Implement speculative decoding

**Lesson 8.3: Emerging Techniques** (4 hours)
- Flash Decoding
- INT4 quantization
- Mixture of Experts optimization
- Sparse attention patterns

**Lesson 8.4: Research to Production** (4 hours)
- Reading optimization papers
- Implementing research techniques
- Validating performance claims
- **Exercise**: Implement technique from recent paper

### Assessment

- **Quiz 8**: Advanced topics (15 questions, 80% to pass)
- **Implementation**: PagedAttention or Speculative Decoding
- **Research Review**: Analysis of recent optimization paper

---

## Assessment Framework

### Continuous Assessment

**Weekly Quizzes** (20% of final grade)
- 8 module quizzes
- Multiple choice and short answer
- 80% passing score required
- Unlimited retakes allowed

**Lab Exercises** (20% of final grade)
- Hands-on coding and analysis
- Graded on correctness and performance
- Peer review component

### Major Projects (50% of final grade)

**Project 1: Model Compression** (15%)
- Implementation correctness: 40%
- Performance targets: 40%
- Documentation: 20%

**Project 2: Custom CUDA Kernels** (20%)
- Kernel correctness: 30%
- Performance targets: 40%
- Profiling analysis: 20%
- Integration: 10%

**Project 3: LLM Inference System** (25%)
- System design: 20%
- Performance targets: 40%
- Production readiness: 20%
- Documentation: 20%

### Practical Examinations (10% of final grade)

**Compression Exam** (Module 5)
- Given model and targets, implement compression
- 4-hour time limit
- Must meet all performance targets

**CUDA Exam** (Module 4)
- Implement CUDA kernel from specification
- 3-hour time limit
- Correctness and performance grading

**Inference Exam** (Module 7)
- Deploy serving system meeting SLAs
- 4-hour time limit
- Load testing validation

### Final Certification

**Requirements**:
- Overall score ≥80%
- All quizzes passed (≥80%)
- All projects submitted and passing
- All practical exams passed

**Certificate Levels**:
- **Pass** (80-89%): AI/ML Performance Engineer Certificate
- **Merit** (90-94%): Certificate with Merit
- **Distinction** (95-100%): Certificate with Distinction

---

## Learning Resources by Module

### Module 1: GPU Fundamentals
- NVIDIA CUDA C++ Programming Guide (Chapters 1-3)
- "Programming Massively Parallel Processors" (Chapters 1-4)
- NVIDIA GPU Architecture Whitepapers

### Module 2: CUDA Programming
- NVIDIA CUDA C++ Programming Guide (Chapters 4-8)
- "Programming Massively Parallel Processors" (Chapters 5-9)
- PyTorch C++ Extension Tutorial

### Module 3: Performance Profiling
- NVIDIA Nsight Compute User Guide
- NVIDIA Nsight Systems User Guide
- "Performance Analysis and Tuning for General Purpose Graphics Processing Units"

### Module 4: Transformer Optimization
- Flash Attention papers (v1 and v2)
- "Attention is All You Need" (Vaswani et al.)
- RoFormer paper (RoPE)

### Module 5: Model Compression
- Qualcomm Quantization White Paper
- "Learning both Weights and Connections" (Pruning)
- NVIDIA TensorRT Documentation

### Module 6: Distributed Inference
- Megatron-LM papers
- NCCL Documentation
- "Efficient Large-Scale Language Model Training"

### Module 7: Production Deployment
- vLLM Paper
- FastAPI Documentation
- Prometheus Best Practices

### Module 8: Advanced Topics
- Latest papers on arXiv (cs.LG, cs.DC)
- MLSys conference proceedings
- NVIDIA Developer Blog

---

## Time Management Guide

### Full-Time Schedule (10-12 weeks)

**Weeks 1-2**: Modules 1-2 (GPU Fundamentals, CUDA)
- 4-5 hours/day of lessons and labs
- Weekend: Practice and review

**Week 3**: Module 3 (Performance Profiling)
- Deep dive into profiling tools
- Practice on real models

**Weeks 4-5**: Project 1 (Model Compression)
- 8 hours/day focused implementation
- Mid-week check-in with mentor

**Weeks 6-7**: Module 4 (Transformer Optimization)
- 5 hours/day lessons
- Daily CUDA coding practice

**Weeks 8-10**: Project 2 (Custom CUDA Kernels)
- Most challenging project
- Pair programming sessions recommended

**Weeks 11-12**: Modules 5-6 (Compression, Distributed)
- Combine theoretical learning with practical labs

**Weeks 13-16**: Project 3 (LLM Inference)
- Capstone project
- Weekly progress presentations

**Weeks 17-18**: Modules 7-8 (Production, Advanced)
- Finalize all projects
- Prepare for practical exams

### Part-Time Schedule (20-25 weeks)

**Daily Commitment**: 3-4 hours
**Weekend**: 6-8 hours

Double the full-time schedule, with:
- Slower module progression
- More spaced repetition
- Extended project timelines

### Self-Paced Guidelines

**Minimum Pace**: 2 hours/day, 10 hours/week
**Maximum Pace**: 8-10 hours/day (burnout risk beyond this)

**Recommended Milestones**:
- Complete Module 1 in first 2 weeks
- First project within 6 weeks
- Second project by week 12
- Final project by week 20

---

## Support and Mentorship

### Office Hours

**Live Q&A Sessions**:
- Wednesdays 2-3 PM PT: CUDA and GPU Programming
- Fridays 11 AM-12 PM PT: Projects and General Questions

**Recorded Sessions**: Available 24 hours after live session

### Discussion Forums

- **General Discussion**: Course logistics and general questions
- **Technical Help**: Debugging and implementation help
- **Project Reviews**: Peer feedback on projects
- **Research Papers**: Discuss latest optimization techniques

### Mentorship Program

**1-on-1 Mentoring** (optional, paid add-on):
- Weekly 30-minute sessions
- Project code reviews
- Career guidance
- Interview preparation

**Peer Study Groups**:
- Self-organized study groups
- Code review circles
- Project collaboration

---

## Continuous Learning

### Staying Current

**Weekly Reading**:
- arXiv papers (cs.LG, cs.DC, cs.PF)
- NVIDIA Developer Blog
- MLSys conference proceedings

**Monthly Challenges**:
- Implement optimization from recent paper
- Contribute to open-source optimization libraries
- Write blog post about learned technique

**Quarterly Goals**:
- Present at meetup or conference
- Publish benchmark results
- Contribute to research discussions

### Advanced Topics for Further Study

After completing this course:

1. **Multi-Node Distributed Inference**
2. **Custom Hardware Accelerators (TPU, Graphcore)**
3. **Compiler Optimizations (XLA, TVM)**
4. **Automated Neural Architecture Search for Efficiency**
5. **Energy-Efficient ML Systems**

---

## Certification Maintenance

**Certificate Validity**: 2 years

**Renewal Requirements**:
- Complete 20 hours of continuing education
- Submit one advanced project
- Pass recertification quiz on new techniques

**Continuing Education Options**:
- Advanced workshops
- Research paper implementations
- Contributions to open-source projects
- Conference attendance (MLSys, NVIDIA GTC)

---

**Ready to begin your journey?**

Start with [Module 001: GPU Fundamentals →](modules/mod-001-gpu-fundamentals/)
