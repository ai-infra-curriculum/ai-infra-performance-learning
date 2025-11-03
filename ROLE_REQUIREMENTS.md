# Role Requirements: AI/ML Performance Engineer

**Level:** 2.5
**Experience Level:** 3-6 years in performance engineering, GPU programming, or ML infrastructure, with demonstrated expertise in model optimization
**Market Demand:** Very High - Critical role as inference costs dominate ML budgets (90% of total). High demand from companies deploying large models (LLMs, vision models) at scale. Shortage of engineers with both ML knowledge and low-level optimization skills. Essential for cost-effective AI deployment.
**Salary Range (US, 2025):** $150,000 - $280,000 (median $195,000)

## Core Responsibilities
- Optimize ML model inference latency and throughput
- Implement model compression techniques (quantization, pruning, distillation)
- Optimize GPU utilization and reduce inference costs
- Profile and optimize model training performance
- Develop custom CUDA kernels for performance-critical operations
- Implement efficient batching and caching strategies
- Optimize memory usage for large model deployments
- Benchmark and compare different optimization techniques
- Design auto-scaling strategies based on performance metrics
- Analyze cost-performance trade-offs for deployment options
- Optimize data loading and preprocessing pipelines
- Implement mixed-precision training and inference
- Tune hyperparameters for optimal performance
- Work with hardware vendors on performance improvements
- Document performance optimization best practices

## Technical Skills
### Required
- Deep understanding of GPU architectures and CUDA programming
- Expertise in model optimization techniques (quantization, pruning)
- Profiling and performance analysis skills
- Strong knowledge of ML frameworks (PyTorch, TensorFlow, ONNX)
- Experience with inference optimization (TensorRT, ONNX Runtime, vLLM)
- Understanding of hardware-software co-optimization
- Knowledge of numerical precision (FP32, FP16, INT8, INT4)
- Experience with distributed inference systems
- Strong programming in Python and C++
- Understanding of memory hierarchies and caching

### Preferred
- Custom CUDA kernel development experience
- Knowledge of specialized hardware (TPUs, AWS Inferentia, Trainium)
- Experience with model serving frameworks (TensorRT-LLM, vLLM, SGLang)
- Understanding of transformer architecture optimizations
- Knowledge of compiler optimizations (TVM, XLA)
- Experience with attention mechanism optimizations (Flash Attention)
- Profiling tools expertise (NVIDIA Nsight, PyTorch Profiler)
- Knowledge of sparsity and structured pruning
- Experience with low-bit quantization (FP4, NF4, AWQ, GPTQ)
- Understanding of KV cache optimization for LLMs

## Soft Skills
- Analytical and data-driven decision making
- Attention to detail for performance metrics
- Balancing multiple optimization objectives
- Communication of technical trade-offs
- Collaboration with ML researchers and engineers
- Persistence in optimization challenges
- Creativity in problem-solving
- Documentation of optimization techniques
- Knowledge sharing and mentoring

## Prerequisites
- Bachelor's or Master's degree in Computer Science, Electrical Engineering, or related field
- Strong foundation in computer architecture and systems
- Experience with high-performance computing or GPU programming
- Understanding of machine learning fundamentals
- Proficiency in C++ and Python
- Experience with performance profiling and optimization

## Day-to-Day Activities
- Profiling model performance and identifying bottlenecks
- Implementing and testing optimization techniques
- Running benchmarks and analyzing results
- Writing custom kernels or optimized operations
- Collaborating with ML teams on model architecture efficiency
- Tuning inference configurations for production deployments
- Monitoring production model performance
- Researching new optimization techniques and tools
- Creating performance reports and recommendations
- Optimizing batch sizes and concurrency settings
- Testing different quantization strategies

## Success Metrics
- Inference latency reduction (e.g., 50%+ improvement)
- Throughput improvement (requests per second)
- GPU utilization increase (target 80%+ for training, 70%+ for inference)
- Cost reduction in inference infrastructure ($/request)
- Memory footprint reduction for model serving
- Model accuracy retention after optimization (minimal degradation)
- Training time reduction
- Energy efficiency improvements
- Successful deployment of optimized models to production

## Common Challenges
- Balancing performance with model accuracy
- Managing accuracy-latency-cost trade-offs
- Optimizing diverse model architectures and sizes
- Keeping up with rapidly evolving optimization techniques
- Addressing long-tail latency issues
- Optimizing for different hardware targets
- Managing complexity of optimization pipelines
- Validating optimizations don't introduce correctness issues
- Handling variable workload patterns and batch sizes

## Key Technologies
- **Programming Languages:** Python (ML and automation), C++ (performance-critical code and CUDA), CUDA (GPU programming), C (for low-level optimization), Assembly (beneficial for deep optimization)
- **ML Frameworks:** PyTorch (with optimization extensions), TensorFlow and TFLite, ONNX and ONNX Runtime, JAX (for performance-critical research), Hugging Face Optimum
- **Infrastructure:** NVIDIA GPUs (A100, H100, L4, T4), Cloud GPU instances (AWS, GCP, Azure), Specialized accelerators (TPUs, Inferentia, Trainium), Kubernetes for scaled deployments
- **DevOps Tools:** Docker for containerized benchmarks, CI/CD for performance testing, Infrastructure as Code for test environments
- **MLOps Tools:** Model optimization platforms, Experiment tracking for optimization runs, Model serving frameworks
- **Monitoring & Observability:** NVIDIA DCGM for GPU monitoring, Custom performance dashboards, Profiling data collection and analysis, Performance regression testing
- **Data Processing:** Efficient data loaders (PyTorch DataLoader, tf.data), Data preprocessing optimization, Caching and prefetching strategies
- **GPU Tools:** NVIDIA TensorRT and TensorRT-LLM, CUDA Toolkit and libraries (cuDNN, cuBLAS, CUTLASS), NVIDIA Nsight (Compute, Systems), PyTorch Profiler, TensorFlow Profiler, vLLM, SGLang, TGI (Text Generation Inference), Flash Attention and optimization libraries, Triton (OpenAI's language for GPU kernels), NCCL for multi-GPU communication
- **Other Tools:** Quantization tools (GPTQ, AWQ, AutoAWQ, bitsandbytes), Model compression frameworks, Compilers (TVM, XLA), Benchmarking frameworks, A/B testing for performance comparisons, KV cache optimization techniques, PagedAttention and continuous batching

## Recommended Certifications
- NVIDIA Deep Learning Institute certifications
- CUDA programming certifications
- AWS Certified Machine Learning – Specialty
- Parallel programming certifications
- Computer architecture certifications
- Performance engineering certifications
