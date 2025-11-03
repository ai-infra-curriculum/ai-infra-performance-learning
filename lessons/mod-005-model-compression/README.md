# Module 5: Model Compression and Quantization

## Overview

Learn techniques to reduce model size and accelerate inference through quantization, pruning, and distillation while maintaining accuracy.

**Duration**: 4-5 weeks (20-25 hours)

## Learning Objectives

- Implement INT8/INT4 quantization for 2-4x speedup
- Apply pruning to reduce model parameters
- Use knowledge distillation to create smaller models
- Deploy compressed models in production

## Key Topics

### 1. Quantization Techniques
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- INT8, INT4, and mixed precision
- Dynamic vs static quantization

### 2. Pruning Methods
- Magnitude-based pruning
- Structured vs unstructured pruning
- Gradual pruning schedules
- Fine-tuning after pruning

### 3. Knowledge Distillation
- Teacher-student training
- Distillation objectives
- Compression ratios and quality trade-offs

### 4. Production Deployment
- TensorRT integration
- ONNX Runtime optimization
- Hardware-specific acceleration

## Expected Outcomes

- **Speed**: 2-4x faster inference
- **Memory**: 4-8x reduction
- **Accuracy**: <2% quality loss

See `lecture-notes/` for detailed content and `exercises/` for hands-on labs.
