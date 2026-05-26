# Lecture 02: Pruning + Distillation

## Pruning

Set unimportant weights to zero. Sparse computation skips them.

| Type | Effect | Hardware acceleration |
|---|---|---|
| Unstructured | individual weights zero | minimal speedup on GPU |
| Structured (2:4) | every 4 weights have at most 2 nonzero | 2× on A100+ via sparse tensor cores |
| Block sparse | larger contiguous blocks zero | requires custom kernels |

In practice, 2:4 sparsity is the only widely-deployable variant: NVIDIA
Sparse Tensor Cores natively support it.

```python
from torch.sparse import sparse_compressed_tensor
# Train with sparsity-aware methods; export pruned model
```

Sparsity stacks with quantization: int8 + 2:4 sparse = ~6× memory savings.

## Distillation

A small "student" model is trained to mimic a large "teacher" model. Student
inherits the teacher's behavior with smaller params + faster inference.

Loss = α × CE(student_logits, ground_truth) + β × KL(student_softmax, teacher_softmax)

Common shapes:
- Llama-70B → Llama-7B: 10× smaller, 95%+ benchmark score retained
- BERT-large → DistilBERT: 40% smaller, 97% accuracy
- Custom distillations: depends heavily on task

The catch: distillation requires the teacher to actually have learned the
task well. Don't distill from a teacher that itself is bad.

## When pruning vs quantization vs distillation

| Goal | Method |
|---|---|
| Smaller inference memory | quantization (AWQ for large models) |
| Faster compute | pruning (only on sparse-tensor-core HW) |
| Smaller weights AND different architecture | distillation |
| All of the above | combine |
