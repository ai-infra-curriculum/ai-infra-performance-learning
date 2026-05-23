# Module 03: Performance Profiling — Quiz

10 questions. 70% pass.

### 1. nsys vs ncu:
- [x] a) nsys for timeline; ncu for per-kernel deep dive
- [ ] b) Synonyms
- [ ] c) nsys for memory; ncu for compute
- [ ] d) nsys is older

### 2. NVTX ranges:
- [x] a) Annotate code regions so nsys shows them on the timeline
- [ ] b) Replace nsys
- [ ] c) Track memory only
- [ ] d) Required for compilation

### 3. The roofline model classifies kernels into:
- [x] a) Memory-bound (below the slope) vs compute-bound (at the ceiling)
- [ ] b) Fast vs slow
- [ ] c) PyTorch vs CUDA
- [ ] d) Single-threaded vs parallel

### 4. If GPU Speed of Light shows both compute % and memory % low:
- [x] a) Kernel isn't using the GPU well; few warps or divergence
- [ ] b) Add more memory
- [ ] c) Use bigger batch
- [ ] d) Switch to CPU

### 5. PyTorch memory snapshot uploads to:
- [x] a) pytorch.org/memory_viz for interactive exploration
- [ ] b) Print to console only
- [ ] c) PagerDuty
- [ ] d) The Apple App Store

### 6. Gradient checkpointing trades:
- [x] a) ~30% recompute time for ~50% activation memory
- [ ] b) Speed for accuracy
- [ ] c) Accuracy for speed
- [ ] d) Nothing — free win

### 7. AdamW8bit reduces:
- [x] a) Optimizer state by ~4×
- [ ] b) Training time
- [ ] c) Model accuracy
- [ ] d) Gradient magnitude

### 8. FlashAttention's main benefit:
- [x] a) Memory: quadratic → linear in sequence length, often faster
- [ ] b) Always slower but more accurate
- [ ] c) Required for GPT
- [ ] d) Replaces softmax

### 9. Small kernel called many times:
- [x] a) Launch overhead dominates; batch or use CUDA graphs
- [ ] b) Always good
- [ ] c) Can't be optimized
- [ ] d) Required by autograd

### 10. To benchmark accurately:
- [x] a) Warm up + use torch.cuda.synchronize + report median over many runs
- [ ] b) Single call is enough
- [ ] c) Time wall clock from process start
- [ ] d) Trust the IDE timer

---

Answers: all `a`
