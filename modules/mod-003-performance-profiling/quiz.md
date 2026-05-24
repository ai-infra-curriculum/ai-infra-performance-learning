# Module 03: Performance Profiling — Quiz

10 questions. 70% pass.

### 1. nsys vs ncu — what's the right split of work?
- [ ] a) They are synonyms; pick one
- [ ] b) nsys for memory; ncu for compute
- [ ] c) nsys is the older deprecated tool
- [x] d) nsys for system-level timeline + workflow; ncu for per-kernel deep dive

### 2. NVTX ranges in your training code:
- [x] a) Annotate code regions so nsys shows them as named bands on the timeline
- [ ] b) Replace nsys; they emit their own trace
- [ ] c) Only track memory allocations
- [ ] d) Are required for the CUDA compiler

### 3. The roofline model classifies kernels into:
- [ ] a) Fast vs slow
- [ ] b) PyTorch ops vs raw CUDA
- [ ] c) Single-threaded vs parallel
- [x] d) Memory-bound (under the slope) vs compute-bound (under the ceiling)

### 4. If GPU Speed-of-Light shows both compute % and memory % below 30%, the diagnosis is usually:
- [x] a) The kernel isn't using the GPU well — too few warps or warp divergence
- [ ] b) Add more memory to the GPU
- [ ] c) Use a bigger batch size automatically
- [ ] d) Switch the workload to CPU

### 5. PyTorch memory snapshots can be uploaded to:
- [ ] a) Console-only output; no visualization
- [ ] b) PagerDuty for incident routing
- [ ] c) The Apple App Store
- [x] d) pytorch.org/memory_viz for interactive exploration

### 6. Gradient checkpointing trades:
- [ ] a) Speed for accuracy
- [ ] b) Accuracy for speed
- [ ] c) Nothing — it's a free optimization
- [x] d) ~30% more recompute time for ~50% less activation memory

### 7. AdamW8bit (from bitsandbytes) reduces:
- [x] a) Optimizer state memory by roughly 4× (fp32 → int8)
- [ ] b) Wall-clock training time
- [ ] c) Final model accuracy
- [ ] d) Gradient magnitude

### 8. FlashAttention's primary benefit:
- [ ] a) Always slower but more numerically accurate
- [ ] b) Required for GPT-style architectures
- [ ] c) Replaces the softmax function
- [x] d) Memory: O(N²) → O(N) in sequence length; often faster too

### 9. A kernel that's small but called thousands of times per second is bottlenecked by:
- [x] a) Per-launch overhead — batch the work or capture as a CUDA graph
- [ ] b) Nothing; small kernels are always good
- [ ] c) Cannot be optimized
- [ ] d) Required by autograd machinery

### 10. To benchmark GPU code accurately:
- [ ] a) A single call is sufficient
- [ ] b) Time wall-clock from process start to end
- [ ] c) Trust the IDE / debugger timer
- [x] d) Warm up the kernel, use `torch.cuda.synchronize()` between runs, report the median

---

## Answer key + rationale

1. **d** — nsys gives "where in the workflow is time going"; ncu zooms into one kernel for tuning.
2. **a** — NVTX is the marker API the profilers respect for human-readable timeline labels.
3. **d** — Roofline is the canonical performance classification.
4. **a** — Both metrics low = under-utilization, not over-loaded. More warps or fixing divergence is the fix.
5. **d** — memory_viz is the canonical Python/PyTorch memory visualizer.
6. **d** — Recompute on backward pass eliminates the need to store activations, trading FLOPs for bytes.
7. **a** — 8-bit optimizer states fit Adam moments in ~1/4 the memory of fp32.
8. **d** — FlashAttention tiles attention computation so intermediate matrices fit in SRAM; memory drops from quadratic to linear in seq length.
9. **a** — Launch overhead is ~few microseconds; small kernels hit that overhead per call.
10. **d** — Without warmup + sync + median you measure noise + JIT compilation, not steady-state.
