# Module 02: CUDA Programming — Quiz

15 questions. 75% pass.

### 1. The recommended block size for most kernels:
- [x] a) Multiple of 32 (warp size); 128 or 256 typical
- [ ] b) Always 1024
- [ ] c) Prime number for spreading
- [ ] d) Power of 2 strictly

### 2. Coalesced memory access means:
- [x] a) Consecutive threads access consecutive memory addresses
- [ ] b) All threads read the same address
- [ ] c) Atomic operations
- [ ] d) Texture memory reads

### 3. Shared memory is faster than global memory by approximately:
- [x] a) 100-1000×
- [ ] b) 2-3×
- [ ] c) Same speed
- [ ] d) 10× slower

### 4. Bank conflicts occur when:
- [x] a) Multiple threads in a warp access the same shared memory bank simultaneously
- [ ] b) Bank capacity is exceeded
- [ ] c) The kernel doesn't use shared memory
- [ ] d) The grid is too large

### 5. Warp divergence:
- [x] a) Threads in the same warp take different branches; execution serializes
- [ ] b) Faster than coalesced execution
- [ ] c) Required for correctness
- [ ] d) Only happens at block boundaries

### 6. Occupancy is bounded by (pick all):
- [x] a) Registers per thread
- [x] b) Shared memory per block
- [x] c) Max threads per block
- [x] d) Max blocks per SM

### 7. Tiled matmul gets close to cuBLAS because:
- [x] a) Data reuse via shared memory amortizes global memory cost
- [ ] b) It uses fewer threads
- [ ] c) It runs on the CPU
- [ ] d) cuBLAS calls it internally

### 8. PyTorch C++ extensions compile via:
- [x] a) `torch.utils.cpp_extension.CUDAExtension` + setuptools
- [ ] b) gcc directly
- [ ] c) nvcc only
- [ ] d) make

### 9. `torch.cuda.synchronize()` is necessary in benchmarks because:
- [x] a) Kernel launches are async; without sync, you measure launch latency not execution
- [ ] b) It's faster
- [ ] c) It triggers JIT
- [ ] d) Required by setuptools

### 10. Block size NOT being a multiple of warp size causes:
- [x] a) Wasted compute (idle threads in partial warps)
- [ ] b) Compilation error
- [ ] c) Runtime crash
- [ ] d) Improved performance

### 11. The first parameter of `<<<grid, block>>>`:
- [x] a) Grid dimensions (number of blocks)
- [ ] b) Block dimensions
- [ ] c) Thread index
- [ ] d) Shared memory size

### 12. `__shfl_down_sync` enables:
- [x] a) Intra-warp data exchange without shared memory
- [ ] b) Atomic add
- [ ] c) Block-level reduction
- [ ] d) Memory coalescing

### 13. `cudaOccupancyMaxPotentialBlockSize` returns:
- [x] a) A starting-point block size; verify with benchmarks
- [ ] b) The exact optimal block size
- [ ] c) Number of SMs
- [ ] d) Warp size

### 14. Padding shared memory tiles by 1 (`[TS][TS+1]`) prevents:
- [x] a) Bank conflicts
- [ ] b) Memory leaks
- [ ] c) Register spills
- [ ] d) Compilation errors

### 15. First call to a CUDA kernel is slower than subsequent calls because of:
- [x] a) JIT compilation + cuBLAS init + CUDA graph capture overhead
- [ ] b) Always; GPUs are slow on the first call
- [ ] c) Memory allocation
- [ ] d) Network latency

---

Answers: 1.a 2.a 3.a 4.a 5.a 6.all 7.a 8.a 9.a 10.a 11.a 12.a 13.a 14.a 15.a
