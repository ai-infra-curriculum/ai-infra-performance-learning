# Lecture 01: Kernels + Launch Configuration

## The shape of a CUDA kernel

```cpp
__global__ void add(int n, const float* a, const float* b, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Host launch:
int block = 256;
int grid = (n + block - 1) / block;
add<<<grid, block>>>(n, a_d, b_d, c_d);
```

## Launch geometry concepts

- **Thread**: smallest unit; runs one instance of the kernel
- **Block**: group of threads (up to 1024); can use shared memory together
- **Grid**: collection of blocks; the full kernel launch

Choose `block` to be a multiple of warp size (32) to avoid wasted threads.

## Picking block size

| Workload | Typical block |
|---|---|
| Memory-bound, simple | 128, 256 |
| Compute-bound, register-heavy | 64, 128 |
| Reduction / cooperative | 256 (full warps) |

Tune empirically. The default starting point is 256.

## Occupancy

Threads / SM is bounded by:
- 32-bit registers per SM (registers per thread × threads)
- Shared memory per SM (per-block × blocks)
- Max threads per block (typically 1024)
- Max blocks per SM

Use `cudaOccupancyMaxPotentialBlockSize` to get a starting point. Then
benchmark variations.

## Common mistakes

- **Block size not a multiple of warp size** — 32-49 threads waste 15 warps
- **Branch divergence within a warp** — `if (threadIdx.x < 16)` splits work
- **Bank conflicts in shared memory** — threads in the same warp accessing
  the same memory bank serialize
- **Insufficient grid size** — grid < SM count leaves SMs idle

## Reading

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/)
