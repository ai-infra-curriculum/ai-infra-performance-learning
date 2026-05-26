# Lecture 02: Memory Coalescing + Shared Memory

## Coalescing

Threads in a warp access memory simultaneously. If consecutive threads access
consecutive memory addresses (aligned), the GPU coalesces them into one
128-byte transaction. Otherwise, each thread takes a separate transaction.

```cpp
// Coalesced (good): each thread reads its own element from a row
c[blockIdx.x * blockDim.x + threadIdx.x] = ...

// Strided (bad): thread i reads element i * 32 — separate transactions
c[threadIdx.x * 32] = ...
```

## Shared memory

Shared memory is per-block scratchpad — ~1000× faster than global memory.
Use it for data reused within a block.

### Tiled matmul (classic example)

Naive matmul reads each A and B element multiple times from global memory.
Tile-based matmul loads tiles into shared memory once + each block reuses
them many times.

```cpp
__global__ void matmul_tiled(int N, const float* A, const float* B, float* C) {
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];
    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < N / TS; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TS + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TS + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < TS; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

This typically reaches 50-70% of cuBLAS. The remaining 30% comes from Tensor
Core fusion + warp specialization, which is beyond a hand-written kernel.

## Bank conflicts

Shared memory is divided into 32 banks. If multiple threads in a warp access
the same bank simultaneously, accesses serialize.

Fix by padding:
```cpp
__shared__ float As[TS][TS + 1];   // pad to avoid bank conflicts
```

## Reading

- [GPU Memory Bandwidth (Mark Harris)](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [Shared Memory Bank Conflicts](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
