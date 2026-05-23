# Lecture 02: NVLink + InfiniBand + GPUDirect

## NVLink

NVIDIA proprietary high-speed link between GPUs (and to NVSwitch).
- H100 NVLink Gen4: 900 GB/s aggregate per GPU
- 8 GPUs / NVSwitch on a node share full bandwidth
- vs PCIe Gen4 16-lane: ~64 GB/s; >10× slower

NVLink is what makes TP fast within a node. Without it, TP is a bad idea.

## InfiniBand

Across-node fabric:
- HDR (200 Gb/s) or NDR (400 Gb/s) per port
- Multi-rail for higher bandwidth
- RDMA: zero-copy memory access between nodes

PP across nodes uses IB to avoid copying activations through CPU memory.

## GPUDirect RDMA

GPU → NIC → remote NIC → remote GPU, all without CPU involvement.
Critical for multi-node training; enabled via `nccl` + IB drivers.

## Diagnosing fabric issues

- `ib_send_bw`: measure raw IB bandwidth
- `nccl-tests`: measure all-reduce throughput across GPUs
- `nvidia-smi topo -m`: shows interconnect topology

## When fabric matters

- Single-node training: NVLink topology
- Multi-node training: IB bandwidth + GPUDirect
- Inference (single replica per node): less critical
