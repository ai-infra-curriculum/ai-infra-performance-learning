# Ex 03: NCCL Tests

Run nccl-tests (`all_reduce_perf`) on your cluster. Compare:
- Single-node (NVLink only)
- Multi-node (NVLink within + IB across)

Report all-reduce bandwidth at 1B-element messages. Diagnose any anomalies.
