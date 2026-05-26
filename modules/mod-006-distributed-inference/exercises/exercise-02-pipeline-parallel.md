# Ex 02: Pipeline-Parallel for 70B+

Deploy a 70B model across 2 nodes (4 GPUs each, NVLink within node, IB across).
Use `--tensor-parallel-size 4 --pipeline-parallel-size 2`.

Compare to single-node TP=8 (if you have an 8-GPU node). Where does PP win
(throughput) vs lose (latency)?
