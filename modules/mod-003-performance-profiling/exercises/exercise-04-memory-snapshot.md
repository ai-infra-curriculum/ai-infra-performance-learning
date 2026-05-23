# Ex 04: Memory Snapshot

Use `torch.cuda.memory._record_memory_history` on a GPT-2 training step.
Generate snapshot.bin. Upload to <https://pytorch.org/memory_viz>.

Identify:
- Peak memory used
- Largest single allocation
- Memory reclaimed at end of step (steady-state?)
- Recommendation for an optimization (grad-ckpt? 8bit optimizer? smaller batch?)

Companion: engineer-solutions/mod-107 ex-07.
