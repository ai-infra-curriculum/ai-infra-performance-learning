# Ex 02: Int8 Static Quantization (PyTorch)

Take ResNet-50 (torchvision pretrained weights). Apply post-training static
int8 quantization with a calibration pass, then verify accuracy and measure
the size and latency wins on CPU. Static quant fuses and calibrates ahead of
time, so observers must see representative data before you convert.

## Tasks

1. Load `torchvision.models.resnet50(weights=IMAGENET1K_V2)` and record the
   fp32 baseline: top-1 accuracy on the ImageNet validation set (or a fixed
   5k-image subset), file size, and mean CPU latency at batch size 1.
2. Fuse modules (`conv + bn + relu`) with `torch.ao.quantization.fuse_modules`
   before inserting observers.
3. Set a static qconfig (`get_default_qconfig("x86")` / FBGEMM on x86),
   call `prepare`, then run a calibration loop over 100-500 representative
   images so the observers collect activation ranges.
4. Call `convert` to produce the int8 model. Use per-channel weight
   quantization; note the difference vs per-tensor if you try both.
5. Re-measure accuracy, on-disk size (`state_dict` saved to file), and CPU
   latency. Pin threads (`torch.set_num_threads`) so latency numbers are
   stable and comparable.

## Acceptance criteria

- Model size reduced by roughly 3.5-4x (fp32 to int8).
- CPU latency improved by at least 1.5x at batch size 1 on the same machine.
- Top-1 accuracy drop within 1.0 absolute percentage point of the fp32
  baseline. If it is worse, expand or rebalance the calibration set and
  retry before reporting.

Document baseline vs int8 (accuracy, size, p50/p99 latency), the calibration
set size, and per-channel vs per-tensor findings in `INT8_REPORT.md`.

Reference: [PyTorch static quantization
tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html).
