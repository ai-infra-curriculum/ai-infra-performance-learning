# Lecture 03: PyTorch C++ Extensions

## Why

You wrote a CUDA kernel. Calling it from Python without ceremony is the goal.

PyTorch's `torch.utils.cpp_extension` builds C++ + CUDA sources into a Python
module that:
- Accepts `torch.Tensor` arguments
- Returns `torch.Tensor` outputs
- Plays well with autograd if you wire backward

## The wrapper

```cpp
#include <torch/extension.h>

extern "C" void launch_add(int n, const float* a, const float* b, float* c);

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be CUDA");
    TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
    auto c = torch::empty_like(a);
    launch_add(a.numel(), a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>());
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Custom add (CUDA)");
}
```

## setup.py

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_ops",
    ext_modules=[CUDAExtension("my_ops", ["my_ops.cpp", "add_kernel.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
```

## Build + use

```bash
pip install -e .
python -c "import my_ops, torch; print(my_ops.add(torch.ones(10).cuda(), torch.ones(10).cuda()))"
```

## Benchmarking vs torch.add

Use `torch.cuda.synchronize()` between launches. Run inside a warm loop;
report median + p95. Don't trust single-run numbers — kernel JIT, cuBLAS
init, and CUDA graph capture all skew the first call.

## Companion

[engineer-solutions/mod-107 ex-02](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-107-gpu-computing/exercise-02-cuda-kernel) — working setup.py + bench.py.
