# Ex 05: Custom PyTorch Extension

Package the kernel from ex-02 as a `torch.utils.cpp_extension`. Add autograd:
forward + backward (matmul backward = same matmul with transposed inputs).

Use it in a real model (small linear layer); confirm gradients match
`torch.matmul`.

Companion: engineer-solutions/mod-107 ex-02.
