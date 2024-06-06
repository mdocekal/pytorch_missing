#include <torch/extension.h>
#include <ATen/Parallel.h>


torch::Tensor indices_dot_product_forward_cuda(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor indices);

std::tuple<torch::Tensor, torch::Tensor> indices_dot_product_backward_cuda(
    torch::Tensor grad,
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor indices);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor indices_dot_product(torch::Tensor x, torch::Tensor y, torch::Tensor indices) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(indices);

    return indices_dot_product_forward_cuda(x, y, indices);
}

std::tuple<torch::Tensor, torch::Tensor> indices_dot_product_backward(torch::Tensor grad, torch::Tensor x, torch::Tensor y, torch::Tensor indices) {
    CHECK_INPUT(grad);
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(indices);

    return indices_dot_product_backward_cuda(grad, x, y, indices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &indices_dot_product, "Indices dot product forward (cuda)");
  m.def("backward", &indices_dot_product_backward, "Indices dot product backward (cuda)");
}
