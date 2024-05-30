#include <torch/extension.h>
#include <ATen/Parallel.h>

/**
 * Computes dot product for each edge.
 * Example:
 *      x = [
 *          [1, 2],
 *          [3, 4],
 *          [5, 6]
 *      ]
 *      y = [
 *          [7, 8],
 *          [9, 10],
 *          [11, 12]
 *      ]
 *      indices = [
 *          [0, 0, 1]
 *          [2, 1, 2]
 *      ]
 *
 *      -> [1*11 + 2*12, 1*7 + 2*8, 3*9 + 4*10] = [35, 23, 67]
 * @param x: matrix of source vectors
 * @param y: matrix of destination vectors
 * @param indices: indices tensor
 *
 * @return The dot product for each edge.
 */
torch::Tensor indices_dot_product(torch::Tensor x, torch::Tensor y, torch::Tensor indices) {
    auto res = torch::zeros({indices.size(1)}, x.options());
    auto from = indices.index({0});
    auto to = indices.index({1});

    at::parallel_for(0, indices.size(1), 1, [&](int64_t begin, int64_t end) {
        for (auto i = begin; i < end; i++) {
            res[i] = (x[from[i]] * y[to[i]]).sum();
        }
    });
    return res;
}

std::tuple<torch::Tensor, torch::Tensor> indices_dot_product_backward(torch::Tensor grad, torch::Tensor x, torch::Tensor y, torch::Tensor indices) {
    auto grad_x = torch::zeros_like(x);
    auto grad_y = torch::zeros_like(y);

    auto from = indices.index({0});
    auto to = indices.index({1});

    at::parallel_for(0, indices.size(1), 1, [&](int64_t begin, int64_t end) {
        for (auto i = begin; i < end; i++) {
            grad_x[from[i]] += grad[i] * y[to[i]];
            grad_y[to[i]] += grad[i] * x[from[i]];
        }
    });
    return std::make_tuple(grad_x, grad_y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &indices_dot_product, "Indices dot product forward");
  m.def("backward", &indices_dot_product_backward, "Indices dot product backward");
}
