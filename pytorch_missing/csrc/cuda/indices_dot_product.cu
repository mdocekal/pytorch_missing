#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/Atomic.cuh>
#include "utils.cuh"


template <typename scalar_t>
__global__ void indices_dot_product_cuda_kernel(const scalar_t* x, const scalar_t* y, const int64_t* from, const int64_t* to, scalar_t* res, int64_t cols, int64_t pairs) {
    int64_t pair_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (pair_i >= pairs) {
        return;
    }

    scalar_t sum = 0;
    auto from_offset = from[pair_i] * cols;
    auto to_offset = to[pair_i] * cols;

    for (int64_t i = 0; i < cols; i++) {
        sum += x[from_offset + i] * y[to_offset + i];
    }
    res[pair_i] = sum;
}

/**
 * Computes dot product for each indices pair.
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

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "indices_dot_product_cuda", ([&] {
        indices_dot_product_cuda_kernel<scalar_t><<<BLOCKS(indices.size(1)), THREADS>>>(
            x.contiguous().data_ptr<scalar_t>(), y.contiguous().data_ptr<scalar_t>(), from.contiguous().data_ptr<int64_t>(), to.contiguous().data_ptr<int64_t>(), res.data_ptr<scalar_t>(), x.size(1), indices.size(1)
        );
    }));

    return res;
}


template <typename scalar_t>
__global__ void indices_dot_product_backward_cuda_kernel(const scalar_t* grad, const scalar_t* x, const scalar_t* y, const int64_t* from, const int64_t* to, scalar_t* grad_x, scalar_t* grad_y, int64_t cols, int64_t pairs) {
    int64_t pair_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (pair_i >= pairs) {
        return;
    }

    auto grad_for_pair = grad[pair_i];
    auto from_offset = from[pair_i] * cols;
    auto to_offset = to[pair_i] * cols;

    for (int64_t i = 0; i < cols; i++) {
        gpuAtomicAdd(&grad_x[from_offset + i], grad_for_pair * y[to_offset + i]);
        gpuAtomicAdd(&grad_y[to_offset + i], grad_for_pair * x[from_offset + i]);
    }
}


std::tuple<torch::Tensor, torch::Tensor> indices_dot_product_backward(torch::Tensor grad, torch::Tensor x, torch::Tensor y, torch::Tensor indices) {
    auto grad_x = torch::zeros_like(x);
    auto grad_y = torch::zeros_like(y);

    auto from = indices.index({0});
    auto to = indices.index({1});

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "indices_dot_product_backward_cuda", ([&] {
        indices_dot_product_backward_cuda_kernel<scalar_t><<<BLOCKS(indices.size(1)), THREADS>>>(
            grad.contiguous().data_ptr<scalar_t>(), x.contiguous().data_ptr<scalar_t>(), y.contiguous().data_ptr<scalar_t>(), from.contiguous().data_ptr<int64_t>(), to.contiguous().data_ptr<int64_t>(), grad_x.data_ptr<scalar_t>(), grad_y.data_ptr<scalar_t>(), x.size(1), indices.size(1)
        );
    }));

    return std::make_tuple(grad_x, grad_y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indices_dot_product_forward_cuda", &indices_dot_product, "Indices dot product forward (cuda)");
  m.def("indices_dot_product_backward_cuda", &indices_dot_product_backward, "Indices dot product backward (cuda)");
}

