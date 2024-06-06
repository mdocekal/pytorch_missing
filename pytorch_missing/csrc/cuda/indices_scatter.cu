#include <torch/extension.h>
#include <ATen/Parallel.h>
#include "utils.cuh"

template <typename scalar_t>
__global__ void indices_weighted_scatter_sum_cuda_kernel(const scalar_t* val, const int64_t* reduce_group, const int64_t* reduce_from, const scalar_t* weights, scalar_t* res, int64_t cols, int64_t pairs) {
    int64_t pair_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (pair_i >= pairs) {
        return;
    }

    for (int64_t i = 0; i < cols; i++) {
        atomicAdd(&res[reduce_group[pair_i] * cols + i], val[reduce_from[pair_i] * cols + i] * weights[pair_i]);
    }
}

/**
 * Reduces all values from val matrix to the target matrix using the indices tensor.
 * Example:
 *      val = [
 *          [1, 2],
 *          [3, 4],
 *          [5, 6]
 *      ]
 *      indices = [
            [0, 0, 1]
            [2, 1, 2]
        ]
 *      weights = [0.5, 0.5, 1]
 *
 *      -> [ [5, 6]*0.5 + [3, 4]*0.5, [5, 6]*1 ] = [ [4, 5], [5, 6] ]
 * @param val: values matrix
 * @param indices: indices matrix
 * @param weights: weights vector for each index pair
 * @return The reduced matrix of shape (max(indices[:, 1]) + 1, val.shape[1])
 */
torch::Tensor indices_weighted_scatter_sum(torch::Tensor val, torch::Tensor indices, torch::Tensor weights) {
    auto res = torch::zeros({indices.index({0}).max().item<int>() + 1, val.size(1)}, val.options());
    auto reduce_group = indices.index({0});
    auto reduce_from = indices.index({1});

    AT_DISPATCH_FLOATING_TYPES(val.scalar_type(), "indices_weighted_scatter_sum_cuda", ([&] {
        indices_weighted_scatter_sum_cuda_kernel<scalar_t><<<BLOCKS(indices.size(1)), THREADS>>>(
            val.contiguous().data_ptr<scalar_t>(), reduce_group.contiguous().data_ptr<int64_t>(), reduce_from.contiguous().data_ptr<int64_t>(), weights.contiguous().data_ptr<scalar_t>(), res.data_ptr<scalar_t>(), val.size(1), indices.size(1)
        );
    }));

    return res;
}

template <typename scalar_t>
__global__ void indices_weighted_scatter_sum_backward_cuda_kernel(const scalar_t* grad, const scalar_t* val, const int64_t* reduce_group, const int64_t* reduce_from, const scalar_t* weights, scalar_t* grad_val, scalar_t* grad_weights, int64_t cols, int64_t pairs) {
    int64_t pair_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (pair_i >= pairs) {
        return;
    }

    for (int64_t i = 0; i < cols; i++) {
        atomicAdd(&grad_val[reduce_from[pair_i] * cols + i], grad[reduce_group[pair_i]*cols + i] * weights[pair_i]);
    }

    grad_weights[pair_i] = 0;

    for (int64_t i = 0; i < cols; i++) {
        grad_weights[pair_i] += grad[reduce_group[pair_i]*cols + i] * val[reduce_from[pair_i]*cols + i];
    }
}


std::tuple<torch::Tensor, torch::Tensor> indices_weighted_scatter_sum_backward(torch::Tensor grad, torch::Tensor val, torch::Tensor indices, torch::Tensor weights) {
    auto grad_val = torch::zeros_like(val);
    auto grad_weights = torch::zeros_like(weights);

    auto reduce_group = indices.index({0});
    auto reduce_from = indices.index({1});

    AT_DISPATCH_FLOATING_TYPES(val.scalar_type(), "indices_weighted_scatter_sum_backward_cuda", ([&] {
    indices_weighted_scatter_sum_backward_cuda_kernel<scalar_t><<<BLOCKS(indices.size(1)), THREADS>>>(
        grad.contiguous().data_ptr<scalar_t>(), val.contiguous().data_ptr<scalar_t>(), reduce_group.contiguous().data_ptr<int64_t>(), reduce_from.contiguous().data_ptr<int64_t>(), weights.contiguous().data_ptr<scalar_t>(), grad_val.data_ptr<scalar_t>(), grad_weights.data_ptr<scalar_t>(), val.size(1), indices.size(1)
    );
    }));

    return std::make_tuple(grad_val, grad_weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indices_weighted_scatter_sum_forward_cuda", &indices_weighted_scatter_sum, "Indices scatter forward (cuda)");
  m.def("indices_weighted_scatter_sum_backward_cuda", &indices_weighted_scatter_sum_backward, "Indices scatter backward (cuda)");
}
