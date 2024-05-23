#include <torch/extension.h>

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

    for (int i = 0; i < indices.size(1); i++) {
        res[reduce_group[i]] += val[reduce_from[i]] * weights[i];
    }
    return res;
}

std::tuple<torch::Tensor, torch::Tensor> indices_weighted_scatter_sum_backward(torch::Tensor grad, torch::Tensor val, torch::Tensor indices, torch::Tensor weights) {
    auto grad_val = torch::zeros_like(val);
    auto grad_weights = torch::zeros_like(weights);

    auto reduce_group = indices.index({0});
    auto reduce_from = indices.index({1});

    for (int i = 0; i < indices.size(1); i++) {
        grad_val[reduce_from[i]] += grad[reduce_group[i]] * weights[i];
        grad_weights[i] = (grad[reduce_group[i]] * val[reduce_from[i]]).sum().item<float>();
    }
    return std::make_tuple(grad_val, grad_weights);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indices_weighted_scatter_sum_forward", &indices_weighted_scatter_sum, "Indices scatter forward");
  m.def("indices_weighted_scatter_sum_backward", &indices_weighted_scatter_sum_backward, "Indices scatter backward");
}
