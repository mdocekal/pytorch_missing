#include <torch/extension.h>
#include <ATen/Parallel.h>


torch::Tensor indices_weighted_scatter_sum(torch::Tensor val, torch::Tensor indices, torch::Tensor weights) {
    auto res = torch::zeros({indices.index({0}).max().item<int>() + 1, val.size(1)}, val.options());
    auto reduce_group = indices.index({0});
    auto reduce_from = indices.index({1});

    at::parallel_for(0, indices.size(1), 0, [&](int64_t begin, int64_t end) {
        for (auto i = begin; i < end; i++) {
            res[reduce_group[i]] += val[reduce_from[i]] * weights[i];
        }
    });
    return res;
}

std::tuple<torch::Tensor, torch::Tensor> indices_weighted_scatter_sum_backward(torch::Tensor grad, torch::Tensor val, torch::Tensor indices, torch::Tensor weights) {
    auto grad_val = torch::zeros_like(val);
    auto grad_weights = torch::zeros_like(weights);

    auto reduce_group = indices.index({0});
    auto reduce_from = indices.index({1});

    at::parallel_for(0, indices.size(1), 0, [&](int64_t begin, int64_t end) {
        for (auto i = begin; i < end; i++) {
            grad_val[reduce_from[i]] += grad[reduce_group[i]] * weights[i];
            grad_weights[i] = (grad[reduce_group[i]] * val[reduce_from[i]]).sum().item<float>();
        }
    });
    return std::make_tuple(grad_val, grad_weights);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indices_weighted_scatter_sum_forward", &indices_weighted_scatter_sum, "Indices scatter forward");
  m.def("indices_weighted_scatter_sum_backward", &indices_weighted_scatter_sum_backward, "Indices scatter backward");
}
