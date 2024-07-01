#include <torch/extension.h>
#include <ATen/Parallel.h>

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

    std::mutex mutex;

    at::parallel_for(0, indices.size(1), 0, [&](int64_t begin, int64_t end) {
        auto res_local = torch::zeros_like(res);
        for (auto i = begin; i < end; i++) {
            res_local[reduce_group[i]] += val[reduce_from[i]] * weights[i];
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            res += res_local;
        }

    });
    return res;
}

std::tuple<torch::Tensor, torch::Tensor> indices_weighted_scatter_sum_backward(torch::Tensor grad, torch::Tensor val, torch::Tensor indices, torch::Tensor weights) {
    auto grad_val = torch::zeros_like(val);
    auto grad_weights = torch::zeros_like(weights);

    auto reduce_group = indices.index({0});
    auto reduce_from = indices.index({1});

    std::mutex mutex;

    at::parallel_for(0, indices.size(1), 0, [&](int64_t begin, int64_t end) {
        auto grad_val_local = torch::zeros_like(val);

        for (auto i = begin; i < end; i++) {
            grad_val_local[reduce_from[i]] += grad[reduce_group[i]] * weights[i];
            grad_weights[i] = (grad[reduce_group[i]] * val[reduce_from[i]]).sum().item<float>();
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            grad_val += grad_val_local;
        }

    });
    return std::make_tuple(grad_val, grad_weights);
}

/**
 * Reduces all values from val matrix to the target matrix using the intervals tensor.
 * Example:
 *      val = [
 *          [1, 2],
 *          [3, 4],
 *          [5, 6]
 *      ]
 *      intervals = [
            [0, 1]  # aggregation groups indices
            [1, 2]  # start of interval
            [3, 3]  # end of interval (exclusive)
        ]
 *      weights = [0.5, 0.5, 1]
 *
 *      -> [ ([3, 4]+[5, 6])*0.5, [5, 6]*1 ] = [ [4, 5], [5, 6] ]
 * @param val: values matrix
 * @param intervals: intervals tensor of shape (3, num_intervals)
 *  It is expected that all intervals have empty intersection and are sorted by the start in ascending order.
 * @param weights: weights vector for each interval
 * @return The reduced matrix of shape (number of groups, val.shape[1])
 */
torch::Tensor intervals_weighted_scatter_sum(torch::Tensor val, torch::Tensor intervals, torch::Tensor weights) {
    auto res = torch::zeros({intervals.index({0}).max().item<int>() + 1, val.size(1)}, val.options());

    auto reduce_group = intervals.index({0});
    auto reduce_interval_start = intervals.index({1});
    auto reduce_interval_end = intervals.index({2});

    auto reduce_interval_size = reduce_interval_end - reduce_interval_start;
    auto total_size = reduce_interval_size.sum();

    std::mutex mutex;

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indices_weighted_scatter_sum_forward", &indices_weighted_scatter_sum, "Indices scatter forward");
  m.def("indices_weighted_scatter_sum_backward", &indices_weighted_scatter_sum_backward, "Indices scatter backward");
}
