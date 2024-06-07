#include <torch/extension.h>
#include <ATen/Parallel.h>


/**
* Creates flattened tensor filled with multiple integer ranges.
* Example:
*      ranges =  [[0,3], [2,3], [1,3]]
*      -> [0, 1, 2, 2, 1, 2]
* @param ranges: 2D tensor of ranges
* @return Flattened tensor with ranges
*/
torch::Tensor mrange(torch::Tensor ranges) {
    if (ranges.size(0) == 0) {
        return torch::empty({0}, ranges.options());
    }
    auto interval_sizes = ranges.index({torch::indexing::Slice(), 1}) - ranges.index({torch::indexing::Slice(), 0});
    auto range_offsets = interval_sizes.cumsum(0);
    auto total_size = range_offsets.index({-1}).item<long>();
    auto res = torch::empty({total_size}, ranges.options());

    if (total_size == 0) {
        return res;
    }

    range_offsets = torch::cat({torch::tensor({0}, ranges.options()), range_offsets}, 0);

    at::parallel_for(0, ranges.size(0), 0, [&](int64_t begin, int64_t end) {
        for (auto i = begin; i < end; i++) {
            auto start = range_offsets[i].item<long>();
            auto end = range_offsets[i+1].item<long>();
            res.index({torch::indexing::Slice(start, end)}) = torch::arange(ranges.index({i, 0}).item<int>(), ranges.index({i, 1}).item<int>(), ranges.options());
        }
    });
    return res;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mrange", &mrange, "Creates flattened tensor filled with multiple ranges");
}
