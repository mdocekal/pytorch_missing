#include <torch/extension.h>
#include <ATen/Parallel.h>
#include "utils.cuh"

template <typename scalar_t>
__global__ void mrange_cuda_kernel(const int64_t* range_offsets, const scalar_t* ranges, int64_t* res, const int64_t num_ranges) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_ranges) {
        return;
    }

    const int64_t start = range_offsets[idx];
    const int64_t end = range_offsets[idx + 1];
    const scalar_t range_start = ranges[idx * 2];
    const scalar_t range_end = ranges[idx * 2 + 1];

    for (int64_t i = start; i < end; i++) {
        res[i] = range_start + (i - start);
    }
}

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

    AT_DISPATCH_INTEGRAL_TYPES(ranges.scalar_type(), "mrange_cuda", ([&] {
        mrange_cuda_kernel<scalar_t><<<BLOCKS(ranges.size(0)), THREADS>>>(
            range_offsets.contiguous().data_ptr<int64_t>(),
            ranges.contiguous().data_ptr<scalar_t>(),
            res.contiguous().data_ptr<int64_t>(),
            ranges.size(0)
        );
    }));

    return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mrange_cuda", &mrange, "Creates flattened tensor filled with multiple ranges (cuda)");
}
