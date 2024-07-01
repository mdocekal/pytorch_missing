#include <torch/extension.h>
#include "utils.cuh"


/**
 * Binary search for intervals.
 * @param val: value to search
 * @param intervals: intervals array
 * @param numb_of_intervals: total number of intervals
 * @return The index of the interval that contains the value or -1 if not found.
 */
template <typename scalar_t>
__device__ __inline__ int64_t intervals_bin_search(const scalar_t val, const scalar_t* intervals, int64_t numb_of_intervals) {
    int64_t start = 0;
    int64_t end = numb_of_intervals - 1;
    while (start < end) {
        int64_t mid = start + (end - start) / 2;
        if(val >= intervals[mid + numb_of_intervals]) {
            start = mid + 1;
        } else {
            end = mid;
        }
    }

    if (intervals[start] <= val && val < intervals[start + numb_of_intervals]) {
        return start;
    }
    return -1;
}

/**
 * Binary search for intervals.
 * @param val: value to search
 * @param intervals: intervals array
 * @param res: result array each element contains the index of the interval that contains the value or -1 if not found.
 * @param size: number of intervals
 * @param numb_of_intervals: total number of intervals
 */
template <typename scalar_t>
__global__ void intervals_bin_search_cuda_kernel(const scalar_t* val, const scalar_t* intervals, int64_t* res, int64_t size, int64_t numb_of_intervals) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    res[idx] = intervals_bin_search(val[idx], intervals, numb_of_intervals);
}

/**
 * Binary search for intervals.
 * @param val: value to search
 * @param intervals: intervals array
 * @return The index of the interval that contains the value or -1 if not found.
 */
torch::Tensor intervals_bin_search_cuda(torch::Tensor val, torch::Tensor intervals) {
    auto res = torch::empty_like(val, val.options().dtype(torch::kInt64));
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, val.scalar_type(), "intervals_bin_search_cuda", ([&] {
        intervals_bin_search_cuda_kernel<scalar_t><<<BLOCKS(val.size(0)), THREADS>>>(
            val.contiguous().data_ptr<scalar_t>(),
            intervals.contiguous().data_ptr<scalar_t>(),
            res.contiguous().data_ptr<int64_t>(),
            val.size(0),
            intervals.size(1)
        );
    }));

    return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("intervals_bin_search_cuda", &intervals_bin_search_cuda, "Binary search for intervals (cuda)");
}