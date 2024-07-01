#include <torch/extension.h>

/**
    * Binary search for intervals
    * Example:
    *      val = 6
    *      intervals = [
    *          [1, 5],  # interval start
    *          [3, 8]   # interval end (exclusive)
    *      ]
    *      -> 1
    *
    * @param val value to search
    * @param intervals to search in
    * @return index of the interval containing the value or -1 if not found
    */
template<typename T>
int64_t intervals_bin_search_single_val(T val, torch::Tensor intervals) {
    auto start = 0;
    auto end = intervals.size(1) - 1;
    auto mid = 0;
    while (start < end) {
        mid = start + (end - start) / 2;
        if (val >= intervals[1][mid].item<T>()) {
            start = mid + 1;
        } else {
            end = mid;
        }
    }

    if(intervals[0][start].item<T>() <= val && val < intervals[1][start].item<T>()) return start;

    return -1;
}

torch::Tensor intervals_bin_search(torch::Tensor vals, torch::Tensor intervals) {
    auto res = torch::empty({vals.size(0)}, vals.options().dtype(torch::kInt64));

    at::parallel_for(0, vals.size(0), 0, [&](int64_t begin, int64_t end) {
        for (auto i = begin; i < end; i++) {
            res[i] = intervals_bin_search_single_val<int64_t>(vals[i].item<int64_t>(), intervals);
        }
    });
    return res;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("intervals_bin_search", &intervals_bin_search, "Binary search for intervals");
}