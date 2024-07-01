# -*- coding: UTF-8 -*-
"""
Created on 21.06.24

:author:     Martin Dočekal
"""
import torch

from pytorch_missing.search_cpu import intervals_bin_search as intervals_bin_search_cpu
from pytorch_missing.search_cuda import intervals_bin_search_cuda


def intervals_bin_search(vals: torch.Tensor, intervals: torch.Tensor) -> torch.Tensor:
    """
    Binary search for intervals

    Example:
        val = [0, 4, 6, 9]
        intervals = [
         [1, 5],  # interval start
         [3, 8]   # interval end (exclusive)
         ]
         -> [-1, -1, 1, -1]
    :param vals: values to search
    :param intervals: intervals to search in
    :return: index of the interval containing the value on position i or -1 if not found
    """

    if vals.is_cuda:
        return intervals_bin_search_cuda(vals, intervals)
    return intervals_bin_search_cpu(vals, intervals)
