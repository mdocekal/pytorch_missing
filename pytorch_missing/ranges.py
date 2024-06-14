# -*- coding: UTF-8 -*-
"""
Created on 06.06.24
Module with utils for working ranges.

:author:     Martin Dočekal
"""
import torch

from pytorch_missing.ranges_cpu import mrange as mrange_cpu
from pytorch_missing.ranges_cuda import mrange_cuda


def mrange(ranges: torch.Tensor) -> torch.Tensor:
    """
    Creates flattened tensor filled with multiple integer ranges.
    Example:
        ranges =  [[0,3], [2,3], [1,3]]
        -> [0, 1, 2, 2, 1, 2]
    :param ranges: 2D tensor of ranges
    :return: Flattened tensor with ranges
    """

    if ranges.is_cuda:
        return mrange_cuda(ranges)
    return mrange_cpu(ranges)
