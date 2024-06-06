# -*- coding: UTF-8 -*-
"""
Created on 03.05.24
Contains indices oriented functions.

:author:     Martin Dočekal
"""

import torch

from pytorch_missing.indices_dot_product import indices_dot_product_forward, indices_dot_product_backward
from pytorch_missing.indices_dot_product_cuda import indices_dot_product_forward_cuda, indices_dot_product_backward_cuda
from pytorch_missing.indices_scatter import indices_weighted_scatter_sum_forward, indices_weighted_scatter_sum_backward
from pytorch_missing.indices_scatter_cuda import indices_weighted_scatter_sum_forward_cuda, indices_weighted_scatter_sum_backward_cuda


class IndicesDotProduct(torch.autograd.Function):
    """
    Computes dot product for each indices pair.

     Example:
        x = [
            [1, 2],
            [3, 4],
            [5, 6]
        ]
        y = [
            [7, 8],
            [9, 10],
            [11, 12]
        ]
        index = [
            [0, 0, 1]
            [2, 1, 2]
        ]

        -> [1*11 + 2*12, 1*7 + 2*8, 3*9 + 4*10] = [35, 23, 67]
    """

    @staticmethod
    def forward(ctx, x, y, index):
        """
        Forward pass.

        :param ctx: context
        :param x: matrix of source vectors
        :param y: matrix of destination vectors
        :param index: indices tensor
        :return: dot product for each index pair
        """

        if x.is_cuda:
            res = indices_dot_product_forward_cuda(x, y, index)
        else:
            res = indices_dot_product_forward(x, y, index)

        ctx.save_for_backward(x, y, index)

        return res

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass.

        :param ctx: context
        :param grad_output: gradient of the output
        :return: gradient of the input
        """

        x, y, index = ctx.saved_tensors
        if x.is_cuda:
            grad_x, grad_y = indices_dot_product_backward_cuda(grad_output, x, y, index)
        else:
            grad_x, grad_y = indices_dot_product_backward(grad_output, x, y, index)

        return grad_x, grad_y, None


def indices_dot_product(x: torch.Tensor, y: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Computes dot product for each indices pair.

    Example:
        x = [
            [1, 2],
            [3, 4],
            [5, 6]
        ]
        y = [
            [7, 8],
            [9, 10],
            [11, 12]
        ]
        index = [
            [0, 0, 1]
            [2, 1, 2]
        ]

        -> [1*11 + 2*12, 1*7 + 2*8, 3*9 + 4*10] = [35, 23, 67]

    :param x: matrix of source vectors
    :param y: matrix of destination vectors
    :param index: indices tensor
    :return: dot product for each index pair
    """

    return IndicesDotProduct.apply(x, y, index)


class IndicesWeightedScatterSum(torch.autograd.Function):
    """
    Reduces all values from val matrix to the target matrix using the indices tensor.

    Example:
        val = [
            [1, 2],
            [3, 4],
            [5, 6]
        ]
        index = [
            [0, 0, 1]
            [2, 1, 2]
        ]
        weights = [0.5, 0.5, 1]

        -> [ [5, 6]*0.5 + [3, 4]*0.5, [5, 6]*1 ] = [ [4, 5], [5, 6] ]
    """

    @staticmethod
    def forward(ctx, val, indices, weights):
        """
        Forward pass.

        :param ctx: context
        :param val: values matrix
        :param indices: indices matrix
        :param weights: weights vector for each index pair

        :return: The reduced matrix of shape (max(indices[:, 1]) + 1, val.shape[1])
        """

        if val.is_cuda:
            res = indices_weighted_scatter_sum_forward_cuda(val, indices, weights)
        else:
            res = indices_weighted_scatter_sum_forward(val, indices, weights)

        ctx.save_for_backward(val, indices, weights)

        return res

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass.

        :param ctx: context
        :param grad_output: gradient of the output
        :return: gradient for val and weights
        """

        val, indices, weights = ctx.saved_tensors
        if val.is_cuda:
            grad_val, grad_weights = indices_weighted_scatter_sum_backward_cuda(grad_output, val, indices, weights)
        else:
            grad_val, grad_weights = indices_weighted_scatter_sum_backward(grad_output, val, indices, weights)

        return grad_val, None, grad_weights


def indices_weighted_scatter_sum(val: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Reduces all values from val matrix to the target matrix using the indices tensor.

    Example:
        val = [
            [1, 2],
            [3, 4],
            [5, 6]
        ]
        index = [
            [0, 0, 1]
            [2, 1, 2]
        ]
        weights = [0.5, 0.5, 1]

        -> [ [5, 6]*0.5 + [3, 4]*0.5, [5, 6]*1 ] = [ [4, 5], [5, 6] ]

    :param val: values matrix
    :param indices: indices matrix
    :param weights: weights vector for each index pair
    :return: The reduced matrix of shape (max(indices[:, 1]) + 1, val.shape[1])
    """

    return IndicesWeightedScatterSum.apply(val, indices, weights)
