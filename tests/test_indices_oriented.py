# -*- coding: UTF-8 -*-
""""
Created on 03.05.24

:author:     Martin Dočekal
"""

import unittest

import torch

from pytorch_missing.indices_oriented import indices_dot_product, indices_weighted_scatter_sum


class TestIndicesDotProduct(unittest.TestCase):
    def test_dot_product(self):
        x = torch.tensor([[0.5475, 0.1543]], requires_grad=True)
        y = torch.tensor([[0.7271, 0.7619], [0.1104, 0.2592]], requires_grad=True)
        indices = torch.tensor([[0, 0], [0, 1]])

        res = indices_dot_product(x, y, indices)
        self.assertTrue(torch.allclose(res, torch.tensor([0.5156, 0.1004]), atol=1e-4))

        loss = res.sum()
        loss.backward()

        self.assertTrue(torch.allclose(x.grad, torch.tensor([[0.8375, 1.0211]]), atol=1e-4))
        self.assertTrue(torch.allclose(y.grad, torch.tensor([[0.5475, 0.1543], [0.5475, 0.1543]]), atol=1e-4))

    def test_dot_product_cuda(self):
        x = torch.tensor([[0.5475, 0.1543]], requires_grad=True, device="cuda")
        y = torch.tensor([[0.7271, 0.7619], [0.1104, 0.2592]], requires_grad=True, device="cuda")
        indices = torch.tensor([[0, 0], [0, 1]]).cuda()

        res = indices_dot_product(x, y, indices)
        self.assertTrue(torch.allclose(res, torch.tensor([0.5156, 0.1004]).cuda(), atol=1e-4))

        loss = res.sum()
        loss.backward()

        self.assertTrue(torch.allclose(x.grad, torch.tensor([[0.8375, 1.0211]]).cuda(), atol=1e-4))
        self.assertTrue(torch.allclose(y.grad, torch.tensor([[0.5475, 0.1543], [0.5475, 0.1543]]).cuda(), atol=1e-4))

    def test_dot_product_cuda_bfloat16(self):
        x = torch.tensor([[0.5475, 0.1543]], requires_grad=True, device="cuda", dtype=torch.bfloat16)
        y = torch.tensor([[0.7271, 0.7619], [0.1104, 0.2592]], requires_grad=True, device="cuda", dtype=torch.bfloat16)
        indices = torch.tensor([[0, 0], [0, 1]]).cuda()

        res = indices_dot_product(x, y, indices)

        self.assertTrue(res.dtype == torch.bfloat16)
        self.assertTrue(torch.allclose(res, torch.tensor([0.5156, 0.1004], device="cuda", dtype=torch.bfloat16), atol=1e-4))

        loss = res.sum()
        loss.backward()

        self.assertTrue(torch.allclose(x.grad, torch.tensor([[0.8375, 1.0211]], device="cuda", dtype=torch.bfloat16), atol=1e-4))
        self.assertTrue(torch.allclose(y.grad, torch.tensor([[0.5475, 0.1543], [0.5475, 0.1543]], device="cuda", dtype=torch.bfloat16), atol=1e-4))


class TestIndicesWeightedScatterSum(unittest.TestCase):
    def test_weighted_scatter_sum(self):
        val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
        indices = torch.tensor([[0, 0, 1], [2, 1, 2]])
        weights = torch.tensor([0.5, 0.5, 1.0], requires_grad=True)

        res = indices_weighted_scatter_sum(val, indices, weights)
        self.assertTrue(torch.allclose(res, torch.tensor([[4.0, 5.0], [5.0, 6.0]]), atol=1e-4))

        loss = res.sum()
        loss.backward()

        self.assertTrue(torch.allclose(val.grad, torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.5, 1.5]]), atol=1e-4))
        self.assertTrue(torch.allclose(weights.grad, torch.tensor([11.0, 7.0, 11.0]), atol=1e-4))

    def test_weighted_scatter_sum_cuda(self):
        val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True, device="cuda")
        indices = torch.tensor([[0, 0, 1], [2, 1, 2]]).cuda()
        weights = torch.tensor([0.5, 0.5, 1.0], requires_grad=True, device="cuda")

        res = indices_weighted_scatter_sum(val, indices, weights)
        self.assertTrue(torch.allclose(res, torch.tensor([[4.0, 5.0], [5.0, 6.0]]).cuda(), atol=1e-4))

        loss = res.sum()
        loss.backward()

        self.assertTrue(torch.allclose(val.grad, torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.5, 1.5]]).cuda(), atol=1e-4))
        self.assertTrue(torch.allclose(weights.grad, torch.tensor([11.0, 7.0, 11.0]).cuda(), atol=1e-4))

    def test_weighted_scatter_sum_cuda_bfloat16(self):
        val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True, device="cuda", dtype=torch.bfloat16)
        indices = torch.tensor([[0, 0, 1], [2, 1, 2]]).cuda()
        weights = torch.tensor([0.5, 0.5, 1.0], requires_grad=True, device="cuda", dtype=torch.bfloat16)

        res = indices_weighted_scatter_sum(val, indices, weights)
        self.assertTrue(res.dtype == torch.bfloat16)
        self.assertTrue(torch.allclose(res, torch.tensor([[4.0, 5.0], [5.0, 6.0]], device="cuda", dtype=torch.bfloat16), atol=1e-4))

        loss = res.sum()
        loss.backward()

        self.assertTrue(torch.allclose(val.grad, torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.5, 1.5]], device="cuda", dtype=torch.bfloat16), atol=1e-4))
        self.assertTrue(torch.allclose(weights.grad, torch.tensor([11.0, 7.0, 11.0], device="cuda", dtype=torch.bfloat16), atol=1e-4))


if __name__ == '__main__':
    unittest.main()
