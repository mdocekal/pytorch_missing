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

    def test_dot_product_cuda_large_random(self):
        x = torch.randn(100, 100, device="cuda", requires_grad=True)
        y = torch.randn(100, 100, device="cuda", requires_grad=True)

        x_ein = x.clone().detach().requires_grad_()
        y_ein = y.clone().detach().requires_grad_()
        indices = torch.randint(0, 100, (2, 1000), device="cuda")

        res = indices_dot_product(x, y, indices)
        self.assertTrue(res.shape == (1000,))
        ein = torch.einsum("ij,ij->i", x_ein[indices[0]], y_ein[indices[1]])
        self.assertTrue(torch.allclose(res, ein, atol=1e-4))

        loss = res.sum()
        loss_ein = ein.sum()

        loss.backward()
        loss_ein.backward()

        self.assertTrue(torch.allclose(x_ein.grad, x.grad, atol=1e-4))
        self.assertTrue(torch.allclose(y_ein.grad, y.grad, atol=1e-4))


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

    def test_weighted_scatter_sum_cuda_large_random(self):
        val = torch.randn(100, 100, device="cuda", requires_grad=True)
        indices = torch.randint(0, 100, (2, 1000), device="cuda")
        indices_aggregate_num = indices[0, :].max()+1
        weights = torch.randn(1000, device="cuda", requires_grad=True)

        val_check = val.clone().detach().requires_grad_()
        weights_check = weights.clone().detach().requires_grad_()

        res = indices_weighted_scatter_sum(val, indices, weights)
        self.assertTrue(res.shape == (indices_aggregate_num, 100))

        weighted_vals = val_check[indices[1]] * weights_check.unsqueeze(1)
        gt_res = torch.zeros((indices_aggregate_num, val.shape[1]), device="cuda")
        for i in range(1000):
            gt_res[indices[0, i]] += weighted_vals[i]

        self.assertTrue(torch.allclose(res, gt_res, atol=1e-4))

        loss = res.sum()
        loss_ein = gt_res.sum()

        loss.backward()
        loss_ein.backward()

        self.assertTrue(torch.allclose(val_check.grad, val.grad, atol=1e-4))
        self.assertTrue(torch.allclose(weights_check.grad, weights.grad, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
