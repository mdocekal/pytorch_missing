# -*- coding: UTF-8 -*-
"""
Created on 06.06.24

:author:     Martin Dočekal
"""
import time
from unittest import TestCase

import torch

from pytorch_missing.ranges import mrange


class TestMrange(TestCase):
    def test_mrange_empty(self):
        self.assertEqual(mrange(torch.tensor([], dtype=torch.int64)).shape, torch.Size([0]))

    def test_mrange(self):
        ranges = torch.tensor([[0, 3], [2, 3], [1, 3]])
        res = mrange(ranges)
        self.assertTrue(torch.allclose(res, torch.tensor([0, 1, 2, 2, 1, 2], dtype=torch.int64)))

    def test_mrange_cuda(self):
        ranges = torch.tensor([[0, 3], [2, 3], [1, 3]], device="cuda")
        res = mrange(ranges)
        self.assertTrue(torch.allclose(res, torch.tensor([0, 1, 2, 2, 1, 2], dtype=torch.int64, device="cuda")))

    def test_mrange_long(self):
        ranges = torch.tensor(
            [[i, min(i + 3, 100_000)] for i in range(0, 100_000, 3)],
        )

        res = mrange(ranges)
        self.assertTrue(torch.allclose(res, torch.arange(100_000)))

    def test_mrange_cuda_long(self):
        ranges = torch.tensor(
            [[i, min(i + 3, 100_000)] for i in range(0, 100_000, 3)],
            device="cuda"
        )
        res = mrange(ranges)
        self.assertTrue(torch.allclose(res, torch.arange(100_000, device="cuda")))


