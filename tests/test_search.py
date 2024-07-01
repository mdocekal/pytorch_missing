# -*- coding: UTF-8 -*-
"""
Created on 21.06.24

:author:     Martin Dočekal
"""
from unittest import TestCase

import torch

from pytorch_missing.search import intervals_bin_search


class TestIntervalsBinSearch(TestCase):

    def setUp(self):
        self.intervals = torch.tensor([
            [1, 5, 10],  # interval start
            [3, 8, 12]   # interval end (exclusive)
        ])

    def test_intervals_bin_search_single_value_not_in(self):
        vals = torch.tensor([0])
        res = intervals_bin_search(vals, self.intervals)
        self.assertTrue(torch.equal(res, torch.tensor([-1])))
        vals = torch.tensor([8])
        res = intervals_bin_search(vals, self.intervals)
        self.assertTrue(torch.equal(res, torch.tensor([-1])))

    def test_intervals_bin_search_single_value_in(self):
        vals = torch.tensor([2])
        res = intervals_bin_search(vals, self.intervals)
        self.assertTrue(torch.equal(res, torch.tensor([0])))
        vals = torch.tensor([5])
        res = intervals_bin_search(vals, self.intervals)
        self.assertTrue(torch.equal(res, torch.tensor([1])))

    def test_intervals_bin_search_multiple_values(self):
        vals = torch.tensor([2, 5, 8, 10, 11, 12, 99])
        res = intervals_bin_search(vals, self.intervals)
        self.assertTrue(torch.equal(res, torch.tensor([0, 1, -1, 2, 2, -1, -1])))

    def test_intervals_bin_search_single_value_not_in_cuda(self):
        vals = torch.tensor([0], device="cuda")
        res = intervals_bin_search(vals, self.intervals.cuda())
        self.assertTrue(torch.equal(res, torch.tensor([-1], device="cuda")))
        vals = torch.tensor([8], device="cuda")
        res = intervals_bin_search(vals, self.intervals.cuda())
        self.assertTrue(torch.equal(res, torch.tensor([-1], device="cuda")))

    def test_intervals_bin_search_single_value_in_cuda(self):
        vals = torch.tensor([2], device="cuda")
        res = intervals_bin_search(vals, self.intervals.cuda())
        self.assertTrue(torch.equal(res, torch.tensor([0], device="cuda")))
        vals = torch.tensor([5], device="cuda")
        res = intervals_bin_search(vals, self.intervals.cuda())
        self.assertTrue(torch.equal(res, torch.tensor([1], device="cuda")))

    def test_intervals_bin_search_multiple_values_cuda(self):
        vals = torch.tensor([2, 5, 8, 10, 11, 12, 99], device="cuda")
        res = intervals_bin_search(vals, self.intervals.cuda())
        self.assertTrue(torch.equal(res, torch.tensor([0, 1, -1, 2, 2, -1, -1], device="cuda")))

