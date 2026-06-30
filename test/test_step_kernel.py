# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import numpy as np
import pytest
import torch
from torch import inf, nan

from torch_supa.testing.common_utils import assert_allclose, assert_equal


class TestNextAfter:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_nextafter(self, device="supa", dtype=torch.float32):
        # Test special cases
        t1 = torch.tensor([0, 0, 10], device=device, dtype=dtype)
        t2 = torch.tensor([inf, -inf, 10], device=device, dtype=dtype)
        actual = torch.nextafter(t1, t2)
        expected = np.nextafter(t1.cpu().numpy(), t2.cpu().numpy())
        assert_equal(actual, torch.Tensor(expected))

        actual = torch.nextafter(t2, t1)
        expected = np.nextafter(t2.cpu().numpy(), t1.cpu().numpy())
        assert_equal(actual, torch.Tensor(expected))

        t1 = torch.tensor([0, nan], device=device, dtype=dtype)
        t2 = torch.tensor([nan, 0], device=device, dtype=dtype)

        a = torch.randn(100, device=device, dtype=dtype)
        b = torch.randn(100, device=device, dtype=dtype)
        actual = torch.nextafter(a, b)
        expected = np.nextafter(a.cpu().numpy(), b.cpu().numpy())
        expected = torch.Tensor(expected)
        assert_allclose(actual.cpu(), expected, atol=0, rtol=0)


class TestHeaviside:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_heaviside(self):
        cpu_x = torch.tensor([-1.5, 0, 2.0])
        supa_x = torch.tensor([-1.5, 0, 2.0]).supa()
        cpu_values = torch.tensor([1.2, -2.0, 3.5])
        supa_values = torch.tensor([1.2, -2.0, 3.5]).supa()

        cpu_out = torch.heaviside(cpu_x, cpu_values)
        supa_out = torch.heaviside(supa_x, supa_values)
        assert_allclose(cpu_out, supa_out, atol=5e-5, rtol=1e-5, equal_nan=True)
