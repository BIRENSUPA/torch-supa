# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose

FLOAT_RTOL, FLOAT_ATOL = 1e-5, 5e-5


class TestHistc:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_histc_br200(self):
        def test_histc_function():
            cpu_x = torch.tensor([1.0, 2, 1])
            cpu_out = torch.histc(cpu_x, bins=4, min=0, max=3)

            supa_x = torch.tensor([1.0, 2, 1]).supa()
            supa_out = torch.histc(supa_x, bins=4, min=0, max=3)

            assert_allclose(cpu_out, supa_out, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)

        test_histc_function()


class TestBinCount:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_bincount_br200(self):
        def test_bincount_function():
            cpu_x = torch.randint(0, 8, (1,), dtype=torch.int64)
            cpu_out = torch.bincount(cpu_x)

            supa_x = cpu_x.clone().supa()
            supa_out = torch.bincount(supa_x)

            assert_allclose(cpu_out, supa_out, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)

        test_bincount_function()
