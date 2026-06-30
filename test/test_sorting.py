# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose

keepdim = [True, False]

dim = [0, 1]

FLOAT_RTOL, FLOAT_ATOL = 1e-5, 5e-5


class Testkthvalue:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize("keepdim", keepdim)
    @pytest.mark.parametrize("dim", dim)
    def test_kthvalue_br200(self, keepdim, dim):
        def test_kthvalue_function(keepdim, dim):
            cpu_x = torch.arange(1.0, 7.0).resize_(2, 3)
            cpu_out = torch.kthvalue(cpu_x, 2, dim, keepdim)

            supa_x = torch.arange(1.0, 7.0).resize_(2, 3).supa()
            supa_out = torch.kthvalue(supa_x, 2, dim, keepdim)

            assert_allclose(
                cpu_out.values, supa_out.values, atol=FLOAT_ATOL, rtol=FLOAT_RTOL
            )
            assert_allclose(
                cpu_out.indices, supa_out.indices, atol=FLOAT_ATOL, rtol=FLOAT_RTOL
            )

        test_kthvalue_function(keepdim, dim)


class TestMedian:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize("keepdim", keepdim)
    @pytest.mark.parametrize("dim", dim)
    def test_median_br200(self, keepdim, dim):
        def test_median_function(keepdim, dim):
            cpu_x = torch.arange(1.0, 7.0).resize_(2, 3)
            cpu_out = torch.median(cpu_x, dim, keepdim)

            supa_x = torch.arange(1.0, 7.0).resize_(2, 3).supa()
            supa_out = torch.median(supa_x, dim, keepdim)

            assert_allclose(
                cpu_out.values, supa_out.values, atol=FLOAT_ATOL, rtol=FLOAT_RTOL
            )
            assert_allclose(
                cpu_out.indices, supa_out.indices, atol=FLOAT_ATOL, rtol=FLOAT_RTOL
            )

        test_median_function(keepdim, dim)
