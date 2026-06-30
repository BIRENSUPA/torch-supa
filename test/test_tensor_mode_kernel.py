# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose

keepdim = [True, False]

dim = [0, 1]


class TestMode:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize("keepdim", keepdim)
    @pytest.mark.parametrize("dim", dim)
    def test_mode_br200(self, keepdim, dim):
        def test_mode_function(keepdim, dim):
            cpu_x = torch.tensor(
                [
                    [0, 0, 0, 2, 0, 0, 2],
                    [0, 3, 0, 0, 2, 0, 1],
                    [2, 2, 1, 0, 0, 0, 3],
                    [2, 2, 3, 0, 1, 1, 0],
                    [1, 1, 0, 0, 2, 0, 2],
                ]
            )
            cpu_out = torch.mode(cpu_x, dim, keepdim)
            supa_x = cpu_x.supa()
            supa_out = torch.mode(supa_x, dim, keepdim)

            assert_allclose(cpu_out.values, supa_out.values, atol=0, rtol=0)
            assert_allclose(cpu_out.indices, supa_out.indices, atol=0, rtol=0)

        test_mode_function(keepdim, dim)
