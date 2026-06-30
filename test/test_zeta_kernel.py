# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose

FLOAT_RTOL, FLOAT_ATOL = 1e-5, 5e-5


class TestZeta:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_weight_norm(self):
        out_cpu = torch.special.zeta(torch.tensor([2.0, 4.0]), torch.tensor([1.0, 2.0]))
        out_supa = torch.special.zeta(
            torch.tensor([2.0, 4.0]).supa(), torch.tensor([1.0, 2.0]).supa()
        )
        assert_allclose(out_cpu, out_supa, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)
