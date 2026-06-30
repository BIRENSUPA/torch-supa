# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

dtypes = [torch.float32, torch.bfloat16, torch.float16]


class TestReSize:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize("dtype", dtypes)
    def test_resize(self, dtype):
        x_cpu, y_supa = create_random_tensor([2, 2, 4, 3], dtype=dtype)

        x_cpu.resize_(1, 2, 3, 4)
        y_supa.resize_(1, 2, 3, 4)

        assert_allclose(y_supa.cpu(), x_cpu, rtol=1e-5, atol=1e-5)
