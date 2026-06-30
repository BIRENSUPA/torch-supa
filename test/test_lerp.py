# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

br200_shape = [
    pytest.param(
        (16,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((1024, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]
dtypes = [torch.float32]


class TestLerpMethod:

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_lerp(self, shape, dtype):
        start_cpu, start_supa = create_random_tensor(
            shape, dtype=dtype, requires_grad=False
        )
        end_cpu, end_supa = create_random_tensor(
            shape, dtype=dtype, requires_grad=False
        )
        out_cpu = torch.lerp(start_cpu, end_cpu, 0.5)
        out_supa = torch.lerp(start_supa, end_supa, 0.5)
        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
