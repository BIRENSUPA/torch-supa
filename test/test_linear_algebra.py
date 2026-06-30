# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

br200_shape = [
    pytest.param(
        (16, 16),
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


class TestLinearAlgebraMethod:
    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_addr(self, shape, dtype):
        vec1_cpu, vec1_supa = create_random_tensor(
            [shape[0]], dtype=dtype, requires_grad=False
        )
        vec2_cpu, vec2_supa = create_random_tensor(
            [shape[1]], dtype=dtype, requires_grad=False
        )
        M_cpu, M_supa = create_random_tensor(shape, dtype=dtype, requires_grad=False)

        out_cpu = torch.addr(M_cpu, vec1_cpu, vec2_cpu)
        out_supa = torch.addr(M_supa, vec1_supa, vec2_supa)

        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
