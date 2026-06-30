# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor


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
dtypes = [torch.float32, torch.bfloat16, torch.float16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


class TestLogAddExpMethod:

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_logaddexp(self, shape, dtype):
        in_cpu, in_supa = create_random_tensor(shape, dtype=dtype, requires_grad=False)
        in_cpu2, in_supa2 = create_random_tensor(
            shape, dtype=dtype, requires_grad=False
        )

        output_cpu = torch.logaddexp(in_cpu, in_cpu2)
        output_supa = torch.logaddexp(in_supa, in_supa2)

        assert_allclose(
            output_cpu, output_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_logaddexp2(self, shape, dtype):
        in_cpu, in_supa = create_random_tensor(shape, dtype=dtype, requires_grad=False)
        in_cpu2, in_supa2 = create_random_tensor(
            shape, dtype=dtype, requires_grad=False
        )

        output_cpu = torch.logaddexp2(in_cpu, in_cpu2)
        output_supa = torch.logaddexp2(in_supa, in_supa2)

        assert_allclose(
            output_cpu, output_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
