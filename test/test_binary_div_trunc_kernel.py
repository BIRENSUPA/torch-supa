# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

supa_device = torch.device("supa")

torch.manual_seed(0)

br200_params = [
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

RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}
dtypes = [torch.float32, torch.bfloat16, torch.float16]


class TestNNMethod:

    @pytest.mark.parametrize("shape", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_div(self, shape, dtype):
        cpu_input1, supa_input1 = create_random_tensor(shape, dtype=dtype)
        cpu_input2, supa_input2 = create_random_tensor(shape, dtype=dtype)
        output_cpu = cpu_input1 / cpu_input2
        output_supa = supa_input1 / supa_input2

        assert_allclose(
            output_cpu, output_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )

    @pytest.mark.parametrize("shape", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_div_rounding_mode(self, shape, dtype):
        cpu_input1, supa_input1 = create_random_tensor(shape, dtype=dtype)
        cpu_input2, supa_input2 = create_random_tensor(shape, dtype=dtype)
        output_cpu = torch.div(cpu_input1, cpu_input2, rounding_mode="trunc")
        output_supa = torch.div(supa_input1, supa_input2, rounding_mode="trunc")

        assert_allclose(
            output_cpu, output_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )

    @pytest.mark.parametrize("shape", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_div_rounding_mode_backward(self, shape, dtype):
        cpu_input1, supa_input1 = create_random_tensor(
            shape, dtype=dtype, requires_grad=True
        )
        cpu_input2, supa_input2 = create_random_tensor(
            shape, dtype=dtype, requires_grad=True
        )
        output_cpu = torch.div(cpu_input1, cpu_input2, rounding_mode="trunc")
        output_supa = torch.div(supa_input1, supa_input2, rounding_mode="trunc")
        cpu_grad, supa_grad = create_random_tensor(
            output_cpu.shape, dtype=dtype, requires_grad=False
        )
        output_cpu.backward(cpu_grad)
        output_supa.backward(supa_grad)

        assert_allclose(
            cpu_input1.grad,
            supa_input1.grad,
            atol=ATOL[dtype],
            rtol=RTOL[dtype],
            equal_nan=True,
        )
        assert_allclose(
            cpu_input2.grad,
            supa_input2.grad,
            atol=ATOL[dtype],
            rtol=RTOL[dtype],
            equal_nan=True,
        )
