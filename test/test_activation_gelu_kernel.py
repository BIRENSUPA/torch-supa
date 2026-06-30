# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

br200_shapes = [
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


class TestGelu:

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_gelu_forward(self, shape, dtype):
        cpu_input, supa_input = create_random_tensor(shape, dtype=dtype)

        cpu_input.requires_grad_(False)
        supa_input.requires_grad_(False)

        m = torch.nn.GELU()
        y_cpu = m(cpu_input)
        y_supa = m(supa_input)

        assert_allclose(y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_gelu_backward(self, shape, dtype):

        cpu_input, supa_input = create_random_tensor(
            shape, dtype=dtype, requires_grad=True
        )
        cpu_grad, supa_grad = create_random_tensor(shape, dtype=dtype)

        m = torch.nn.GELU()

        y_cpu = m(cpu_input)
        y_cpu.backward(cpu_grad)
        cpu_res = cpu_input.grad.data

        y_supa = m(supa_input)
        y_supa.backward(supa_grad)
        supa_res = supa_input.grad.data

        assert_allclose(y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])
        assert_allclose(cpu_res, supa_res, atol=ATOL[dtype], rtol=RTOL[dtype])
