# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


br200_softplus = [
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

requires_grads = [True]


class TestSoftplus:

    @pytest.mark.parametrize("input_shape", br200_softplus)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("requires_grad", requires_grads)
    def test_softplus(self, input_shape, dtype, requires_grad):
        cpu_input, supa_input = create_random_tensor(
            input_shape, dtype=dtype, requires_grad=requires_grad
        )

        cpu_out = torch.nn.functional.softplus(cpu_input)
        supa_out = torch.nn.functional.softplus(supa_input)

        assert_allclose(cpu_out, supa_out, atol=ATOL[dtype], rtol=RTOL[dtype])

        if requires_grad:
            cpu_grad, supa_grad = create_random_tensor(cpu_out.shape, dtype=dtype)
            cpu_out.backward(cpu_grad)
            cpu_res = cpu_input.grad

            supa_out.backward(supa_grad)
            supa_res = supa_input.grad

            assert_allclose(cpu_res, supa_res, atol=ATOL[dtype], rtol=RTOL[dtype])
