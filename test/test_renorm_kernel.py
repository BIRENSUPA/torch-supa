# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

dtype = [torch.float32]

params = [
    pytest.param(
        (1, 16),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]

dims = [0, 1]
power = [1, 2]

dtypes = [torch.float32, torch.bfloat16, torch.float16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 5e-4, torch.bfloat16: 5e-4, torch.float16: 5e-4}


class TestRenorm:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize("dtype", dtype)
    def test_renorm_forward(self, dtype):
        x_cpu = torch.ones(3, 3, dtype=dtype)
        x_cuda = torch.ones(3, 3, dtype=dtype).supa()

        x_cpu[1].fill_(2)
        x_cuda[1].fill_(2)
        x_cpu[2].fill_(3)
        x_cuda[2].fill_(3)

        out_cpu = torch.renorm(x_cpu, 1, 0, 5)
        out_cuda = torch.renorm(x_cuda, 1, 0, 5)

        assert_allclose(out_cpu, out_cuda, rtol=1e-5, atol=5e-5)

    @pytest.mark.parametrize("input_shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("dim", dims)
    @pytest.mark.parametrize("p", power)
    def test_renorm(self, input_shape, dtype, dim, p):
        cpu_input, supa_input = create_random_tensor(
            input_shape, dtype=dtype, requires_grad=True
        )
        y_cpu = torch.renorm(cpu_input, p, dim, maxnorm=5)
        y_supa = torch.renorm(supa_input, p, dim, maxnorm=5)

        assert_allclose(y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])

        cpu_grad, supa_grad = create_random_tensor(y_cpu.shape, dtype=dtype)
        y_cpu.backward(cpu_grad)
        y_supa.backward(supa_grad)

        assert_allclose(
            cpu_input.grad, supa_input.grad, atol=ATOL[dtype], rtol=RTOL[dtype]
        )
