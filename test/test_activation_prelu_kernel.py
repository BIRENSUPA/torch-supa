# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import copy

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


dtypes = [torch.float32, torch.bfloat16, torch.float16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


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

requires_grads = [
    False,
    # True,    (TODO: need to add aten::sum.IntList_out)
]


class TestPrelu:

    @pytest.mark.parametrize("input_shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("requires_grad", requires_grads)
    def test_prelu(self, input_shape, dtype, requires_grad):
        cpu_input, supa_input = create_random_tensor(
            input_shape, dtype=dtype, requires_grad=requires_grad
        )
        num_parameters = 1
        if len(input_shape) >= 2:
            num_parameters = input_shape[1]
        cpu_weight, supa_weight = create_random_tensor(
            (num_parameters,), dtype=dtype, requires_grad=requires_grad
        )
        prelu_cpu = torch.nn.PReLU(num_parameters)
        prelu_supa = copy.deepcopy(prelu_cpu).to(supa_device)
        prelu_cpu.weight.data = cpu_weight
        prelu_supa.weight.data = supa_weight

        cpu_out = prelu_cpu(cpu_input)
        supa_out = prelu_supa(supa_input)

        assert_allclose(cpu_out, supa_out.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])

        if requires_grad:
            cpu_grad, supa_grad = create_random_tensor(
                cpu_out.shape, dtype=dtype, requires_grad=False
            )
            cpu_out.backward(cpu_grad)
            cpu_in_grad = cpu_input.grad
            cpu_weight_grad = prelu_cpu.weight.grad

            supa_out.backward(supa_grad)
            supa_in_grad = supa_input.grad
            supa_weight_grad = prelu_supa.weight.grad

            assert_allclose(
                cpu_in_grad, supa_in_grad, atol=ATOL[dtype], rtol=RTOL[dtype]
            )
            assert_allclose(
                cpu_weight_grad, supa_weight_grad, atol=ATOL[dtype], rtol=RTOL[dtype]
            )
