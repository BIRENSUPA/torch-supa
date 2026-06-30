# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor


tensor_shape = [
    pytest.param(
        (4,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((1024, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
]

dtypes = [torch.float32]
RTOL = {torch.float32: 5e-5, torch.bfloat16: 0.016}
ATOL = {torch.float32: 5e-5, torch.bfloat16: 1e-3}


@pytest.mark.parametrize("shape", tensor_shape)
@pytest.mark.parametrize("dtype", dtypes)
def test_sigmoid(shape, dtype):
    x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)
    y_cpu = torch.sigmoid(x_cpu)
    y_supa = torch.sigmoid(x_supa)

    cpu_grad, supa_grad = create_random_tensor(shape, dtype=dtype)
    y_cpu.backward(cpu_grad)
    cpu_in_grad = x_cpu.grad

    y_supa.backward(supa_grad)
    supa_in_grad = x_supa.grad
    assert_allclose(cpu_in_grad, supa_in_grad.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", tensor_shape)
@pytest.mark.parametrize("dtype", dtypes)
def test_logit(shape, dtype):
    x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)
    y_cpu = torch.logit(x_cpu)
    y_supa = torch.logit(x_supa)

    cpu_grad_data, supa_grad_data = create_random_tensor(
        y_cpu.shape, min_value=0, max_value=1, dtype=dtype
    )
    y_cpu.backward(cpu_grad_data)
    y_supa.backward(supa_grad_data)
    cpu_grad = x_cpu.grad.clone()
    supa_grad = x_supa.grad
    assert_allclose(
        y_cpu, y_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
    )
    assert_allclose(
        cpu_grad, supa_grad.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
    )


@pytest.mark.parametrize("shape", tensor_shape)
@pytest.mark.parametrize("dtype", dtypes)
def test_tanh(shape, dtype):
    tanh_cpu = torch.nn.Tanh()
    tanh_supa = tanh_cpu.supa()

    x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)
    dy_cpu, dy_supa = create_random_tensor(shape, dtype=dtype)

    y_cpu = tanh_cpu(x_cpu)
    y_cpu.backward(dy_cpu)

    y_supa = tanh_supa(x_supa)
    y_supa.backward(dy_supa)

    assert_allclose(x_cpu.grad, x_supa.grad, atol=ATOL[dtype], rtol=RTOL[dtype])
