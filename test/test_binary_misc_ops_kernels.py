# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    RandomMode,
)


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
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]

reductions = [
    "none",
    "mean",
    "sum",
]

betas = [0.5, 1.0, 0.2]
dtypes = [torch.float32, torch.bfloat16, torch.float16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


@pytest.mark.parametrize("shape", tensor_shape)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("reduction", reductions)
def test_smooth_l1_loss(shape, dtype, reduction):
    x_cpu, x_supa = create_random_tensor(
        shape,
        dtype=dtype,
        min_value=-1,
        max_value=1,
        requires_grad=True,
        mode=RandomMode.uniform,
    )
    target_cpu, target_supa = create_random_tensor(
        shape,
        dtype=dtype,
        min_value=-1,
        max_value=1,
        requires_grad=True,
        mode=RandomMode.uniform,
    )

    loss = torch.nn.SmoothL1Loss(reduction=reduction)
    y_cpu = loss(x_cpu.to(float), target_cpu)

    # target_supa = target_cpu.to(supa_device)
    loss_supa = loss.to(supa_device)
    y_supa = loss_supa(x_supa, target_supa)

    if reduction == "none":
        cpu_grad_data, supa_grad_data = create_random_tensor(
            x_cpu.shape, min_value=0, max_value=1, dtype=dtype, mode=RandomMode.uniform
        )
        y_cpu.backward(cpu_grad_data.to(float))
        y_supa.backward(supa_grad_data)
    else:
        y_cpu.backward()

        y_supa.backward()

    cpu_grad = x_cpu.grad.clone()
    supa_grad = x_supa.grad

    assert_allclose(cpu_grad.to(dtype), supa_grad, atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", tensor_shape)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("reduction", reductions)
def test_huber_loss(shape, reduction, dtype):
    x_cpu, x_supa = create_random_tensor(
        shape,
        dtype=dtype,
        min_value=-1,
        max_value=1,
        requires_grad=True,
        mode=RandomMode.uniform,
    )
    target_cpu, target_supa = create_random_tensor(
        shape,
        dtype=dtype,
        min_value=-1,
        max_value=1,
        requires_grad=True,
        mode=RandomMode.uniform,
    )

    loss = torch.nn.HuberLoss(reduction=reduction)
    y_cpu = loss(x_cpu, target_cpu)

    loss_supa = loss.to(supa_device)
    y_supa = loss_supa(x_supa, target_supa)

    if reduction == "none":
        cpu_grad_data, supa_grad_data = create_random_tensor(
            x_cpu.shape, min_value=0, max_value=1, dtype=dtype, mode=RandomMode.uniform
        )
        y_cpu.backward(cpu_grad_data)
        y_supa.backward(supa_grad_data)
    else:
        y_cpu.backward()
        y_supa.backward()

    cpu_grad = x_cpu.grad.clone()
    supa_grad = x_supa.grad

    assert_allclose(cpu_grad, supa_grad, atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", tensor_shape)
@pytest.mark.parametrize("reduction", reductions)
@pytest.mark.parametrize("dtype", dtypes)
def test_mse_loss(shape, reduction, dtype):
    x_cpu, x_supa = create_random_tensor(
        shape,
        dtype=dtype,
        min_value=-1,
        max_value=1,
        requires_grad=True,
        mode=RandomMode.uniform,
    )
    target_cpu, target_supa = create_random_tensor(
        shape,
        dtype=dtype,
        min_value=-1,
        max_value=1,
        requires_grad=True,
        mode=RandomMode.uniform,
    )

    loss = torch.nn.MSELoss(reduction=reduction)
    y_cpu = loss(x_cpu.float(), target_cpu.float())

    loss_supa = loss.to(supa_device)
    y_supa = loss_supa(x_supa, target_supa)

    if reduction == "none":
        cpu_grad_data, supa_grad_data = create_random_tensor(
            x_cpu.shape, min_value=0, max_value=1, dtype=dtype, mode=RandomMode.uniform
        )
        y_cpu.backward(cpu_grad_data.float())
        y_supa.backward(supa_grad_data)
    else:
        y_cpu.backward()
        y_supa.backward()

    cpu_grad = x_cpu.grad.clone()
    supa_grad = x_supa.grad

    assert_allclose(cpu_grad.to(dtype), supa_grad, atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize(
    "shape, dtype, reduction, with_weight, with_pos_weight, pos_weight_shape",
    [
        pytest.param((4,), torch.float32, "none", False, False, None),
        pytest.param((1024, 512), torch.float32, "sum", True, False, None),
        pytest.param((1024, 512), torch.bfloat16, "mean", False, True, None),
        pytest.param((1023, 511), torch.float16, "none", True, True, None),
        pytest.param((1025, 513), torch.float16, "sum", True, True, None),
        pytest.param((2, 3, 4), torch.float32, "mean", False, True, (3, 1)),
    ],
)
def test_binary_cross_entropy_with_logits_loss(
    shape, reduction, dtype, with_weight, with_pos_weight, pos_weight_shape
):
    input_cpu, input_supa = create_random_tensor(
        shape,
        dtype=dtype,
        min_value=-5,
        max_value=5,
        mode=RandomMode.uniform,
    )
    target_cpu, target_supa = create_random_tensor(
        shape,
        dtype=dtype,
        min_value=0,
        max_value=1,
        mode=RandomMode.uniform,
    )

    weight_cpu = weight_supa = None
    if with_weight:
        weight_cpu, weight_supa = create_random_tensor(
            shape,
            dtype=dtype,
            min_value=0.1,
            max_value=2,
            mode=RandomMode.uniform,
        )

    pos_weight_cpu = pos_weight_supa = None
    if with_pos_weight:
        if pos_weight_shape is None:
            pos_weight_shape = (shape[-1],) if len(shape) > 0 else shape
        pos_weight_cpu, pos_weight_supa = create_random_tensor(
            pos_weight_shape,
            dtype=dtype,
            min_value=0.1,
            max_value=3,
            mode=RandomMode.uniform,
        )

    y_cpu = torch.nn.functional.binary_cross_entropy_with_logits(
        input_cpu.float(),
        target_cpu.float(),
        weight=None if weight_cpu is None else weight_cpu.float(),
        pos_weight=None if pos_weight_cpu is None else pos_weight_cpu.float(),
        reduction=reduction,
    )
    y_supa = torch.nn.functional.binary_cross_entropy_with_logits(
        input_supa,
        target_supa,
        weight=weight_supa,
        pos_weight=pos_weight_supa,
        reduction=reduction,
    )

    expected = y_cpu.to(y_supa.dtype)
    assert_allclose(expected, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", tensor_shape)
@pytest.mark.parametrize("dtype", dtypes)
def test_xlogy(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)
    c_cpu = torch.xlogy(a_cpu, b_cpu)
    c_supa = torch.xlogy(a_supa, b_supa)

    assert_allclose(c_cpu, c_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True)


@pytest.mark.parametrize("shape", tensor_shape)
@pytest.mark.parametrize("dtype", dtypes)
def test_xlog1py(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)
    c_cpu = torch.special.xlog1py(a_cpu, b_cpu)
    c_supa = torch.special.xlog1py(a_supa, b_supa)

    assert_allclose(c_cpu, c_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True)
