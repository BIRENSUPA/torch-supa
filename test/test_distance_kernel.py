# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


params = [
    pytest.param(
        (8, 8),
        2,
        torch.float32,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (512, 1024),
        2,
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param((511, 511), 1, torch.float32, marks=[pytest.mark.gcuStress]),
    pytest.param((1026, 516), 0, torch.float32, marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2138), 2, torch.float32, marks=[pytest.mark.gcuStress]),
]

params1 = [
    pytest.param(
        (2, 4, 8),
        (2, 2, 8),
        2,
        torch.float32,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (20, 40, 8),
        (20, 20, 8),
        2,
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (20, 80, 8),
        (20, 40, 8),
        0,
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (20, 62, 8),
        (20, 31, 8),
        1,
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (20, 72, 8),
        (20, 36, 8),
        2,
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
]


@pytest.mark.parametrize("input_shape, p, dtype", params)
def test_pdist(input_shape, p, dtype):
    x_cpu, x_supa = create_random_tensor(input_shape, dtype=dtype, requires_grad=True)
    y_cpu = torch.nn.functional.pdist(x_cpu, p=p)
    y_supa = torch.nn.functional.pdist(x_supa, p=p)
    assert_allclose(y_cpu, y_supa, rtol=1e-5, atol=5e-5)

    cpu_grad, supa_grad = create_random_tensor(
        y_cpu.shape, dtype=dtype, requires_grad=True
    )
    y_cpu.backward(cpu_grad)
    y_supa.backward(supa_grad)
    assert_allclose(x_cpu.grad, x_supa.grad, rtol=1e-5, atol=5e-5)


@pytest.mark.parametrize("input_shape1, input_shape2, p, dtype", params1)
def test_cdist(input_shape1, input_shape2, p, dtype):
    x_cpu1, x_supa1 = create_random_tensor(
        input_shape1, dtype=dtype, requires_grad=True
    )
    x_cpu2, x_supa2 = create_random_tensor(
        input_shape2, dtype=dtype, requires_grad=True
    )
    y_cpu = torch.cdist(x_cpu1, x_cpu2, p=p)
    y_supa = torch.cdist(x_supa1, x_supa2, p=p)
    assert_allclose(y_cpu, y_supa, rtol=1e-5, atol=5e-5)

    cpu_grad, supa_grad = create_random_tensor(
        y_cpu.shape, dtype=dtype, requires_grad=True
    )
    y_cpu.backward(cpu_grad)
    y_supa.backward(supa_grad)
    assert_allclose(x_cpu1.grad, x_supa1.grad, rtol=1e-5, atol=5e-5)
    assert_allclose(x_cpu2.grad, x_supa2.grad, rtol=1e-5, atol=5e-5)
