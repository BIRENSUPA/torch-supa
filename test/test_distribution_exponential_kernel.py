# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest

import torch
from torch_supa.testing.common_utils import create_random_tensor, RandomMode, assert_allclose

params = [
    pytest.param(
        (1000),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1022, 514), marks=[pytest.mark.gcuStress]),
    pytest.param((1026, 516), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2138), marks=[pytest.mark.gcuStress]),
]

SEED = [6, 10]
intern_vl_params = [
    pytest.param(
        (2, 1, 1),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (3, 1, 1),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (4, 1, 1),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (5, 1, 1),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (6, 1, 1),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (7, 1, 1),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((200, 1, 1), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((2200, 1, 1), marks=[pytest.mark.gcuStress]),
    pytest.param((2230, 1, 1), marks=[pytest.mark.gcuStress]),
]


@pytest.mark.parametrize("shape", params)
def test_tensor(shape):
    _, x_supa = create_random_tensor(
        shape, dtype=torch.float32, requires_grad=False, mode=RandomMode.uniform
    )
    y_supa = x_supa.exponential_()
    y_cpu = y_supa.cpu()
    y_cpu_mean = torch.mean(y_cpu)
    y_cpu_std = torch.std(y_cpu)
    print(y_cpu_mean)
    print(y_cpu_std)

    assert abs(y_cpu_mean - y_cpu_std) < 0.1


@pytest.mark.parametrize("shape", intern_vl_params)
@pytest.mark.parametrize("seed", SEED)
def test_exponential(shape, seed):
    _, x_supa = create_random_tensor(
        shape,
        dtype=torch.float32,
        requires_grad=False,
        min_value=0,
        max_value=1,
        mode=RandomMode.uniform,
    )
    torch.manual_seed(seed)
    output0 = x_supa.clone()
    output0.exponential_()
    torch.manual_seed(seed)
    output1 = x_supa.clone()
    output1.exponential_()
    assert_allclose(output0, output1, rtol=0, atol=0)

    torch.manual_seed(seed)
    output0 = x_supa.exponential_()
    torch.manual_seed(seed)
    output1 = x_supa.exponential_()
    assert_allclose(output0, output1, rtol=0, atol=0)


@pytest.mark.parametrize("shape", intern_vl_params)
@pytest.mark.parametrize("seed", SEED)
def test_exponential_bf16(shape, seed):
    _, x_supa = create_random_tensor(
        shape,
        dtype=torch.bfloat16,
        requires_grad=False,
        min_value=0,
        max_value=1,
        mode=RandomMode.uniform,
    )
    torch.manual_seed(seed)
    output0 = x_supa.clone()
    output0.exponential_()
    torch.manual_seed(seed)
    output1 = x_supa.clone()
    output1.exponential_()
    assert_allclose(output0, output1, rtol=0, atol=0)

    torch.manual_seed(seed)
    output0 = x_supa.exponential_()
    torch.manual_seed(seed)
    output1 = x_supa.exponential_()
    assert_allclose(output0, output1, rtol=0, atol=0)


@pytest.mark.parametrize("shape", intern_vl_params)
@pytest.mark.parametrize("seed", SEED)
def test_exponential_fp16(shape, seed):
    _, x_supa = create_random_tensor(
        shape,
        dtype=torch.float16,
        requires_grad=False,
        min_value=0,
        max_value=1,
        mode=RandomMode.uniform,
    )
    torch.manual_seed(seed)
    output0 = x_supa.clone()
    output0.exponential_()
    torch.manual_seed(seed)
    output1 = x_supa.clone()
    output1.exponential_()
    assert_allclose(output0, output1, rtol=0, atol=0)

    torch.manual_seed(seed)
    output0 = x_supa.exponential_()
    torch.manual_seed(seed)
    output1 = x_supa.exponential_()
    assert_allclose(output0, output1, rtol=0, atol=0)
