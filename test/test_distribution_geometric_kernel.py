# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest

import torch
from torch_supa.testing.common_utils import create_random_tensor, RandomMode, assert_allclose

params = [
    pytest.param(
        (100),
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
probability = [0.3, 0.5, 0.7, 0.9]

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


@pytest.mark.parametrize("p", probability)
@pytest.mark.parametrize("shape", params)
def test_tensor(p, shape):
    torch.manual_seed(0)
    _, x_supa = create_random_tensor(
        shape, dtype=torch.float32, requires_grad=False, mode=RandomMode.uniform
    )
    y_supa = x_supa.geometric_(p)

    y_cpu = y_supa.cpu()
    y_cpu_var = torch.var(y_cpu)
    var_golden = (1 - p) / (p * p)

    print(y_cpu_var)
    print(var_golden)

    assert abs(y_cpu_var - var_golden) < 1


@pytest.mark.parametrize("shape", intern_vl_params)
@pytest.mark.parametrize("p", probability)
@pytest.mark.parametrize("seed", SEED)
def test_geometric(shape, p, seed):
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
    output0.geometric_(p)
    torch.manual_seed(seed)
    output1 = x_supa.clone()
    output1.geometric_(p)
    assert_allclose(output0, output1, rtol=0, atol=0)

    torch.manual_seed(seed)
    output0 = x_supa.geometric_(p)
    torch.manual_seed(seed)
    output1 = x_supa.geometric_(p)
    assert_allclose(output0, output1, rtol=0, atol=0)


@pytest.mark.parametrize("shape", intern_vl_params)
@pytest.mark.parametrize("p", probability)
@pytest.mark.parametrize("seed", SEED)
def test_geometric_bf16(shape, p, seed):
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
    output0.geometric_(p)
    torch.manual_seed(seed)
    output1 = x_supa.clone()
    output1.geometric_(p)
    assert_allclose(output0, output1, rtol=0, atol=0)

    torch.manual_seed(seed)
    output0 = x_supa.geometric_(p)
    torch.manual_seed(seed)
    output1 = x_supa.geometric_(p)
    assert_allclose(output0, output1, rtol=0, atol=0)


@pytest.mark.parametrize("shape", intern_vl_params)
@pytest.mark.parametrize("p", probability)
@pytest.mark.parametrize("seed", SEED)
def test_geometric_fp16(shape, p, seed):
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
    output0.geometric_(p)
    torch.manual_seed(seed)
    output1 = x_supa.clone()
    output1.geometric_(p)
    assert_allclose(output0, output1, rtol=0, atol=0)

    torch.manual_seed(seed)
    output0 = x_supa.geometric_(p)
    torch.manual_seed(seed)
    output1 = x_supa.geometric_(p)
    assert_allclose(output0, output1, rtol=0, atol=0)
