# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

params = [
    pytest.param(
        (1, 4),
        (3, 4),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (512, 1024), (512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((1023, 511), (1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), (1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((2046, 2135), (2046, 2135), marks=[pytest.mark.gcuStress]),
]


params_scalar = [
    pytest.param(
        (5,),
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

dtypes = [
    torch.float32,
    torch.float,
    torch.bfloat16,
    torch.float16,
    torch.uint8,
    torch.int8,
    torch.int32,
    torch.int,
    torch.int64,
    torch.long,
    torch.bool,
]


@pytest.mark.parametrize("a, b", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_copysign(a, b, dtype):
    a_cpu, a_supa = create_random_tensor(a, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(b, dtype=dtype)

    y_cpu = torch.copysign(a_cpu, b_cpu)
    y_supa = torch.copysign(a_supa, b_supa)

    assert_allclose(y_cpu, y_supa.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("a", params_scalar)
def test_copysign_scalar(a):
    dtype = torch.float32
    a_cpu, a_supa = create_random_tensor(a, dtype=dtype)

    y_cpu = torch.copysign(a_cpu, 1)
    y_supa = torch.copysign(a_supa, 1)

    assert_allclose(y_cpu, y_supa.cpu(), rtol=0, atol=0)
