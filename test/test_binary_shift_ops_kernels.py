# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# noqa
import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

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
    pytest.param((512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.int8, torch.int32]


@pytest.mark.parametrize("shape", br200_shapes)
@pytest.mark.parametrize("dtype", dtypes)
def test_lshift(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)
    c_cpu = a_cpu << b_cpu
    c_supa = a_supa << b_supa

    assert_allclose(c_cpu, c_supa, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("shape", br200_shapes)
@pytest.mark.parametrize("dtype", dtypes)
def test_rshift(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)
    c_cpu = a_cpu >> b_cpu
    c_supa = a_supa >> b_supa

    assert_allclose(c_cpu, c_supa, rtol=0, atol=0, equal_nan=True)
