# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


params = [
    pytest.param(
        (16, 1),
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

shapes_broadcast = [
    pytest.param(
        (1, 1, 4, 4),
        (4, 4),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 2, 256, 256),
        (256, 256),
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param((6, 2, 255, 255), (255, 255), marks=[pytest.mark.gcuStress]),
    pytest.param((2, 2, 257, 257), (257, 257), marks=[pytest.mark.gcuStress]),
    pytest.param((2, 3, 511, 511), (511, 511), marks=[pytest.mark.gcuStress]),
]
dtypes = [torch.float32, torch.bfloat16, torch.float16]


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_eq(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    c_cpu = a_cpu == b_cpu
    c_supa = a_supa == b_supa

    assert_allclose(c_cpu, c_supa.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_eq_scalar_float(shape, dtype):
    x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)

    y_cpu = x_cpu == 1.0
    y_supa = x_supa == 1.0

    assert_allclose(y_cpu, y_supa.cpu(), rtol=0, atol=0)


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.regression
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
def test_eq_scalar_scalar():
    x_cpu = torch.tensor(1.0)
    x_supa = torch.tensor(1.0).to(device=supa_device)
    y_cpu = x_cpu == 1.0
    y_supa = x_supa == 1.0

    assert_allclose(y_cpu, y_supa.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("shape_a, shape_b", shapes_broadcast)
def test_eq_broadcast_function(shape_a, shape_b, dtype):
    a_cpu, a_supa = create_random_tensor(shape_a, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape_b, dtype=dtype)

    c_cpu = a_cpu == b_cpu
    c_supa = a_supa == b_supa

    assert_allclose(c_cpu, c_supa.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_eq_scalar_float_supa(shape, dtype):
    x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
    y_cpu = x_cpu == 1.0
    y_supa = x_supa == 1.0
    assert_allclose(y_cpu, y_supa.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_ne(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    c_cpu = a_cpu != b_cpu
    c_supa = a_supa != b_supa

    assert_allclose(c_cpu, c_supa.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_ne_scalar_float(shape, dtype):
    x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)

    y_cpu = x_cpu != 1.0
    y_supa = x_supa != 1.0

    assert_allclose(y_cpu, y_supa.cpu(), rtol=0, atol=0)
