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
def test_ge_tensor_method(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    c_cpu = a_cpu >= b_cpu
    c_supa = a_supa >= b_supa

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_ge_tensor_function(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    c_cpu = torch.ge(a_cpu, b_cpu)
    c_supa = torch.ge(a_supa, b_supa)

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_ge_scalar_method(shape, dtype):
    scalar = 0.5
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = scalar, scalar

    c_cpu = a_cpu >= b_cpu
    c_supa = a_supa >= b_supa

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_ge_scalar_function(shape, dtype):
    scalar = 0.5
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = scalar, scalar

    c_cpu = torch.ge(a_cpu, b_cpu)
    c_supa = torch.ge(a_supa, b_supa)

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape_a, shape_b", shapes_broadcast)
@pytest.mark.parametrize("dtype", dtypes)
def test_ge_broadcast_function(shape_a, shape_b, dtype):
    a_cpu, a_supa = create_random_tensor(shape_a, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape_b, dtype=dtype)

    c_cpu = torch.ge(a_cpu, b_cpu)
    c_supa = torch.ge(a_supa, b_supa)

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_gt_tensor_method(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    c_cpu = a_cpu > b_cpu
    c_supa = a_supa > b_supa

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_gt_tensor_function(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    c_cpu = torch.gt(a_cpu, b_cpu)
    c_supa = torch.gt(a_supa, b_supa)

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_gt_scalar_method(shape, dtype):
    scalar = 0.5
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = scalar, scalar

    c_cpu = a_cpu > b_cpu
    c_supa = a_supa > b_supa

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_gt_scalar_function(shape, dtype):
    scalar = 0.5
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = scalar, scalar

    c_cpu = torch.gt(a_cpu, b_cpu)
    c_supa = torch.gt(a_supa, b_supa)

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape_a, shape_b", shapes_broadcast)
@pytest.mark.parametrize("dtype", dtypes)
def test_gt_broadcast_function(shape_a, shape_b, dtype):
    a_cpu, a_supa = create_random_tensor(shape_a, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape_b, dtype=dtype)

    c_cpu = torch.gt(a_cpu, b_cpu)
    c_supa = torch.gt(a_supa, b_supa)

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_le_tensor_method(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    c_cpu = a_cpu <= b_cpu
    c_supa = a_supa <= b_supa

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_le_tensor_function(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    c_cpu = torch.le(a_cpu, b_cpu)
    c_supa = torch.le(a_supa, b_supa)

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_le_scalar_method(shape, dtype):
    scalar = 0.5
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = scalar, scalar

    c_cpu = a_cpu <= b_cpu
    c_supa = a_supa <= b_supa

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_le_scalar_function(shape, dtype):
    scalar = 0.5
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = scalar, scalar

    c_cpu = torch.le(a_cpu, b_cpu)
    c_supa = torch.le(a_supa, b_supa)

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape_a, shape_b", shapes_broadcast)
@pytest.mark.parametrize("dtype", dtypes)
def test_le_broadcast_function(shape_a, shape_b, dtype):
    a_cpu, a_supa = create_random_tensor(shape_a, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape_b, dtype=dtype)

    c_cpu = torch.le(a_cpu, b_cpu)
    c_supa = torch.le(a_supa, b_supa)

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_lt_tensor_method(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    c_cpu = a_cpu < b_cpu
    c_supa = a_supa < b_supa

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_lt_tensor_function(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    c_cpu = torch.lt(a_cpu, b_cpu)
    c_supa = torch.lt(a_supa, b_supa)

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_lt_scalar_method(shape, dtype):
    scalar = 0.5
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = scalar, scalar

    c_cpu = a_cpu < b_cpu
    c_supa = a_supa < b_supa

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_lt_scalar_function(shape, dtype):
    scalar = 0.5
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = scalar, scalar

    c_cpu = torch.lt(a_cpu, b_cpu)
    c_supa = torch.lt(a_supa, b_supa)

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)


@pytest.mark.parametrize("shape_a, shape_b", shapes_broadcast)
@pytest.mark.parametrize("dtype", dtypes)
def test_lt_broadcast_function(shape_a, shape_b, dtype):
    a_cpu, a_supa = create_random_tensor(shape_a, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape_b, dtype=dtype)

    c_cpu = torch.lt(a_cpu, b_cpu)
    c_supa = torch.lt(a_supa, b_supa)

    assert_allclose(c_cpu, c_supa.cpu().to(torch.bool), rtol=0, atol=0)
