# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

params = [
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
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}

dtypes = [torch.float32, torch.bfloat16, torch.float16]

values = [0.5, 1.0, 1000]


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_add(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    c_cpu = a_cpu + b_cpu
    c_supa = a_supa + b_supa

    assert_allclose(c_cpu, c_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_add_broadcast(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu = torch.randn([1], dtype=dtype, device="cpu", requires_grad=False)
    b_supa = b_cpu.to("supa")

    c_cpu = a_cpu + b_cpu
    c_supa = a_supa + b_supa

    assert_allclose(c_cpu, c_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_add_out(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    a_cpu.add_(b_cpu)
    a_supa.add_(b_supa)

    assert_allclose(a_cpu, a_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_add_scalar(shape, dtype):
    x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)

    y_cpu = torch.add(x_cpu, 3.0)
    y_supa = torch.add(x_supa, 3.0)

    assert_allclose(y_cpu, y_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_add_strided(dtype):
    a_cpu, a_supa = create_random_tensor([1, 1024, 512], dtype=dtype)
    b_cpu, b_supa = create_random_tensor([1, 1024, 512], dtype=dtype)

    a_cpu = torch.as_strided(a_cpu, [1, 1024, 512], [524288, 1, 1024])
    a_supa = torch.as_strided(a_supa, [1, 1024, 512], [524288, 1, 1024])
    b_cpu = torch.as_strided(b_cpu, [1, 1024, 512], [524288, 512, 1])
    b_supa = torch.as_strided(b_supa, [1, 1024, 512], [524288, 512, 1])

    y_cpu = torch.add(a_cpu, b_cpu, alpha=1)
    y_supa = torch.add(a_supa, b_supa, alpha=1)

    assert_allclose(y_cpu, y_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_sub(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    c_cpu = a_cpu - b_cpu
    c_supa = a_supa - b_supa

    assert_allclose(c_cpu, c_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_sub_broadcast(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu = torch.randn([1], dtype=dtype, device="cpu", requires_grad=False)
    b_supa = b_cpu.to("supa")

    c_cpu = a_cpu - b_cpu
    c_supa = a_supa - b_supa

    assert_allclose(c_cpu, c_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_sub_out(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    a_cpu.sub_(b_cpu)
    a_supa.sub_(b_supa)

    assert_allclose(a_cpu, a_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_sub_scalar(shape, dtype):
    x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)

    y_cpu = torch.sub(x_cpu, 3.0)
    y_supa = torch.sub(x_supa, 3.0)

    assert_allclose(y_cpu, y_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])
