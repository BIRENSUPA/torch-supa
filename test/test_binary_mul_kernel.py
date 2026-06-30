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

mixed_dtype_pairs = [
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
]

values = [0.5, 1.0, 1000]


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_mul(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    c_cpu = a_cpu * b_cpu
    c_supa = a_supa * b_supa

    assert_allclose(c_cpu, c_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_mul_broadcast(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu = torch.randn([1], dtype=dtype, device="cpu", requires_grad=False)
    b_supa = b_cpu.to("supa")

    c_cpu = a_cpu * b_cpu
    c_supa = a_supa * b_supa

    assert_allclose(c_cpu, c_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_mul_out(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    a_cpu.mul_(b_cpu)
    a_supa.mul_(b_supa)

    assert_allclose(a_cpu, a_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_mul_scalar_float(shape, dtype):
    x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)

    y_cpu = torch.mul(x_cpu, 3.0)
    y_supa = torch.mul(x_supa, 3.0)

    assert_allclose(y_cpu, y_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("value", values)
def test_mul_scalar_out(shape, dtype, value):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)

    a_cpu.mul_(value)
    a_supa.mul_(value)

    assert_allclose(a_cpu, a_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])



@pytest.mark.parametrize("lhs_dtype, rhs_dtype", mixed_dtype_pairs)
def test_mul_contiguous_mixed_dtype_to_float(lhs_dtype, rhs_dtype):
    shape = (131072, 128)
    a_cpu, a_supa = create_random_tensor(shape, dtype=lhs_dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=rhs_dtype)

    c_cpu = a_cpu * b_cpu
    c_supa = a_supa * b_supa

    assert c_cpu.dtype == torch.float32, f"Expected float32, got {c_cpu.dtype}"
    assert c_supa.dtype == torch.float32, f"Expected float32, got {c_supa.dtype}"
    assert_allclose(c_cpu, c_supa.cpu(), atol=ATOL[torch.float32], rtol=RTOL[torch.float32])


params_opt = [
    # 10 typical shapes from elementwise_bf16_float_diff_shape.csv
    pytest.param((1, 2048, 24, 64), (1, 2048, 1, 64), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]), 
    # 3D shape with broadcast
    pytest.param((1, 1024, 5120), (1, 1, 5120), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]), 
    # 2D shapes with broadcast (column vector * scalar)
    pytest.param((594, 4096), (594, 1), marks=[pytest.mark.gcuStress]),  # small size
    pytest.param((601, 4096), (601, 1), marks=[pytest.mark.gcuStress]),
    pytest.param((620, 4096), (620, 1), marks=[pytest.mark.gcuStress]),
    pytest.param((635, 4096), (635, 1), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((647, 4096), (647, 1), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),  # existing
    pytest.param((656, 4096), (656, 1), marks=[pytest.mark.gcuStress]),
    pytest.param((663, 4096), (663, 1), marks=[pytest.mark.gcuStress]),
    pytest.param((670, 4096), (670, 1), marks=[pytest.mark.gcuStress]),
    pytest.param((679, 4096), (679, 1), marks=[pytest.mark.gcuStress]),  # large size
]
@pytest.mark.parametrize("shape, shape1", params_opt)
def test_mul_bfloat16_float(shape, shape1):
    """Test bfloat16 * float32 = float32"""
    a_cpu, a_supa = create_random_tensor(shape, dtype=torch.bfloat16)
    b_cpu, b_supa = create_random_tensor(shape1, dtype=torch.float32)

    c_cpu = a_cpu * b_cpu
    c_supa = a_supa * b_supa


    assert c_cpu.dtype == torch.float32, f"Expected float32, got {c_cpu.dtype}"
    assert c_supa.dtype == torch.float32, f"Expected float32, got {c_supa.dtype}"
    assert_allclose(c_cpu, c_supa.cpu(), atol=ATOL[torch.float32], rtol=RTOL[torch.float32])
