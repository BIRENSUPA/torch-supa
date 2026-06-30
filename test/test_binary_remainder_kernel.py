# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

# noqa
from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    RandomMode,
)

params = [
    pytest.param(
        (1, 3, 2),
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
        (16, 32, 1024),
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param((15, 1023, 31), torch.float32, marks=[pytest.mark.gcuStress]),
    pytest.param((17, 1025, 513), torch.float32, marks=[pytest.mark.gcuStress]),
    pytest.param((50, 1000, 300), torch.float32, marks=[pytest.mark.gcuStress]),
]

RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


@pytest.mark.parametrize("shape, dtype", params)
def test_remainder(shape, dtype):
    cpu_input1, supa_input1 = create_random_tensor(
        shape, dtype=dtype, mode=RandomMode.uniform
    )
    cpu_input2, supa_input2 = create_random_tensor(
        shape, dtype=dtype, mode=RandomMode.uniform
    )
    output_cpu = cpu_input1 % cpu_input2
    output_supa = supa_input1 % supa_input2

    assert_allclose(output_cpu, output_supa, atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape, dtype", params)
def test_remainder_backward(shape, dtype):
    cpu_input1, supa_input1 = create_random_tensor(
        shape, dtype=dtype, requires_grad=True, mode=RandomMode.uniform
    )
    cpu_input2, supa_input2 = create_random_tensor(
        shape, dtype=dtype, mode=RandomMode.uniform
    )
    output_cpu = torch.remainder(cpu_input1, cpu_input2)
    output_supa = torch.remainder(supa_input1, supa_input2)
    cpu_grad, supa_grad = create_random_tensor(
        output_cpu.shape, dtype=dtype, requires_grad=False
    )
    output_cpu.backward(cpu_grad)
    output_supa.backward(supa_grad)

    assert_allclose(
        cpu_input1.grad, supa_input1.grad, atol=ATOL[dtype], rtol=RTOL[dtype]
    )


@pytest.mark.parametrize("shape, dtype", params)
def test_remainder_inplace(shape, dtype):
    cpu_input1, supa_input1 = create_random_tensor(
        shape, dtype=dtype, mode=RandomMode.uniform
    )
    cpu_input2, supa_input2 = create_random_tensor(
        shape, dtype=dtype, mode=RandomMode.uniform
    )
    cpu_input1.remainder_(cpu_input2)
    supa_input1.remainder_(supa_input2)

    assert_allclose(cpu_input1, supa_input1, atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape, dtype", params)
def test_remainder_scalar_float(shape, dtype):
    x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, mode=RandomMode.uniform)

    y_cpu = torch.remainder(x_cpu, 3.0)
    y_supa = torch.remainder(x_supa, 3.0)

    assert_allclose(y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape, dtype", params)
def test_remainder_int64(shape, dtype):
    dtype = torch.int64
    cpu_input1, supa_input1 = create_random_tensor(
        shape, dtype=dtype, min_value=-10000, max_value=10000
    )

    cpu_input2_pos, supa_input2_pos = create_random_tensor(
        shape, dtype=dtype, min_value=1, max_value=10000
    )
    cpu_input2_neg, supa_input2_neg = create_random_tensor(
        shape, dtype=dtype, min_value=-10000, max_value=-1
    )
    output_cpu_pos = cpu_input1 % cpu_input2_pos
    output_supa_pos = supa_input1 % supa_input2_pos

    output_cpu_neg = cpu_input1 % cpu_input2_neg
    output_supa_neg = supa_input1 % supa_input2_neg

    assert_allclose(output_cpu_pos, output_supa_pos, rtol=0, atol=0)
    assert_allclose(output_cpu_neg, output_supa_neg, rtol=0, atol=0)


@pytest.mark.parametrize("shape, dtype", params)
def test_remainder_scalar_int64(shape, dtype):
    dtype = torch.int64
    x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)

    import random

    rand_ele = random.choice([-1, -2, -3, -4, -5, 5, 4, 3, 2, 1])
    y_cpu = torch.remainder(x_cpu, rand_ele)
    y_supa = torch.remainder(x_supa, rand_ele)

    assert_allclose(y_cpu, y_supa, rtol=0, atol=0)


@pytest.mark.parametrize("shape, dtype", params)
def test_fmod(shape, dtype):
    a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
    b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

    c_cpu = torch.fmod(a_cpu, b_cpu)
    c_supa = torch.fmod(a_supa, b_supa)

    assert_allclose(c_cpu, c_supa, atol=ATOL[dtype], rtol=RTOL[dtype])
