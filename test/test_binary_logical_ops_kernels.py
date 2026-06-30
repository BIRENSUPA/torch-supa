# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor


params = [
    pytest.param(
        (1, 2, 4, 4),
        True,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (1, 2, 4, 4),
        False,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (1024, 512), True, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param(
        (1024, 512), False, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((1023, 511), True, marks=[pytest.mark.gcuStress]),
    pytest.param((1023, 511), False, marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), True, marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), False, marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), True, marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), False, marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.float32, torch.bfloat16, torch.float16, torch.bool]
RTOL = 0
ATOL = 0


@pytest.mark.parametrize("shape, in_place", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_logical_not(shape, in_place, dtype):
    input_cpu, input_supa = create_random_tensor(shape, dtype)
    if in_place:
        output_cpu = input_cpu.logical_not_()
        output_supa = input_supa.logical_not_()
    else:
        output_cpu = torch.logical_not(input_cpu)
        output_supa = torch.logical_not(input_supa)

    assert_allclose(output_cpu, output_supa, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("shape, in_place", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_logical_and(shape, in_place, dtype):
    input_cpu, input_supa = create_random_tensor(shape, dtype)
    input2_cpu, input2_supa = create_random_tensor(shape, dtype)
    if in_place:
        output_cpu = input_cpu.logical_and_(input2_cpu)
        output_supa = input_supa.logical_and_(input2_supa)
    else:
        output_cpu = torch.logical_and(input_cpu, input2_cpu)
        output_supa = torch.logical_and(input_supa, input2_supa)

    assert_allclose(output_cpu, output_supa, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("shape, in_place", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_logical_or(shape, in_place, dtype):
    input_cpu, input_supa = create_random_tensor(shape, dtype)
    input2_cpu, input2_supa = create_random_tensor(shape, dtype)
    if in_place:
        output_cpu = input_cpu.logical_or_(input2_cpu)
        output_supa = input_supa.logical_or_(input2_supa)
    else:
        output_cpu = torch.logical_or(input_cpu, input2_cpu)
        output_supa = torch.logical_or(input_supa, input2_supa)

    assert_allclose(output_cpu, output_supa, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("shape, in_place", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_logical_xor(shape, in_place, dtype):
    input_cpu, input_supa = create_random_tensor(shape, dtype)
    input2_cpu, input2_supa = create_random_tensor(shape, dtype)
    if in_place:
        output_cpu = input_cpu.logical_xor_(input2_cpu)
        output_supa = input_supa.logical_xor_(input2_supa)
    else:
        output_cpu = torch.logical_xor(input_cpu, input2_cpu)
        output_supa = torch.logical_xor(input_supa, input2_supa)

    assert_allclose(output_cpu, output_supa, atol=ATOL, rtol=RTOL)
