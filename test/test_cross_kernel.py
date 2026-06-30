# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# noqa
import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

ATOL = 8 * 1e-3
RTOL = 1 * 1e-5


shapes = [
    pytest.param(
        (4, 3),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 3), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 3), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 3), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 3), marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.float32]


@pytest.mark.parametrize("input_shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
def test_cross(input_shape, dtype):
    cpu_a, supa_a = create_random_tensor(input_shape, dtype=dtype)
    cpu_b, supa_b = create_random_tensor(input_shape, dtype=dtype)
    output_cpu = torch.cross(cpu_a, cpu_b)
    output_supa = torch.cross(supa_a, supa_b)

    assert_allclose(output_cpu, output_supa, rtol=1e-5, atol=5e-5)


@pytest.mark.parametrize("input_shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
def test_cross_dim1(input_shape, dtype):
    cpu_a, supa_a = create_random_tensor(input_shape, dtype=dtype)
    cpu_b, supa_b = create_random_tensor(input_shape, dtype=dtype)
    output_cpu = torch.cross(cpu_a, cpu_b, dim=1)
    output_supa = torch.cross(supa_a, supa_b, dim=1)

    assert_allclose(output_cpu, output_supa, rtol=1e-5, atol=5e-5)
