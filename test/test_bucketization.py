# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import copy

import numpy as np
import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

# shapes = [
#     (1, 50),
#     (3, 435),
#     (23, 435),
#     (13, 435),
#     (16, 434),
#     (64, 134),
#     (64, 23),
#     (64, 1000),
#     (32, 999),
#     (64, 456),
#     (32, 192, 64)
# ]


shapes = [
    pytest.param(
        (1, 8),
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

boundaries = [(4,)]

out_int32s = [
    True,
    # False
]

rights = [
    # True,
    False
]

dtypes = [torch.float32, torch.bfloat16, torch.float16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("boundaries", boundaries)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("out_int32", out_int32s)
@pytest.mark.parametrize("right", rights)
def test_bucketize(shape, boundaries, dtype, out_int32, right):
    x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
    boundaries_cpu = torch.from_numpy(
        copy.deepcopy(np.array(range(boundaries[0]), dtype=np.float32))
    ).to(cpu_device)
    boundaries_supa = torch.from_numpy(
        copy.deepcopy(np.array(range(boundaries[0]), dtype=np.float32))
    ).to(supa_device)
    boundaries_supa = boundaries_cpu.to(supa_device)
    out_cpu = torch.bucketize(x_cpu, boundaries_cpu, out_int32=out_int32, right=right)
    out_supa = torch.bucketize(
        x_supa, boundaries_supa, out_int32=out_int32, right=right
    )
    assert_allclose(out_cpu, out_supa, atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("boundaries", boundaries)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("out_int32", out_int32s)
@pytest.mark.parametrize("right", rights)
def test_searchsorted(shape, boundaries, dtype, out_int32, right):
    sorted_sequence = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
    values = torch.tensor([[3, 6, 9], [3, 6, 9]])
    sorted_sequence_supa = copy.deepcopy(sorted_sequence).to("supa")
    values_supa = copy.deepcopy(values).to("supa")

    out_cpu = torch.searchsorted(sorted_sequence, values)
    out_supa = torch.searchsorted(sorted_sequence_supa, values_supa)
    assert_allclose(out_cpu, out_supa, atol=ATOL[dtype], rtol=RTOL[dtype])
