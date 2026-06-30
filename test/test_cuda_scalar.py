# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# noqa
import copy

import pytest
import torch

params = [
    pytest.param(
        (2, 3, 3, 3),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (16, 16, 16, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((16, 16, 15, 2135), marks=[pytest.mark.gcuStress]),
    pytest.param((16, 16, 15, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((16, 15, 15, 1257), marks=[pytest.mark.gcuStress]),
]


dtypes = [torch.float32]


@pytest.mark.parametrize("shape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_local_scalar_dense(shape, dtype):
    x_cpu = torch.randn(shape, dtype=dtype)
    x_supa = copy.deepcopy(x_cpu).to("supa")
    o_cpu = torch.ops.aten._local_scalar_dense.default(x_cpu)
    o_supa = torch.ops.aten._local_scalar_dense.default(x_supa)
    assert o_cpu == o_supa
