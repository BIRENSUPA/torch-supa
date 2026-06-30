# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    RandomMode,
)

single_shapes = [
    pytest.param(
        (15,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
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
    pytest.param(
        (31,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((65536,), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]

empty_shapes = [
    pytest.param(
        (0,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (
            0,
            4,
        ),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (
            4,
            0,
        ),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((4, 0, 4), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((512, 0, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 0, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 0, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 0, 2135), marks=[pytest.mark.gcuStress]),
]

dtypes = [
    torch.float32,
    # torch.bfloat16, #NOTE: "unique" not implemented for 'BFloat16'
    torch.float16,
    torch.int32,
    torch.int64,
]

FLOAT_RTOL, FLOAT_ATOL = 1e-5, 5e-5


class TestUnique:

    @pytest.mark.parametrize("shape", single_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_unique_function(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(
            shape, dtype=dtype, min_value=-500, max_value=500, mode=RandomMode.uniform
        )

        y_cpu_value = torch.unique(x_cpu, True, False, False)
        y_supa_value = torch.unique(x_supa, True, False, False)

        # sort to keep value position same
        y_cpu, _ = torch.sort(y_cpu_value)
        y_supa_cpu, _ = torch.sort(y_supa_value.cpu())
        assert_allclose(y_cpu, y_supa_cpu, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)

    @pytest.mark.parametrize("shape", empty_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_unique_empty_function(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(
            shape, dtype=dtype, min_value=-500, max_value=500, mode=RandomMode.uniform
        )

        y_cpu_value = torch.unique(x_cpu, True, False, False)
        y_supa_value = torch.unique(x_supa, True, False, False)

        assert y_cpu_value.shape == y_supa_value.shape
        assert y_cpu_value.dtype == y_supa_value.dtype
