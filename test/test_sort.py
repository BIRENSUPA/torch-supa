# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor, RandomMode

br200_shapes = [
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
    pytest.param((512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]

dtypes = [
    torch.float32,
    torch.int32,
    torch.int64,
    torch.bfloat16,
    torch.float16,
]

descendings = [
    True,
    False,
]

FLOAT_RTOL, FLOAT_ATOL = 1e-5, 5e-5


class TestSort:

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("descending", descendings)
    def test_sort_br200(self, shape, dtype, descending):
        def test_sort_function(shape, dtype, descending):
            x_cpu, x_supa = create_random_tensor(
                shape,
                dtype=dtype,
                min_value=-500,
                max_value=500,
                mode=RandomMode.uniform,
            )

            y_cpu_value, y_cpu_idx = torch.sort(
                x_cpu, descending=descending, stable=True
            )
            y_supa_value, y_supa_idx = torch.sort(
                x_supa, descending=descending, stable=True
            )

            assert_allclose(y_cpu_value, y_supa_value, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)
            assert_allclose(y_cpu_idx, y_supa_idx, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)

        test_sort_function(shape, dtype, descending)
