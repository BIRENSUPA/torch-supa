# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    RandomMode,
)


br200_shapes = [
    pytest.param(
        (12,),
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
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.bool,
]

# data distribution, min_value, max_value
data_distributions = [("norm", -5, 5), ("uniform", 0, 0)]


class TestNonzero:

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("mode, min_value, max_value", data_distributions[:1])
    def test_nonzero_br200(self, shape, dtype, mode, min_value, max_value):
        def test_nonzero(shape, dtype, mode, min_value, max_value):
            if mode == "norm":
                cpu_input, supa_input = create_random_tensor(
                    shape,
                    dtype=dtype,
                    min_value=min_value,
                    max_value=max_value,
                    mode=RandomMode.norm,
                )
            elif mode == "uniform":
                cpu_input, supa_input = create_random_tensor(
                    shape,
                    dtype=dtype,
                    min_value=min_value,
                    max_value=max_value,
                    mode=RandomMode.uniform,
                )
            cpu_output = torch.nonzero(cpu_input)
            supa_output = torch.nonzero(supa_input)
            assert_allclose(cpu_output, supa_output.cpu(), rtol=1e-5, atol=5e-5)

        test_nonzero(shape, dtype, mode, min_value, max_value)
