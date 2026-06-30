# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_equal, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


br200_params = [
    pytest.param(
        (1, 16),
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

dtypes = [torch.float32, torch.bfloat16, torch.float16]


class TestEqual:
    @pytest.mark.parametrize("shape", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_equal(self, shape, dtype):
        a_cpu, a_supa = create_random_tensor(shape, dtype=dtype)
        b_cpu, b_supa = create_random_tensor(shape, dtype=dtype)

        c_cpu = torch.equal(a_cpu, b_cpu)
        c_supa = torch.equal(a_supa, b_supa)
        a = torch.tensor([c_cpu], dtype=torch.bool)
        b = torch.tensor([c_supa], dtype=torch.bool)

        assert_equal(a, b)
