# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_equal, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


br200_params = [
    pytest.param(
        (32,),
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


class TestGCDLCM:

    @pytest.mark.parametrize("shape", br200_params)
    def test_gcd(self, shape):
        a_cpu, a_supa = create_random_tensor(shape, dtype=torch.int32)
        b_cpu, b_supa = create_random_tensor(shape, dtype=torch.int32)
        c_cpu = torch.gcd(a_cpu, b_cpu)
        c_supa = torch.gcd(a_supa, b_supa)
        assert_equal(c_cpu, c_supa)

    @pytest.mark.parametrize("shape", br200_params)
    def test_lcm(self, shape):
        a_cpu, a_supa = create_random_tensor(shape, dtype=torch.int32)
        b_cpu, b_supa = create_random_tensor(shape, dtype=torch.int32)
        c_cpu = torch.lcm(a_cpu, b_cpu)
        c_supa = torch.lcm(a_supa, b_supa)
        assert_equal(c_cpu, c_supa)
