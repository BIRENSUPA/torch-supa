# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    RandomMode,
)

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

params = [
    pytest.param(
        (2,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
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


dtypes = [torch.float32, torch.bfloat16, torch.float16]
base_nums = ["2", "10", "e", "1p"]

atol = 5e-4
rtol = 1e-4
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


class TestLog:
    def get_result(self, cpu_input, supa_input, cpu_ouput, supa_ouput, base_num):
        y_cpu, y_cpu_log_, y_supa, y_supa_log_ = 0, 0, 0, 0
        if base_num == "2":
            y_cpu = torch.log2(cpu_input)
            y_supa = torch.log2(supa_input)

            y_cpu_log_ = cpu_input.log2_()
            y_supa_log_ = supa_input.log2_()

            torch.log2(cpu_input, out=cpu_ouput)
            torch.log2(supa_input, out=supa_ouput)
        elif base_num == "e":
            y_cpu = torch.log(cpu_input)
            y_supa = torch.log(supa_input)

            y_cpu_log_ = cpu_input.log_()
            y_supa_log_ = supa_input.log_()

            torch.log(cpu_input, out=cpu_ouput)
            torch.log(supa_input, out=supa_ouput)
        elif base_num == "10":
            y_cpu = torch.log10(cpu_input)
            y_supa = torch.log10(supa_input)

            y_cpu_log_ = cpu_input.log10_()
            y_supa_log_ = supa_input.log10_()

            torch.log10(cpu_input, out=cpu_ouput)
            torch.log10(supa_input, out=supa_ouput)
        elif base_num == "1p":
            y_cpu = torch.log1p(cpu_input)
            y_supa = torch.log1p(supa_input)

            y_cpu_log_ = cpu_input.log1p_()
            y_supa_log_ = supa_input.log1p_()

            torch.log1p(cpu_input, out=cpu_ouput)
            torch.log1p(supa_input, out=supa_ouput)

        return y_cpu, y_cpu_log_, cpu_ouput, y_supa, y_supa_log_, supa_ouput

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("base_num", base_nums)
    def test_log_function(self, shape, dtype, base_num):
        cpu_input, supa_input = create_random_tensor(
            shape, min_value=1.0, max_value=10.0, dtype=dtype, mode=RandomMode.uniform
        )
        cpu_ouput, supa_ouput = create_random_tensor(
            shape, min_value=1.0, max_value=10.0, dtype=dtype, mode=RandomMode.uniform
        )

        y_cpu, y_cpu_log_, cpu_ouput, y_supa, y_supa_log_, supa_ouput = self.get_result(
            cpu_input, supa_input, cpu_ouput, supa_ouput, base_num
        )

        assert_allclose(
            y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )

        assert_allclose(
            y_cpu_log_, y_supa_log_, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )

        assert_allclose(
            cpu_ouput, supa_ouput, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
