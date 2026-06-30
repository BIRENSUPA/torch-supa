# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import math
import random

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

br200_shape = [
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
    pytest.param((512, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1022, 500), marks=[pytest.mark.gcuStress]),
]
func_types = ["_foreach_norm"]

dtypes = [torch.float32, torch.float16, torch.bfloat16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


def GetAccumulationRtolAtol(accumulation_cnt, dtype):
    alpha = 4.0
    if accumulation_cnt <= 32:
        k1 = alpha
    else:
        k1 = alpha * (math.log2(accumulation_cnt) - 4.0)

    cur_check_atol = k1 * 1e-5
    cur_check_rtol = k1 * 1.3e-6
    if dtype == torch.bfloat16 or dtype == torch.float16:
        cur_check_atol = 0.001
        cur_check_rtol = 0.016
    return cur_check_atol, cur_check_rtol


class TestForeachReduceMethod:

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("func_type", func_types)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_foreach_reduce_op(self, shape, func_type, dtype):
        pt_func = eval("torch." + func_type)

        in_cpu, in_supa = create_random_tensor(shape, dtype)
        in_cpu2, in_supa2 = create_random_tensor(shape, dtype)

        in_cpu_list1 = [in_cpu, in_cpu2]
        in_supa_list1 = [in_supa, in_supa2]

        norm_type = random.randint(1, 50)
        res_cpu = pt_func(in_cpu_list1, norm_type)
        res_supa = pt_func(in_supa_list1, norm_type)

        atol, rtol = GetAccumulationRtolAtol(in_cpu.view(-1).size()[0], dtype)
        assert_allclose(res_cpu[0], res_supa[0], atol=atol, rtol=rtol, equal_nan=True)
        assert_allclose(res_cpu[1], res_supa[1], atol=atol, rtol=rtol, equal_nan=True)
