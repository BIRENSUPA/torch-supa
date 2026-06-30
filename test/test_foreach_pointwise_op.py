# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

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
    pytest.param((512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]

func_types = ["_foreach_addcmul", "_foreach_addcdiv"]
input_types = ["scalar", "tensor", "scalar_list"]
dtypes = [torch.float32, torch.float16, torch.bfloat16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


class TestForeachPointwiseMethod:

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("input_type", input_types)
    @pytest.mark.parametrize("func_type", func_types)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_foreach_pointwise_op(self, shape, input_type, func_type, dtype):
        pt_func = eval("torch." + func_type)

        in_cpu, in_supa = create_random_tensor(shape, dtype)
        in_cpu2, in_supa2 = create_random_tensor(shape, dtype)
        in_cpu3, in_supa3 = create_random_tensor(shape, dtype)
        in_cpu_list1 = [in_cpu, in_cpu]
        in_supa_list1 = [in_supa, in_supa]

        in_cpu_list2 = [in_cpu2, in_cpu2]
        in_supa_list2 = [in_supa2, in_supa2]

        in_cpu_list3 = [in_cpu3, in_cpu3]
        in_supa_list3 = [in_supa3, in_supa3]

        if input_type == "scalar":
            value_cpu = 2.0
            value_supa = 2.0
        elif input_type == "tensor":
            value_cpu, value_supa = create_random_tensor([2], dtype)
        elif input_type == "scalar_list":
            value_cpu = [2, 2]
            value_supa = [2, 2]
        res_cpu = pt_func(in_cpu_list1, in_cpu_list2, in_cpu_list3, value_cpu)
        res_supa = pt_func(in_supa_list1, in_supa_list2, in_supa_list3, value_cpu)

        assert_allclose(
            res_cpu[0], res_supa[0], atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
        assert_allclose(
            res_cpu[1], res_supa[1], atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
