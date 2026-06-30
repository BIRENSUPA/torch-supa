# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor


tensor_shape = [
    pytest.param(
        (4,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((1024, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]

func_types2 = ["atan2", "hypot"]

inplace = [True, False]
input_types = ["scalar", "tensor"]
dtypes = [torch.float32, torch.bfloat16, torch.float16]

RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


class TestSingleMethod:

    @pytest.mark.parametrize("shape", tensor_shape)
    @pytest.mark.parametrize("func_type", func_types2)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("in_place", inplace)
    def test_single_ptwise2(self, shape, func_type, dtype, in_place):
        pt_func = eval("torch." + func_type)
        in_cpu, in_supa = create_random_tensor(shape, dtype)
        other_cpu, other_supa = create_random_tensor(shape, dtype)

        if in_place:
            pt_func_cpu = eval("in_cpu." + func_type + "_")
            pt_func_supa = eval("in_supa." + func_type + "_")
            pt_func_cpu(other_cpu)
            pt_func_supa(other_supa)
            assert_allclose(
                in_cpu, in_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
            )
        else:
            res_cpu = pt_func(in_cpu, other_cpu)
            res_supa = pt_func(in_supa, other_supa)
            assert_allclose(
                res_cpu, res_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
            )

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_isposinf(self):
        a = torch.tensor([-float("inf"), float("inf"), 1.2])
        b = torch.tensor([-float("inf"), float("inf"), 1.2]).to("supa")
        res_a = torch.isposinf(a)
        res_b = torch.isposinf(b)
        assert_allclose(res_a, res_b, rtol=0, atol=0, equal_nan=True)
