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
func_types = [
    "_foreach_abs",
    "_foreach_acos",
    "_foreach_asin",
    "_foreach_atan",
    "_foreach_ceil",
    "_foreach_cos",
    "_foreach_cosh",
    "_foreach_erf",
    "_foreach_erfc",
    "_foreach_exp",
    "_foreach_expm1",
    "_foreach_floor",
    "_foreach_frac",
    "_foreach_lgamma",
    "_foreach_log10",
    "_foreach_log1p",
    "_foreach_log2",
    "_foreach_log",
    "_foreach_neg",
    "_foreach_reciprocal",
    "_foreach_round",
    "_foreach_sigmoid",
    "_foreach_sign",
    "_foreach_sin",
    "_foreach_sinh",
    "_foreach_sqrt",
    "_foreach_tan",
    "_foreach_tanh",
    "_foreach_trunc",
    "_foreach_zero_",
]

dtypes = [torch.float32, torch.float16, torch.bfloat16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


class TestForeachUnaryMethod:

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("func_type", func_types)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_foreach_unary_op(self, shape, func_type, dtype):
        pt_func = eval("torch." + func_type)

        in_cpu, in_supa = create_random_tensor(shape, dtype)
        in_cpu2, in_supa2 = create_random_tensor(shape, dtype)

        in_cpu_list = [in_cpu, in_cpu2]
        in_supa_list = [in_supa, in_supa2]

        res_cpu = pt_func(in_cpu_list)

        # NOTE: "foreach_unary_op_cuda: _foreach_lgamma" not implemented for 'BFloat16'
        if dtype != torch.bfloat16 and func_type != "_foreach_lgamma":
            res_supa = pt_func(in_supa_list)
            assert_allclose(
                res_cpu[0],
                res_supa[0],
                atol=ATOL[dtype],
                rtol=RTOL[dtype],
                equal_nan=True,
            )
            assert_allclose(
                res_cpu[1],
                res_supa[1],
                atol=ATOL[dtype],
                rtol=RTOL[dtype],
                equal_nan=True,
            )
