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
    "_foreach_add",
    "_foreach_sub",
    "_foreach_mul",
    "_foreach_div",
    "_foreach_clamp_max",
    "_foreach_clamp_min",
]

dtypes = [torch.float32, torch.float16, torch.bfloat16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


class TestForeachBinaryMethod:

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("func_type", func_types)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_foreach_binary_op_scalar_list(self, shape, func_type, dtype):
        pt_func = eval("torch." + func_type)

        in_cpu, in_supa = create_random_tensor(shape, dtype)
        in_cpu2, in_supa2 = create_random_tensor(shape, dtype)

        in_cpu_list = [in_cpu, in_cpu2]
        in_supa_list = [in_supa, in_supa2]

        scalar_cpu = torch.tensor(2.0, dtype=dtype).to(cpu_device)
        scalar_supa = torch.tensor(2.0, dtype=dtype).to(supa_device)

        scalar_cpu1 = torch.tensor(3.0, dtype=dtype).to(cpu_device)
        scalar_supa1 = torch.tensor(3.0, dtype=dtype).to(supa_device)

        res_cpu = pt_func(in_cpu_list, [scalar_cpu, scalar_cpu1])
        res_supa = pt_func(in_supa_list, [scalar_supa, scalar_supa1])

        assert_allclose(
            res_cpu[0], res_supa[0], atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
        assert_allclose(
            res_cpu[1], res_supa[1], atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_foreach_pow_scalar_list(self, shape, dtype):
        in_cpu, in_supa = create_random_tensor(shape, dtype)
        in_cpu2, in_supa2 = create_random_tensor(shape, dtype)

        in_cpu_list = [in_cpu, in_cpu2]
        in_supa_list = [in_supa, in_supa2]

        scalar_cpu = torch.tensor(2.0, dtype=dtype).to(cpu_device)
        scalar_supa = torch.tensor(2.0, dtype=dtype).to(supa_device)

        scalar_cpu1 = torch.tensor(3.0, dtype=dtype).to(cpu_device)
        scalar_supa1 = torch.tensor(3.0, dtype=dtype).to(supa_device)
        res_cpu = torch._foreach_pow([scalar_cpu, scalar_cpu1], in_cpu_list)
        res_supa = torch._foreach_pow([scalar_supa, scalar_supa1], in_supa_list)

        assert_allclose(
            res_cpu[0], res_supa[0], atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
        assert_allclose(
            res_cpu[1], res_supa[1], atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
