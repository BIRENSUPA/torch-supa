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
    "_foreach_pow",
]

dtypes = [torch.float32]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


class TestForeachBinaryMethod:

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("func_type", func_types)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_foreach_binary_op_list(self, shape, func_type, dtype):
        pt_func = eval("torch." + func_type)

        in_cpu, in_supa = create_random_tensor(shape, dtype)
        in_cpu2, in_supa2 = create_random_tensor(shape, dtype)

        in_cpu_list = [in_cpu, in_cpu2]
        in_supa_list = [in_supa, in_supa2]

        res_cpu = pt_func(in_cpu_list, in_cpu_list)
        res_supa = pt_func(in_supa_list, in_supa_list)
        assert_allclose(
            res_cpu[0], res_supa[0], atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
        assert_allclose(
            res_cpu[1], res_supa[1], atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_clip_grad_norm_(self):
        max_norm = 10.0
        p1, p2, p3, p4, p5 = (
            torch.randn(64, 100),
            torch.randn(64, 100),
            torch.randn(64),
            torch.randn(32),
            torch.randn(3, 32, 3, 3),
        )
        p1_supa, p2_supa, p3_supa, p4_supa, p5_supa = (
            p1.to(supa_device),
            p2.to(supa_device),
            p3.to(supa_device),
            p4.to(supa_device),
            p5.to(supa_device),
        )

        g = torch.arange(1.0, 6401).view(64, 100)
        g1 = torch.arange(1.0, 65).view(64)
        g2 = torch.arange(1.0, 33).view(32)
        g3 = torch.arange(1.0, 865).view(3, 32, 3, 3)

        g_supa = g.to(supa_device)
        g1_supa = g1.to(supa_device)
        g2_supa = g2.to(supa_device)
        g3_supa = g3.to(supa_device)
        p1._grad = g.clone()
        p2._grad = g.clone()
        p3._grad = g1.clone()
        p4._grad = g2.clone()
        p5._grad = g3.clone()
        p1_supa._grad = g_supa.clone()
        p2_supa._grad = g_supa.clone()
        p3_supa._grad = g1_supa.clone()
        p4_supa._grad = g2_supa.clone()
        p5_supa._grad = g3_supa.clone()

        clip_norm_res = torch.nn.utils.clip_grad_norm_([p1, p2, p3, p4, p5], max_norm)
        clip_norm_res_supa = torch.nn.utils.clip_grad_norm_(
            [p1_supa, p2_supa, p3_supa, p4_supa, p5_supa], max_norm
        )
        assert_allclose(p1.grad, p1_supa.grad, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(p2.grad, p2_supa.grad, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(p3.grad, p3_supa.grad, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(p4.grad, p4_supa.grad, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(p5.grad, p5_supa.grad, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(
            clip_norm_res, clip_norm_res_supa, rtol=1e-5, atol=5e-5, equal_nan=True
        )
