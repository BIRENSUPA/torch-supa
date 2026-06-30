# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import math

import pytest
import torch

import torch.nn as nn
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

br200_params = [
    pytest.param(
        (15, 5),
        (15),
        "none",
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (512, 1024), (512), "none", marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((1023, 511), (1023), "none", marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), (1025), "none", marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), (1028), "none", marks=[pytest.mark.gcuStress]),
]


dtypes = [torch.float32, torch.bfloat16, torch.float16]


def GetAccumulationRtolAtol(accumulation_cnt, dtype):
    alpha = 4.0
    if accumulation_cnt <= 32:
        k1 = alpha
    else:
        k1 = alpha * (math.log2(accumulation_cnt) - 4.0)

    cur_check_atol = k1 * 1e-5
    cur_check_rtol = k1 * 1.3e-6
    if dtype == torch.bfloat16:
        cur_check_atol = 0.001
        cur_check_rtol = 0.016
    return cur_check_atol, cur_check_rtol


class TestNllLoss:

    @pytest.mark.parametrize("shape, target_shape, reduce", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_nll_loss_br200(self, shape, target_shape, dtype, reduce):
        def test_nll_loss(shape, target_shape, dtype, reduce):
            C = shape[1]
            x_cpu = torch.randn(shape, requires_grad=True)
            target_cpu = torch.empty(target_shape, dtype=torch.long).random_(C)
            loss = nn.NLLLoss(reduction=reduce)
            y_cpu = loss(x_cpu.float(), target_cpu)

            x_supa = x_cpu.to(dtype).to(supa_device)
            target_supa = target_cpu.to(supa_device)
            loss_supa = loss.to(supa_device)
            y_supa = loss_supa(x_supa, target_supa)
            atol, rtol = GetAccumulationRtolAtol(shape[0], dtype)
            if reduce == "none":
                atol = 1e-2
                rtol = 1.3e-3
            assert_allclose(y_cpu.to(dtype), y_supa.cpu(), atol=atol, rtol=rtol)
            # Ensure target not changed after loss
            assert_allclose(target_cpu, target_supa.cpu(), atol=1e-10, rtol=1e-10)

        def test_nll_loss_bwd(shape, target_shape, dtype, reduce):
            C = shape[1]
            x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)
            target_cpu = torch.empty(target_shape, dtype=torch.long).random_(C)
            loss = nn.NLLLoss(reduction=reduce)
            y_cpu = loss(x_cpu.float(), target_cpu)
            g_cpu = torch.ones_like(y_cpu, dtype=dtype, device=cpu_device)
            y_cpu.backward(g_cpu.float())
            cpu_grad = x_cpu.grad.clone().to(dtype)

            target_supa = target_cpu.to(supa_device)
            loss_supa = loss.to(supa_device)
            y_supa = loss_supa(x_supa, target_supa)
            g_supa = g_cpu.to(supa_device)
            y_supa.backward(g_supa)
            supa_grad = x_supa.grad.cpu()
            atol, rtol = GetAccumulationRtolAtol(shape[0], dtype)
            if reduce == "none":
                atol = 1e-5
                rtol = 1.3e-6
            assert_allclose(cpu_grad, supa_grad, atol=atol, rtol=rtol)
            # Ensure target not changed after loss
            assert_allclose(target_cpu, target_supa.cpu(), atol=1e-10, rtol=1e-10)

        test_nll_loss(shape, target_shape, dtype, reduce)
        test_nll_loss_bwd(shape, target_shape, dtype, reduce)

    @pytest.mark.parametrize("shape, target_shape, reduce", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_nll_loss_bwd_weight(self, shape, target_shape, dtype, reduce):
        C = shape[1]
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)
        target_cpu = torch.empty(target_shape, dtype=torch.long).random_(C)
        w = torch.ones(C, dtype=dtype)
        loss = nn.NLLLoss(weight=w, reduction=reduce).float()
        y_cpu = loss(x_cpu.float(), target_cpu)
        g_cpu = torch.ones_like(y_cpu, dtype=dtype, device=cpu_device)
        y_cpu.backward(g_cpu.float())
        cpu_grad = x_cpu.grad.clone().to(dtype)

        target_supa = target_cpu.to(supa_device)
        loss_supa = loss.to(dtype).to(supa_device)
        y_supa = loss_supa(x_supa, target_supa)
        g_supa = g_cpu.to(supa_device)
        y_supa.backward(g_supa)
        supa_grad = x_supa.grad.cpu()
        atol, rtol = GetAccumulationRtolAtol(shape[0], dtype)
        if reduce == "none":
            atol = 1e-5
            rtol = 1.3e-6
        assert_allclose(cpu_grad, supa_grad, atol=atol, rtol=rtol)
        assert_allclose(target_cpu, target_supa.cpu(), atol=1e-10, rtol=1e-10)
