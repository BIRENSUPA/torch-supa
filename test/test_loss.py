# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import math

import pytest
import torch
import torch.nn as nn
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

params = [
    [(15, 5), (15), "mean"],
    [(16, 1000), (16), "mean"],
    [(16, 1000), (16), "sum"],
    [(16, 1000), (16), "none"],
    [(8732, 81), (8732), "sum"],
    [(8732, 81), (8732), "none"],
    [(1000, 200000), (1000), "mean"],
    [(200000, 1000), (200000), "mean"],
    [(200000, 1000), (200000), "sum"],
    [(200000, 1000), (200000), "none"],
    [(2, 19, 512, 1024), (2, 512, 1024), "none"],
    [(15, 5), (15), "mean"],
    [(16, 1000), (16), "mean"],
    [(16, 1000), (16), "sum"],
    [(16, 1000), (16), "none"],
    [(8732, 81), (8732), "sum"],
    [(8732, 81), (8732), "none"],
    [(1000, 200000), (1000), "mean"],
    [(200000, 1000), (200000), "mean"],
    [(200000, 1000), (200000), "sum"],
    [(200000, 1000), (200000), "none"],
    [(2, 19, 512, 1024), (2, 512, 1024), "none"],
]

nll_loss_br200_params = [
    pytest.param(
        (15, 5),
        (15),
        torch.float32,
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
        (512, 1024),
        (512),
        torch.float32,
        "none",
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (1023, 511), (1023), torch.float32, "none", marks=[pytest.mark.gcuStress]
    ),
    pytest.param(
        (1025, 513), (1025), torch.float32, "none", marks=[pytest.mark.gcuStress]
    ),
    pytest.param(
        (1028, 2135), (1028), torch.float32, "none", marks=[pytest.mark.gcuStress]
    ),
]

dtypes = [torch.float32, torch.bfloat16, torch.float16]


cross_entropy_loss_params = [
    [(1, 1000), (1), "mean"],
    [(16, 1000), (16), "mean"],
    [(16, 1000), (16), "sum"],
    [(16, 1000), (16), "none"],
    [(16, 1000), (16), "none"],
]
cross_entropy_br200_params = [
    [(1, 6), (1), torch.float32, "none"],
]

cross_entropy_br200_params = [
    pytest.param(
        (1, 6),
        (1),
        torch.float32,
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
        (512, 1024),
        (512),
        torch.float32,
        "none",
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (512, 1024),
        (512),
        torch.float32,
        "sum",
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (512, 1024),
        (512),
        torch.float32,
        "mean",
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (512, 1027), (512), torch.float32, "none", marks=[pytest.mark.gcuStress]
    ),
    pytest.param(
        (512, 1027), (512), torch.float32, "sum", marks=[pytest.mark.gcuStress]
    ),
    pytest.param(
        (512, 1027), (512), torch.float32, "mean", marks=[pytest.mark.gcuStress]
    ),
]

RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


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


class TestLoss:
    @pytest.mark.parametrize("shape, target_shape, reduce", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_nll_loss(self, shape, target_shape, dtype, reduce):
        C = shape[1]
        x_cpu = torch.randn(shape, requires_grad=True)
        target_cpu = torch.empty(target_shape, dtype=torch.long).random_(C)
        loss = nn.NLLLoss(reduction=reduce)
        y_cpu = loss(x_cpu, target_cpu)

        x_supa = x_cpu.to(supa_device)
        target_supa = target_cpu.to(supa_device)
        loss_supa = loss.to(supa_device)
        y_supa = loss_supa(x_supa, target_supa)
        atol, rtol = GetAccumulationRtolAtol(shape[0], dtype)
        if reduce == "none":
            atol = 1e-2
            rtol = 1.3e-3
        assert_allclose(y_cpu, y_supa.cpu(), atol=atol, rtol=rtol)
        # Ensure target not changed after loss
        assert_allclose(target_cpu, target_supa.cpu(), atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("shape, target_shape, reduce", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_nll_loss_bwd(self, shape, target_shape, dtype, reduce):
        C = shape[1]
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)
        target_cpu = torch.empty(target_shape, dtype=torch.long).random_(C)
        loss = nn.NLLLoss(reduction=reduce).float()
        y_cpu = loss(x_cpu.float(), target_cpu)
        g_cpu = torch.ones_like(y_cpu, dtype=dtype, device=cpu_device)
        y_cpu.backward(g_cpu.float())
        cpu_grad = x_cpu.grad.clone()

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
        assert_allclose(cpu_grad.to(dtype), supa_grad, atol=atol, rtol=rtol)
        # Ensure target not changed after loss
        assert_allclose(target_cpu, target_supa.cpu(), atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize(
        "shape, target_shape, dtype, reduce", nll_loss_br200_params
    )
    def test_nll_loss_br200(self, shape, target_shape, dtype, reduce):
        self.test_nll_loss_bwd(shape, target_shape, dtype, reduce)

    @pytest.mark.parametrize("shape, target_shape, reduce", params)
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
        cpu_grad = x_cpu.grad.clone()

        target_supa = target_cpu.to(supa_device)
        loss_supa = loss.to(dtype).to(supa_device)
        y_supa = loss_supa(x_supa, target_supa)
        g_supa = g_cpu.to(dtype).to(supa_device)
        y_supa.backward(g_supa)
        supa_grad = x_supa.grad.cpu()
        atol, rtol = GetAccumulationRtolAtol(shape[0], dtype)
        if reduce == "none":
            atol = 1e-5
            rtol = 1.3e-6
        assert_allclose(cpu_grad.to(dtype), supa_grad, atol=atol, rtol=rtol)
        assert_allclose(target_cpu, target_supa.cpu(), atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize(
        "shape, target_shape, dtype, reduce", nll_loss_br200_params
    )
    def test_nll_loss_bwd_weight_br200(self, shape, target_shape, dtype, reduce):
        self.test_nll_loss_bwd_weight(shape, target_shape, dtype, reduce)

    @pytest.mark.parametrize("shape, target_shape, reduce", cross_entropy_loss_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_cross_entropy_loss(self, shape, target_shape, dtype, reduce):
        x_cpu = torch.randn(shape, requires_grad=True)
        target_cpu = torch.empty(target_shape, dtype=torch.long).random_(shape[1])
        loss = nn.CrossEntropyLoss(reduction=reduce)
        y_cpu = loss(x_cpu, target_cpu)

        x_supa = x_cpu.to(supa_device)
        target_supa = target_cpu.to(supa_device)
        loss_supa = loss.to(supa_device)
        y_supa = loss_supa(x_supa, target_supa)

        assert_allclose(y_cpu, y_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])

    @pytest.mark.parametrize(
        "shape, target_shape, dtype, reduce", cross_entropy_br200_params
    )
    def test_cross_entropy_loss_br200(self, shape, target_shape, dtype, reduce):
        self.test_cross_entropy_loss(shape, target_shape, dtype, reduce)

    @pytest.mark.parametrize("shape, target_shape, reduce", cross_entropy_loss_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_cross_entropy_loss_bwd(self, shape, target_shape, dtype, reduce):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)
        target_cpu = torch.empty(target_shape, dtype=torch.long).random_(shape[1])
        loss = nn.CrossEntropyLoss(reduction=reduce).float()
        y_cpu = loss(x_cpu.float(), target_cpu)
        g_cpu = torch.ones_like(y_cpu, dtype=dtype, device=cpu_device)
        y_cpu.backward(g_cpu.float())
        cpu_grad = x_cpu.grad.clone()

        target_supa = target_cpu.to(supa_device)
        loss_supa = loss.to(dtype).to(supa_device)
        y_supa = loss_supa(x_supa, target_supa)
        g_supa = g_cpu.to(supa_device)
        y_supa.backward(g_supa)
        supa_grad = x_supa.grad.cpu()

        assert_allclose(
            cpu_grad.to(dtype), supa_grad, atol=ATOL[dtype], rtol=RTOL[dtype]
        )

    @pytest.mark.parametrize(
        "shape, target_shape, dtype, reduce", cross_entropy_br200_params
    )
    def test_cross_entropy_loss_bwd_br200(self, shape, target_shape, dtype, reduce):
        self.test_cross_entropy_loss_bwd(shape, target_shape, dtype, reduce)
