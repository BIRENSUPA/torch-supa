# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
import torch.nn as nn
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


br200_params = [
    pytest.param(
        (2, 6),
        torch.float32,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (512, 1024), torch.float32, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((1023, 511), torch.float32, marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), torch.float32, marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), torch.float32, marks=[pytest.mark.gcuStress]),
]

RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


class TestMultiMarginLoss:
    @pytest.mark.parametrize("shape, dtype", br200_params)
    def test_multi_label_margin_loss_fwd(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=False)
        target_cpu = torch.empty(shape, dtype=torch.long).random_(shape[1])
        loss = nn.MultiLabelMarginLoss()
        y_cpu = loss(x_cpu.float(), target_cpu)
        target_supa = target_cpu.to(supa_device)
        loss_supa = loss.to(supa_device)
        y_supa = loss_supa(x_supa, target_supa)
        assert_allclose(y_cpu.to(dtype), y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])

    @pytest.mark.parametrize("shape, dtype", br200_params)
    def test_multi_label_margin_loss_bwd(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)
        target_cpu = torch.empty(shape, dtype=torch.long).random_(shape[1])
        loss = nn.MultiLabelMarginLoss()
        y_cpu = loss(x_cpu.float(), target_cpu)
        g_cpu = torch.ones_like(y_cpu, dtype=dtype, device=cpu_device)
        y_cpu.backward(g_cpu.float())
        cpu_grad = x_cpu.grad.clone()

        target_supa = target_cpu.to(supa_device)
        loss_supa = loss.to(supa_device)
        y_supa = loss_supa(x_supa, target_supa)
        g_supa = g_cpu.to(supa_device)
        y_supa.backward(g_supa)
        supa_grad = x_supa.grad.cpu()

        assert_allclose(cpu_grad.to(dtype), supa_grad, atol=9e-4, rtol=RTOL[dtype])
