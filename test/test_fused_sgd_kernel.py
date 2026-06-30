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

shapes = [
    pytest.param(
        (1, 1000),
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

dtypes = [torch.float32, torch.bfloat16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


params = [
    #  optim_step,lr,  momentum, weight_decay, dampening, nesterov
    [1, 0.1, 0, 0, 0, False],
]


class TestSgdFusedOp:
    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize(
        "optim_step, lr, momentum, weight_decay, dampening, nesterov", params
    )
    def test_fused_sgd(
        self, optim_step, shape, dtype, lr, momentum, weight_decay, dampening, nesterov
    ):
        w_cpu, w_supa = create_random_tensor(
            shape,
            dtype=dtype,
            min_value=-1,
            max_value=1,
            requires_grad=False,
            mode=RandomMode.uniform,
        )
        target_cpu = torch.empty(shape[0], dtype=torch.long).random_(shape[1])
        target_supa = target_cpu.to(supa_device)

        criterion_cpu = torch.nn.CrossEntropyLoss()
        criterion_supa = criterion_cpu.to(supa_device)
        w_cpu.requires_grad = True
        w_supa.requires_grad = True

        optimizer_cpu = torch.optim.SGD(
            params=[w_cpu],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
            fused=False,
        )
        optimizer_supa = torch.optim.SGD(
            params=[w_supa],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
            fused=True,
        )

        for _ in range(optim_step):
            loss_cpu = criterion_cpu(w_cpu, target_cpu)
            loss_cpu.backward()
            optimizer_cpu.step()
            w_cpu.grad.detach_()
            w_cpu.grad.zero_()

            loss_supa = criterion_supa(w_supa, target_supa)
            loss_supa.backward()
            optimizer_supa.step()
            supa_out_cpu = w_supa.cpu()
            w_supa.grad.detach_()
            w_supa.grad.zero_()
            assert_allclose(w_cpu, supa_out_cpu, rtol=1e-3, atol=1e-4)
