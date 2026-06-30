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
        (1, 100),
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

dtypes = [torch.float32]
lrs = [1e-2]
weight_decays = [0.01]


class TestAdamFusedOp:

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("lr", lrs)
    @pytest.mark.parametrize("weight_decay", weight_decays)
    def test_adam(self, shape, dtype, lr, weight_decay):
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

        optimizer_golden = torch.optim.Adam(
            params=[w_cpu], lr=lr, weight_decay=weight_decay, fused=False
        )
        optimizer_supa = torch.optim.Adam(
            params=[w_supa], lr=lr, weight_decay=weight_decay, fused=True
        )
        for _ in range(5):
            loss_cpu = criterion_cpu(w_cpu, target_cpu)
            loss_cpu.backward()
            optimizer_golden.step()
            w_cpu.grad.detach_()
            w_cpu.grad.zero_()

            loss_supa = criterion_supa(w_supa, target_supa)
            loss_supa.backward()
            optimizer_supa.step()
            w_supa.grad.detach_()
            w_supa.grad.zero_()
            assert_allclose(w_cpu, w_supa, rtol=1e-2, atol=1e-3, equal_nan=True)
