# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
import torch.nn as nn

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

params = [
    pytest.param(
        (2, 5, 13),
        (5, 3),
        torch.float32,
        0,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (24, 12, 60),
        (12, 24),
        torch.float32,
        0,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (19, 82, 42), (82, 20), torch.float32, 0, marks=[pytest.mark.gcuStress]
    ),
    pytest.param(
        (101, 97, 4233), (97, 19), torch.float32, 0, marks=[pytest.mark.gcuStress]
    ),
    pytest.param(
        (136, 72, 4233), (72, 24), torch.float32, 0, marks=[pytest.mark.gcuStress]
    ),
]
modes = [
    "none",
    "sum",
    "mean",
]
"""
1. none mode output is 1d vector, need to check partial oob zero
2. sum and reduce mode output is one point, reduce op should cover these scenes
"""

use_cudnn = False


class TestCTCLoss:
    @pytest.mark.parametrize("shape0, shape1,  dtype0, blank", params)
    @pytest.mark.parametrize("mode", modes)
    def test_ctc_loss_targets_1dv(self, shape0, shape1, dtype0, blank, mode):
        with torch.backends.cudnn.flags(enabled=use_cudnn):
            T = shape0[0]
            N = shape0[1]
            C = shape0[2]
            S = shape1[1]
            x_cpu, _ = create_random_tensor(shape0, dtype=dtype0)
            log_probs_cpu = x_cpu.log_softmax(2).detach().requires_grad_()
            log_probs_supa = log_probs_cpu.to(supa_device).detach().requires_grad_()
            input_lengths_cpu = torch.randint(
                low=T // 2 + 1, high=T + 1, size=(N,), dtype=torch.int
            )
            input_lengths_supa = input_lengths_cpu.to(supa_device)
            target_lengths_cpu = torch.randint(
                low=1, high=S, size=(N,), dtype=torch.int
            )
            target_lengths_supa = target_lengths_cpu.to(supa_device)
            targets_cpu = torch.randint(
                low=1, high=C, size=(sum(target_lengths_cpu),), dtype=torch.int
            )
            targets_supa = targets_cpu.to(supa_device)
            loss_cpu = nn.functional.ctc_loss(
                log_probs_cpu,
                targets_cpu,
                input_lengths_cpu,
                target_lengths_cpu,
                blank,
                mode,
            )

            loss_supa = nn.functional.ctc_loss(
                log_probs_supa,
                targets_supa,
                input_lengths_supa,
                target_lengths_supa,
                blank,
                mode,
            )

            assert_allclose(loss_cpu, loss_supa, rtol=1e-4, atol=5e-4)
