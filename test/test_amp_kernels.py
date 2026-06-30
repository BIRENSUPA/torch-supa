# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch


cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


class TestAmp:
    def assertEqual(self, a, b):
        if a == b:
            return True
        else:
            return False

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_grad_scaling_update_scale(self):
        growth = 2.0
        backoff = 0.25
        growth_interval = 2
        scale = torch.full((1,), 4.0, dtype=torch.float32, device=supa_device)
        growth_tracker = torch.full((1,), 0.0, dtype=torch.int32, device=supa_device)
        found_inf = torch.full((1,), 0.0, dtype=torch.float, device=supa_device)

        # Simulates 2 consecutive unskipped iterations
        torch._amp_update_scale_(
            scale, growth_tracker, found_inf, growth, backoff, growth_interval
        )

        self.assertEqual(
            growth_tracker.cpu(),
            [
                1,
            ],
        )
        self.assertEqual(scale.cpu(), 4.0)
        torch._amp_update_scale_(
            scale, growth_tracker, found_inf, growth, backoff, growth_interval
        )
        self.assertEqual(
            growth_tracker.cpu(),
            [
                0,
            ],
        )
        self.assertEqual(
            scale.cpu(),
            [
                8.0,
            ],
        )

        # Simulates a skipped iteration
        found_inf.fill_(1.0)
        torch._amp_update_scale_(
            scale, growth_tracker, found_inf, growth, backoff, growth_interval
        )
        self.assertEqual(
            growth_tracker.cpu(),
            [
                0,
            ],
        )
        self.assertEqual(
            scale.cpu(),
            [
                2.0,
            ],
        )
