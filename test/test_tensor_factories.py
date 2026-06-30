# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose


class TestEye:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_eye_n(self):
        out_cpu = torch.eye(8, device="cpu")
        out_supa = torch.eye(8, device="supa")
        assert_allclose(out_cpu, out_supa, atol=0, rtol=0)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_eye_n_m(self):
        out_cpu = torch.eye(8, 4, device="cpu")
        out_supa = torch.eye(8, 4, device="supa")
        assert_allclose(out_cpu, out_supa, atol=0, rtol=0)


class TestEmpty:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_empty(self):
        torch.empty((2, 3), device="cpu")
        torch.empty((2, 3), device="supa")

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_empty_strided(self):
        cpu_x = torch.empty_strided((2, 3), (1, 2), device="cpu")
        supa_x = torch.empty_strided((2, 3), (1, 2), device="supa")
        assert cpu_x.stride() == supa_x.stride()
        assert cpu_x.size() == supa_x.size()
