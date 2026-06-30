# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose

FLOAT_RTOL, FLOAT_ATOL = 1e-5, 5e-5


class TestLinspace:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_linspace(self):
        out_cpu = torch.linspace(-10, 10, steps=5)
        out_supa = torch.linspace(-10, 10, steps=5, device="supa")
        assert_allclose(out_cpu, out_supa, atol=0, rtol=0)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_logspace(self):
        out_cpu = torch.logspace(-10, 10, steps=5)
        out_supa = torch.logspace(-10, 10, steps=5, device="supa")
        assert_allclose(out_cpu, out_supa, atol=FLOAT_RTOL, rtol=FLOAT_ATOL)


params = [
    [0, 52, 1],
    [0, 24, 1],
    [0, 10, 1],
    [0, 9, 1],
    [0, 32, 1.1],
]


br200_params = [
    pytest.param(
        [0, 52, 1],
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [0, 24, 1],
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [0, 10, 1],
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [0, 9, 1],
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [0, 32, 1.1],
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param([0, 512, 1], marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param([0, 2048, 1], marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param([0, 2047, 1], marks=[pytest.mark.gcuStress]),
    pytest.param([0, 65535, 1], marks=[pytest.mark.gcuStress]),
]

ATOL = {
    torch.float32: 5e-6,
    torch.bfloat16: 1e-5,
    torch.int32: 0,
    torch.int64: 0,
    torch.float16: 1e-5,
}
RTOL = {
    torch.float32: 1e-6,
    torch.bfloat16: 1.6e-2,
    torch.int32: 0,
    torch.int64: 0,
    torch.float16: 1e-3,
}

dtypes = [
    torch.float32,
    torch.int64,
    torch.int32,
    torch.bfloat16,
    torch.float16,
]


class TestArange:

    @pytest.mark.parametrize("target", br200_params)
    @pytest.mark.parametrize("datatype", dtypes)
    def test_arange_start_end(self, target, datatype):
        d_supa = torch.arange(
            target[0], target[1], device=torch.device("supa"), dtype=datatype
        )
        d_cpu = torch.arange(
            target[0], target[1], device=torch.device("cpu"), dtype=datatype
        )
        assert_allclose(
            d_cpu,
            d_supa.cpu(),
            atol=ATOL[datatype],
            rtol=RTOL[datatype],
            equal_nan=True,
        )

    @pytest.mark.parametrize("target", br200_params)
    def test_arange_end(self, target):
        d_supa = torch.arange(target[1], device=torch.device("supa"))
        d_cpu = torch.arange(target[1], device=torch.device("cpu"))
        assert_allclose(d_cpu, d_supa, rtol=1e-6, atol=5e-6, equal_nan=True)

    @pytest.mark.parametrize("target", br200_params)
    def test_arange_start_end_step(self, target):
        d_supa = torch.arange(
            target[0], target[1], target[2], device=torch.device("supa")
        )
        d_cpu = torch.arange(
            target[0], target[1], target[2], device=torch.device("cpu")
        )
        assert_allclose(d_cpu, d_supa, rtol=1e-6, atol=5e-6, equal_nan=True)


class TestRange:

    @pytest.mark.parametrize("target", br200_params)
    @pytest.mark.parametrize("datatype", dtypes)
    def test_range_start_end(self, target, datatype):
        # NOTE: "range_cuda" not implemented for 'BFloat16'
        if datatype != torch.bfloat16:
            d_supa = torch.range(
                target[0], target[1], device=torch.device("supa"), dtype=datatype
            )
            d_cpu = torch.range(
                target[0], target[1], device=torch.device("cpu"), dtype=datatype
            )
            assert_allclose(
                d_cpu,
                d_supa.cpu(),
                atol=ATOL[datatype],
                rtol=RTOL[datatype],
                equal_nan=True,
            )

    @pytest.mark.parametrize("target", br200_params)
    def test_range_start_end_step(self, target):
        d_supa = torch.range(
            target[0], target[1], target[2], device=torch.device("supa")
        )
        d_cpu = torch.range(target[0], target[1], target[2], device=torch.device("cpu"))
        assert_allclose(d_cpu, d_supa, rtol=1e-6, atol=5e-6, equal_nan=True)
