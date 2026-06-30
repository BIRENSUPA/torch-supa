# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

tensor_shape = [
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
    pytest.param(
        (50,),
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

br200_shape = [
    pytest.param(
        (16,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
]

dtypes = [torch.float32, torch.bfloat16, torch.float16]


class TestSingleMethod:

    @pytest.mark.parametrize("shape", tensor_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_ceil(self, dtype, shape):
        in_cpu, in_supa = create_random_tensor(shape, dtype)

        out_cpu = torch.ceil(in_cpu)
        out_supa = torch.ceil(in_supa)
        in_cpu.ceil_()
        in_supa.ceil_()

        assert_allclose(in_cpu, in_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(in_supa.cpu(), out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_ceil_br200(self, dtype, shape):
        self.test_ceil(dtype, shape)

    @pytest.mark.parametrize("shape", tensor_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_frac(self, dtype, shape):
        in_cpu, in_supa = create_random_tensor(shape, dtype)

        out_cpu = torch.frac(in_cpu)
        out_supa = torch.frac(in_supa)
        in_cpu.frac_()
        in_supa.frac_()

        assert_allclose(in_cpu, in_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(in_supa.cpu(), out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_frac_br200(self, dtype, shape):
        self.test_frac(dtype, shape)

    @pytest.mark.parametrize("shape", tensor_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_floor(self, dtype, shape):
        in_cpu, in_supa = create_random_tensor(shape, dtype)

        out_cpu = torch.floor(in_cpu)
        out_supa = torch.floor(in_supa)
        in_cpu.floor_()
        in_supa.floor_()

        assert_allclose(in_cpu, in_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(in_supa.cpu(), out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_floor_br200(self, dtype, shape):
        self.test_floor(dtype, shape)

    @pytest.mark.parametrize("shape", tensor_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_reciprocal(self, dtype, shape):
        if dtype == torch.bfloat16:
            return
        in_cpu, in_supa = create_random_tensor(shape, dtype)

        out_cpu = torch.reciprocal(in_cpu)
        out_supa = torch.reciprocal(in_supa)
        in_cpu.reciprocal_()
        in_supa.reciprocal_()

        assert_allclose(in_cpu, in_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(in_supa.cpu(), out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_reciprocal_br200(self, dtype, shape):
        self.test_reciprocal(dtype, shape)

    @pytest.mark.parametrize("shape", tensor_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_round(self, dtype, shape):
        in_cpu, in_supa = create_random_tensor(shape, dtype)

        out_cpu = torch.round(in_cpu)
        out_supa = torch.round(in_supa)
        in_cpu.round_()
        in_supa.round_()

        assert_allclose(in_cpu, in_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(in_supa.cpu(), out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_round_br200(self, dtype, shape):
        self.test_round(dtype, shape)

    @pytest.mark.parametrize("shape", tensor_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_trunc(self, dtype, shape):
        in_cpu, in_supa = create_random_tensor(shape, dtype)

        out_cpu = torch.trunc(in_cpu)
        out_supa = torch.trunc(in_supa)
        in_cpu.trunc_()
        in_supa.trunc_()

        assert_allclose(in_cpu, in_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(in_supa.cpu(), out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_trunc_br200(self, dtype, shape):
        self.test_trunc(dtype, shape)
