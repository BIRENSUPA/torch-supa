# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

br200_shapes = [
    pytest.param(
        (1, 3, 4, 4),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (16, 16, 16, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((15, 15, 15, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((17, 17, 17, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((21, 21, 21, 1022), marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.int8, torch.int32]
scalars = [1, 2, 3, 4, 5]


class TestBitwise:

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_bitwise_not(self, shape, dtype):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.bitwise_not(cpu_in)
        supa_out = torch.bitwise_not(supa_in)

        cpu_in.bitwise_not_()
        supa_in.bitwise_not_()
        assert_allclose(cpu_in, cpu_out, atol=5 * 1e-5, rtol=1 * 1e-5)
        assert_allclose(supa_in.cpu(), supa_out, atol=5 * 1e-5, rtol=1 * 1e-5)
        assert_allclose(cpu_out, supa_out, atol=5 * 1e-5, rtol=1 * 1e-5)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_bitwise_and(self, shape, dtype):
        cpu_in1, supa_in1 = create_random_tensor(shape, dtype=dtype)
        cpu_in2, supa_in2 = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.bitwise_and(cpu_in1, cpu_in2)
        supa_out = torch.bitwise_and(supa_in1, supa_in2)
        assert_allclose(cpu_out, supa_out, atol=5 * 1e-5, rtol=1 * 1e-5)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("scalar", scalars)
    def test_scalar_bitwise_and(self, shape, dtype, scalar):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_out, supa_out = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.bitwise_and(cpu_in, scalar, out=cpu_out)
        supa_out = torch.bitwise_and(supa_in, scalar, out=supa_out)
        assert_allclose(cpu_out, supa_out, atol=5 * 1e-5, rtol=1 * 1e-5)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_bitwise_or(self, shape, dtype):
        cpu_in1, supa_in1 = create_random_tensor(shape, dtype=dtype)
        cpu_in2, supa_in2 = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.bitwise_or(cpu_in1, cpu_in2)
        supa_out = torch.bitwise_or(supa_in1, supa_in2)
        assert_allclose(cpu_out, supa_out, atol=5 * 1e-5, rtol=1 * 1e-5)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("scalar", scalars)
    def test_scalar_bitwise_or(self, shape, dtype, scalar):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.bitwise_or(cpu_in, scalar)
        supa_out = torch.bitwise_or(supa_in, scalar)
        assert_allclose(cpu_out, supa_out, atol=5 * 1e-5, rtol=1 * 1e-5)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_bitwise_xor(self, shape, dtype):
        cpu_in1, supa_in1 = create_random_tensor(shape, dtype=dtype)
        cpu_in2, supa_in2 = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.bitwise_xor(cpu_in1, cpu_in2)
        supa_out = torch.bitwise_xor(supa_in1, supa_in2)
        assert_allclose(cpu_out, supa_out, atol=5 * 1e-5, rtol=1 * 1e-5)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("scalar", scalars)
    def test_scalar_bitwise_xor(self, shape, dtype, scalar):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.bitwise_xor(cpu_in, scalar)
        supa_out = torch.bitwise_xor(supa_in, scalar)
        assert_allclose(cpu_out, supa_out, atol=5 * 1e-5, rtol=1 * 1e-5)
