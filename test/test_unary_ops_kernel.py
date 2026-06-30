# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

shapes = [
    pytest.param(
        (2, 3, 4),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
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
        (1, 3, 4, 4),
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

params = [
    pytest.param(
        (16,),
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

dtypes = [torch.int32]
scalars = [1, 2, 3, 4, 5]


class TestBitwise:

    @pytest.mark.parametrize("shape", shapes)
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

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_bitwise_and(self, shape, dtype):
        cpu_in1, supa_in1 = create_random_tensor(shape, dtype=dtype)
        cpu_in2, supa_in2 = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.bitwise_and(cpu_in1, cpu_in2)
        supa_out = torch.bitwise_and(supa_in1, supa_in2)
        assert_allclose(cpu_out, supa_out, atol=5 * 1e-5, rtol=1 * 1e-5)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("scalar", scalars)
    def test_scalar_bitwise_and(self, shape, dtype, scalar):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_out, supa_out = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.bitwise_and(cpu_in, scalar, out=cpu_out)
        supa_out = torch.bitwise_and(supa_in, scalar, out=supa_out)
        assert_allclose(cpu_out, supa_out, atol=5 * 1e-5, rtol=1 * 1e-5)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_bitwise_or(self, shape, dtype):
        cpu_in1, supa_in1 = create_random_tensor(shape, dtype=dtype)
        cpu_in2, supa_in2 = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.bitwise_or(cpu_in1, cpu_in2)
        supa_out = torch.bitwise_or(supa_in1, supa_in2)
        assert_allclose(cpu_out, supa_out, atol=5 * 1e-5, rtol=1 * 1e-5)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("scalar", scalars)
    def test_scalar_bitwise_or(self, shape, dtype, scalar):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.bitwise_or(cpu_in, scalar)
        supa_out = torch.bitwise_or(supa_in, scalar)
        assert_allclose(cpu_out, supa_out, atol=5 * 1e-5, rtol=1 * 1e-5)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_bitwise_xor(self, shape, dtype):
        cpu_in1, supa_in1 = create_random_tensor(shape, dtype=dtype)
        cpu_in2, supa_in2 = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.bitwise_xor(cpu_in1, cpu_in2)
        supa_out = torch.bitwise_xor(supa_in1, supa_in2)
        assert_allclose(cpu_out, supa_out, atol=5 * 1e-5, rtol=1 * 1e-5)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("scalar", scalars)
    def test_scalar_bitwise_xor(self, shape, dtype, scalar):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.bitwise_xor(cpu_in, scalar)
        supa_out = torch.bitwise_xor(supa_in, scalar)
        assert_allclose(cpu_out, supa_out, atol=5 * 1e-5, rtol=1 * 1e-5)


dtypes = [torch.float32, torch.bfloat16]


class TestSingleMethod:

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_exp_ptwise(self, shape, dtype):
        in_cpu, in_supa = create_random_tensor(shape, dtype)

        out_cpu = torch.exp(in_cpu)
        out_supa = torch.exp(in_supa)
        in_cpu.exp_()
        in_supa.exp_()
        if dtype == torch.bfloat16:
            return
        assert_allclose(in_cpu, in_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(in_supa.cpu(), out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_expm1_ptwise(self, shape, dtype):
        in_cpu, in_supa = create_random_tensor(shape, dtype)

        out_cpu = torch.expm1(in_cpu)
        out_supa = torch.expm1(in_supa)
        in_cpu.expm1_()
        in_supa.expm1_()

        assert_allclose(in_cpu, in_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(in_supa.cpu(), out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_exp2_ptwise(self, shape, dtype):
        in_cpu, in_supa = create_random_tensor(shape, dtype)

        out_cpu = torch.exp2(in_cpu)
        out_supa = torch.exp2(in_supa)
        in_cpu.exp2_()
        in_supa.exp2_()

        assert_allclose(in_cpu, in_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(in_supa.cpu(), out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)


RTOL = {torch.float32: 1e-5, torch.bfloat16: 0.016}
ATOL = {torch.float32: 5e-5, torch.bfloat16: 1e-3}


class TestRsqrt:

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_rsqrt(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        y_cpu = torch.rsqrt(x_cpu)
        y_supa = torch.rsqrt(x_supa)
        x_cpu.rsqrt_()
        x_supa.rsqrt_()

        if dtype == torch.bfloat16:
            return

        assert_allclose(
            y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
        assert_allclose(
            x_cpu, x_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
        assert_allclose(
            x_supa.cpu(), y_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )


class TestSqrt:

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_sqrt(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        y_cpu = torch.sqrt(x_cpu)
        y_supa = torch.sqrt(x_supa)
        x_cpu.sqrt_()
        x_supa.sqrt_()

        if dtype == torch.bfloat16:
            return

        assert_allclose(
            y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
        assert_allclose(
            x_cpu, x_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
        assert_allclose(
            x_supa.cpu(), y_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )


class TestNanToNum:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_nan_to_num(self):
        x_cpu = torch.tensor([float("nan"), float("inf"), -float("inf"), 3.14])
        x_supa = torch.tensor([float("nan"), float("inf"), -float("inf"), 3.14]).to(
            "supa"
        )
        out_cpu = torch.nan_to_num(x_cpu)
        out_supa = torch.nan_to_num(x_supa)

        assert_allclose(out_cpu, out_supa, rtol=0, atol=0, equal_nan=True)


class TestFrexp:
    @pytest.mark.parametrize("shape, dtype", params)
    def test_frexp(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)

        mantissa_cpu, exponent_cpu = torch.frexp(x_cpu)
        mantissa_supa, exponent_supa = torch.frexp(x_supa)

        assert_allclose(mantissa_cpu, mantissa_supa, rtol=1e-5, atol=5e-5)
        assert_allclose(exponent_cpu, exponent_supa, rtol=1e-5, atol=5e-5)
