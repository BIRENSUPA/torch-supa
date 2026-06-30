# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

shapes = [
    pytest.param(
        (2, 8),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((32, 155, 155), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]

dtypes = [
    torch.float32,
    torch.uint8,
    torch.int8,
    torch.int32,
    torch.float16,
    torch.bfloat16,
]
added_dtypes = [torch.int8]
values = [0.5, 1.0, 1000.0]


class TestFillMethod:
    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes + added_dtypes)
    def test_zeros(self, shape, dtype):
        zeros_cpu = torch.zeros(shape, dtype=dtype)
        zeros_supa = torch.zeros(shape, dtype=dtype, device=supa_device)

        assert_allclose(zeros_cpu, zeros_supa, atol=0, rtol=0)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes + added_dtypes)
    def test_zeroslike(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)

        zeros_cpu = torch.zeros_like(x_cpu)
        zeros_supa = torch.zeros_like(x_supa)

        assert_allclose(zeros_cpu, zeros_supa, atol=0, rtol=0)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_zero_(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)

        x_cpu.zero_()
        x_supa.zero_()

        assert_allclose(x_cpu, x_supa, atol=0, rtol=0)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_new_zeros(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor([1, 3, 224, 224], dtype=dtype)

        zeros_cpu = x_cpu.new_zeros(shape)
        zeros_supa = x_supa.new_zeros(shape)

        assert_allclose(zeros_cpu, zeros_supa, atol=0, rtol=0)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_zeros_out(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)

        torch.zeros(shape, out=x_cpu)
        torch.zeros(shape, out=x_supa)

        assert_allclose(x_cpu, x_supa, atol=0, rtol=0)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes + added_dtypes)
    def test_ones(self, shape, dtype):
        ones_cpu = torch.ones(shape, dtype=dtype)
        ones_supa = torch.ones(shape, dtype=dtype, device=supa_device)

        assert_allclose(ones_cpu, ones_supa, atol=0, rtol=0)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_oneslike(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)

        ones_cpu = torch.ones_like(x_cpu)
        ones_supa = torch.ones_like(x_supa)

        assert_allclose(ones_cpu, ones_supa, atol=0, rtol=0)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_ones_out(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        torch.ones(shape, out=x_cpu)
        torch.ones(shape, out=x_supa)

        assert_allclose(x_cpu, x_supa, atol=0, rtol=0)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("value", values)
    def test_full(self, shape, value):
        full_cpu = torch.full(shape, value, dtype=torch.float32)
        full_supa = torch.full(shape, value, dtype=torch.float32)

        assert_allclose(full_cpu, full_supa, atol=0, rtol=0)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("value", values)
    def test_fulllike(self, shape, value):
        x_cpu, x_supa = create_random_tensor(shape, dtype=torch.float32)

        full_cpu = torch.full_like(x_cpu, value, dtype=torch.float32)
        full_supa = torch.full_like(x_supa, value, dtype=torch.float32)

        assert_allclose(full_cpu, full_supa, atol=0, rtol=0)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("value", values)
    def test_new_full(self, shape, value):
        x_cpu, x_supa = create_random_tensor([1, 3, 224, 224], dtype=torch.float32)

        full_cpu = x_cpu.new_full(shape, value, dtype=torch.float32)
        full_supa = x_supa.new_full(shape, value, dtype=torch.float32)

        assert_allclose(full_cpu, full_supa, atol=0, rtol=0)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("value", values)
    def test_full_out(self, shape, value):
        x_cpu, x_supa = create_random_tensor(shape, dtype=torch.float32)

        torch.full(shape, value, out=x_cpu, dtype=torch.float32)
        torch.full(shape, value, out=x_supa, dtype=torch.float32)

        assert_allclose(x_cpu, x_supa, atol=0, rtol=0)
