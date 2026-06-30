# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    create_torch_tensor_from_np,
)

params = [
    pytest.param(
        (2, 16),
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

dtypes = [
    torch.float32,
    torch.bfloat16,
    torch.float16,
]
supa_device = torch.device("supa")


class TestWhere:

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_where(self, shape, dtype):
        RTOL = 1e-5
        ATOL = 5e-5

        input_cpu, input_supa = create_random_tensor(shape, dtype, requires_grad=True)
        input2_cpu, input2_supa = create_random_tensor(shape, dtype, requires_grad=True)
        cpu_grad, supa_grad = create_random_tensor(shape, dtype=dtype)

        output_cpu = torch.where(input_cpu > 0, input_cpu, input2_cpu)
        output_supa = torch.where(input_supa > 0, input_supa, input2_supa)

        output_cpu.backward(cpu_grad)
        output_supa.backward(supa_grad)

        cpu_res = input_cpu.grad
        supa_res = input_supa.grad

        assert_allclose(output_cpu, output_supa, atol=ATOL, rtol=RTOL)
        assert_allclose(cpu_res, supa_res, atol=ATOL, rtol=RTOL)


params_scalar = [
    [-1.42],
    [-0.42],
    [0.42],
    [1.42],
]


class TestClamp:

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_clamp_function(self, shape, dtype):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.clamp(cpu_in, 0, 1)
        supa_out = torch.clamp(supa_in, 0, 1)
        assert_allclose(cpu_out, supa_out, rtol=0, atol=0)

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_clamp_method(self, shape, dtype):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_out = cpu_in.clamp(0, 1)
        supa_out = supa_in.clamp(0, 1)
        assert_allclose(cpu_out, supa_out, rtol=0, atol=0)

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_clamp_max_none(self, shape, dtype):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_out = cpu_in.clamp(min=0.0)
        supa_out = supa_in.clamp(min=0.0)
        assert_allclose(cpu_out, supa_out, rtol=0, atol=0)

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_clamp_min_none(self, shape, dtype):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_out = cpu_in.clamp(max=1.0)
        supa_out = supa_in.clamp(max=1.0)
        assert_allclose(cpu_out, supa_out, rtol=0, atol=0)

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_clamp_min_function(self, shape, dtype):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.clamp_min(cpu_in, 1)
        supa_out = torch.clamp_min(supa_in, 1)
        assert_allclose(cpu_out, supa_out, rtol=0, atol=0)

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_clamp_max_function(self, shape, dtype):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_out = torch.clamp_max(cpu_in, 1)
        supa_out = torch.clamp_max(supa_in, 1)
        assert_allclose(cpu_out, supa_out, rtol=0, atol=0)

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_clamp_min_out(self, shape, dtype):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_out, supa_out = create_random_tensor(shape, dtype=dtype)
        torch.clamp_min(cpu_in, 1, out=cpu_out)
        torch.clamp_min(supa_in, 1, out=supa_out)
        assert_allclose(cpu_out, supa_out, rtol=0, atol=0)

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_clamp_max_out(self, shape, dtype):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_out, supa_out = create_random_tensor(shape, dtype=dtype)
        torch.clamp_max(cpu_in, 1, out=cpu_out)
        torch.clamp_max(supa_in, 1, out=supa_out)
        assert_allclose(cpu_out, supa_out, rtol=0, atol=0)

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_clamp_min_inplace(self, shape, dtype):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_in.clamp_min_(1)
        supa_in.clamp_min_(1)
        assert_allclose(cpu_in, supa_in, rtol=0, atol=0)

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_clamp_max_inplace(self, shape, dtype):
        cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
        cpu_in.clamp_max_(1)
        supa_in.clamp_max_(1)
        assert_allclose(cpu_in, supa_in, rtol=0, atol=0)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize("val", params_scalar)
    def test_clamp_scalar(self, val):
        cpu_in, supa_in = create_torch_tensor_from_np(val[0])
        cpu_out = torch.clamp(cpu_in, 0, 1)
        supa_out = torch.clamp(supa_in, 0, 1)
        assert_allclose(cpu_out, supa_out, rtol=0, atol=0)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize("val", params_scalar)
    def test_clamp_min_none_scalar(self, val):
        cpu_in, supa_in = create_torch_tensor_from_np(val[0])
        cpu_out = torch.clamp(cpu_in, max=1)
        supa_out = torch.clamp(supa_in, max=1)
        assert_allclose(cpu_out, supa_out, rtol=0, atol=0)


class TestIsPosinf:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_isposinf(self):
        cpu_x = torch.tensor([-float("inf"), float("inf"), 1.2])
        supa_x = torch.tensor([-float("inf"), float("inf"), 1.2]).supa()

        out_cpu = torch.isposinf(cpu_x)
        out_supa = torch.isposinf(supa_x)

        assert_allclose(out_cpu, out_supa, atol=0, rtol=0)


class TestIsNotPosinf:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_isposinf(self):
        cpu_x = torch.tensor([-float("inf"), float("inf"), 1.2])
        supa_x = torch.tensor([-float("inf"), float("inf"), 1.2]).supa()

        out_cpu = torch.isposinf(cpu_x)
        out_supa = torch.isposinf(supa_x)

        assert_allclose(out_cpu, out_supa, atol=0, rtol=0)


class TestIsNeginf:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_isneginf(self):
        cpu_x = torch.tensor([-float("inf"), float("inf"), 1.2])
        supa_x = torch.tensor([-float("inf"), float("inf"), 1.2]).supa()

        out_cpu = torch.isneginf(cpu_x)
        out_supa = torch.isneginf(supa_x)

        assert_allclose(out_cpu, out_supa, atol=0, rtol=0)
