# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
import torch.nn as nn

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

params_1d = [
    pytest.param(
        (2, 4, 5),
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
        (4, 5),
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
        (2, 512, 1024),
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (512, 1024), torch.float32, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((1023, 511), torch.float32, marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), torch.float32, marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), torch.float32, marks=[pytest.mark.gcuStress]),
]
pad_1d = [
    (0, 1),
    (1, 0),
]

params_2d = [
    pytest.param(
        (4, 2, 5, 5),
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
        (3, 4, 5),
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
        (4, 2, 512, 1024),
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (3, 512, 1024),
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param((4, 2, 1023, 511), torch.float32, marks=[pytest.mark.gcuStress]),
    pytest.param((4, 2, 1025, 513), torch.float32, marks=[pytest.mark.gcuStress]),
    pytest.param((4, 2, 1028, 2135), torch.float32, marks=[pytest.mark.gcuStress]),
]

params_3d = [
    pytest.param(
        (4, 2, 5, 5, 2),
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
        (3, 4, 5, 2),
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
        (4, 2, 16, 256, 512),
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (3, 4, 32, 512),
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param((4, 2, 1023, 511), torch.float32, marks=[pytest.mark.gcuStress]),
    pytest.param((4, 2, 1025, 513), torch.float32, marks=[pytest.mark.gcuStress]),
    pytest.param((4, 2, 1028, 2135), torch.float32, marks=[pytest.mark.gcuStress]),
]

pad_2d = [
    (1, 1, 2, 0),
    (1, 1, 1, 1),
]

pad_3d = [
    (1, 1, 1, 1, 1, 1),
]


class TestReflectionPad1d:
    @pytest.mark.parametrize("shape, dtype", params_1d)
    @pytest.mark.parametrize("pad", pad_1d)
    def test_reflectionpad1d(self, shape, dtype, pad):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)

        y_cpu = nn.functional.pad(x_cpu, pad, mode="reflect")
        y_supa = nn.functional.pad(x_supa, pad, mode="reflect")

        cpu_grad, supa_grad = create_random_tensor(
            y_cpu.shape, dtype=dtype, requires_grad=False
        )
        y_cpu.backward(cpu_grad)
        x_cpu_grad = x_cpu.grad

        y_supa.backward(supa_grad)
        x_supa_grad = x_supa.grad

        assert_allclose(y_cpu, y_supa, rtol=0, atol=0)
        assert_allclose(x_cpu_grad, x_supa_grad, rtol=1e-5, atol=5e-5)


class TestReflectionPad2d:
    @pytest.mark.parametrize("shape, dtype", params_2d)
    @pytest.mark.parametrize("pad", pad_2d)
    def test_reflectionpad2d(self, shape, dtype, pad):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)

        y_cpu = nn.functional.pad(x_cpu, pad, mode="reflect")
        y_supa = nn.functional.pad(x_supa, pad, mode="reflect")

        cpu_grad, supa_grad = create_random_tensor(
            y_cpu.shape, dtype=dtype, requires_grad=False
        )
        y_cpu.backward(cpu_grad)
        x_cpu_grad = x_cpu.grad

        y_supa.backward(supa_grad)
        x_supa_grad = x_supa.grad

        assert_allclose(y_cpu, y_supa, rtol=0, atol=0)
        assert_allclose(x_cpu_grad, x_supa_grad, rtol=1e-5, atol=5e-5)

    @pytest.mark.parametrize("shape, dtype", params_2d)
    @pytest.mark.parametrize("pad", pad_2d)
    def test_reflectionpad2d_out(self, shape, dtype, pad):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=False)

        m = nn.ReflectionPad2d(pad)
        s_cpu = m(x_cpu)

        y_cpu, y_supa = create_random_tensor(
            s_cpu.shape, dtype=dtype, requires_grad=False
        )

        y_cpu = torch._C._nn.reflection_pad2d(x_cpu, pad, out=y_cpu)
        y_supa = torch._C._nn.reflection_pad2d(x_supa, pad, out=y_supa)

        assert_allclose(y_cpu, y_supa, rtol=0, atol=0)


class TestReflectionPad3d:

    @pytest.mark.parametrize("shape, dtype", params_3d)
    @pytest.mark.parametrize("pad", pad_3d)
    def test_reflectionpad3d(self, shape, dtype, pad):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)

        y_cpu = nn.functional.pad(x_cpu, pad, mode="reflect")
        y_supa = nn.functional.pad(x_supa, pad, mode="reflect")

        cpu_grad, supa_grad = create_random_tensor(
            y_cpu.shape, dtype=dtype, requires_grad=False
        )
        y_cpu.backward(cpu_grad)
        x_cpu_grad = x_cpu.grad

        y_supa.backward(supa_grad)
        x_supa_grad = x_supa.grad

        assert_allclose(y_cpu, y_supa, rtol=0, atol=0)
        assert_allclose(x_cpu_grad, x_supa_grad, rtol=1e-5, atol=5e-5)

    @pytest.mark.parametrize("shape, dtype", params_3d)
    def test_reflectionpad3d_out(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=False)

        m = nn.ReflectionPad3d(1)
        s_cpu = m(x_cpu)
        m = m.supa()
        s_supa = m(x_supa)

        assert_allclose(s_cpu, s_supa, rtol=0, atol=0)
