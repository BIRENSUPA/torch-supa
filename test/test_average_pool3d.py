# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
import torch.nn as nn

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

torch.manual_seed(0)


br200_params = [
    pytest.param(
        (1, 1, 18, 18, 18),
        (1, 1, 17, 17, 17),
        torch.float,
        2,
        1,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (16, 2, 18, 18, 18),
        (16, 2, 17, 17, 17),
        torch.float,
        (2, 2, 2),
        (1, 1, 1),
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (17, 5, 20, 20, 20),
        (17, 5, 19, 19, 19),
        torch.float,
        (2, 2, 2),
        (1, 1, 1),
        marks=[pytest.mark.gcuStress],
    ),
    pytest.param(
        (17, 7, 20, 20, 20),
        (17, 7, 19, 19, 19),
        torch.float,
        (2, 2, 2),
        (1, 1, 1),
        marks=[pytest.mark.gcuStress],
    ),
]

params_with_pad = [
    pytest.param(
        (1, 1, 18, 18, 18),
        (1, 1, 18, 18, 18),
        torch.float,
        3,
        1,
        1,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (16, 4, 12, 12, 12),
        (16, 4, 12, 12, 12),
        torch.float,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (17, 5, 17, 17, 17),
        (17, 5, 17, 17, 17),
        torch.float,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        marks=[pytest.mark.gcuStress],
    ),
    pytest.param(
        (17, 7, 21, 21, 21),
        (17, 7, 21, 21, 21),
        torch.float,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        marks=[pytest.mark.gcuStress],
    ),
]

count_include_pad = [True, False]


class TestAvgPool3dMethod:

    @pytest.mark.parametrize(
        "input_shape, output_shape, dtype, kernel, stride", br200_params
    )
    def test_avgpool3d(self, input_shape, output_shape, dtype, kernel, stride):
        x_cpu, x_supa = create_random_tensor(
            input_shape, dtype=dtype, requires_grad=True
        )
        x_supa = x_supa
        cpu_grad, supa_grad = create_random_tensor(
            output_shape, dtype=dtype, requires_grad=False
        )

        avgpool3d_cpu = nn.AvgPool3d(kernel, stride=stride)
        y_cpu = avgpool3d_cpu(x_cpu)
        y_cpu.backward(cpu_grad)

        avg_pool_supa = avgpool3d_cpu.to(supa_device)
        y_supa = avg_pool_supa(x_supa)
        y_supa.backward(supa_grad)

        assert_allclose(y_cpu, y_supa.cpu(), rtol=1e-5, atol=5e-5)
        assert_allclose(x_cpu.grad, x_supa.grad.cpu(), rtol=1e-5, atol=5e-5)

    @pytest.mark.parametrize(
        "input_shape, output_shape, dtype, kernel, stride, padding", params_with_pad
    )
    def test_avgpool3d_pad(
        self, input_shape, output_shape, dtype, kernel, stride, padding
    ):
        x_cpu, x_supa = create_random_tensor(
            input_shape, dtype=dtype, requires_grad=True
        )
        cpu_grad, supa_grad = create_random_tensor(
            output_shape, dtype=dtype, requires_grad=False
        )

        avgpool3d_cpu = nn.AvgPool3d(kernel, stride=stride, padding=padding)
        y_cpu = avgpool3d_cpu(x_cpu)
        y_cpu.backward(cpu_grad)

        avg_pool_supa = avgpool3d_cpu.to(supa_device)
        y_supa = avg_pool_supa(x_supa)
        y_supa.backward(supa_grad)

        assert_allclose(y_cpu, y_supa.cpu(), rtol=1e-5, atol=5e-5)
        # wait sulib check padding output
        assert_allclose(x_cpu.grad, x_supa.grad.cpu(), rtol=1e-5, atol=5e-5)

    @pytest.mark.parametrize(
        "input_shape, output_shape, dtype, kernel, stride, padding", params_with_pad
    )
    def test_avgpool3d_exclude(
        self, input_shape, output_shape, dtype, kernel, stride, padding
    ):
        x_cpu, x_supa = create_random_tensor(
            input_shape, dtype=dtype, requires_grad=True
        )
        cpu_grad, supa_grad = create_random_tensor(
            output_shape, dtype=dtype, requires_grad=True
        )

        avgpool3d_cpu = nn.AvgPool3d(
            kernel, stride=stride, padding=padding, count_include_pad=False
        )
        y_cpu = avgpool3d_cpu(x_cpu)
        y_cpu.backward(cpu_grad)

        avg_pool_supa = avgpool3d_cpu.to(supa_device)
        y_supa = avg_pool_supa(x_supa)
        y_supa.backward(supa_grad)

        assert_allclose(y_cpu, y_supa.cpu(), rtol=1e-5, atol=5e-5)
        assert_allclose(x_cpu.grad, x_supa.grad.cpu(), rtol=1e-5, atol=5e-5)

    @pytest.mark.parametrize(
        "input_shape, output_shape, dtype, kernel, stride, padding", params_with_pad
    )
    @pytest.mark.parametrize("count_include_pad", count_include_pad)
    def test_avgpool3d_exclude_supa_kernel(
        self,
        input_shape,
        output_shape,
        dtype,
        kernel,
        stride,
        padding,
        count_include_pad,
    ):
        x_cpu, x_supa = create_random_tensor(
            input_shape, dtype=dtype, requires_grad=True
        )
        cpu_grad, supa_grad = create_random_tensor(
            output_shape, dtype=dtype, requires_grad=False
        )

        avgpool3d_cpu = nn.AvgPool3d(
            kernel, stride=stride, padding=padding, count_include_pad=count_include_pad
        )
        y_cpu = avgpool3d_cpu(x_cpu)
        y_cpu.backward(cpu_grad)

        avg_pool_supa = avgpool3d_cpu.to(supa_device)
        y_supa = avg_pool_supa(x_supa)
        y_supa.backward(supa_grad)

        assert_allclose(y_cpu, y_supa.cpu(), rtol=1e-5, atol=5e-5)
        assert_allclose(x_cpu.grad, x_supa.grad.cpu(), rtol=1e-5, atol=5e-5)
