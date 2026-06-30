# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

import torch.nn as nn

# from torch.testing._internal.common_utils import TestCase
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

shapes = [
    (1, 512, 112, 112),
    (1, 8, 224, 224),
    (1, 1, 112, 112),
    (16, 32, 112, 112),
    # (16, 32, 224, 224),
    (16, 128, 112, 112),
    (16, 128, 224, 224),
    (16, 256, 112, 112),
    (16, 256, 224, 224),
    # (16, 512, 112, 112),
    (16, 512, 224, 224),
]

br200_params = [
    pytest.param(
        (1, 8, 16, 16),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (16, 4, 256, 256), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param(
        (4, 1024, 112, 112), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((8, 132, 256, 256), marks=[pytest.mark.gcuStress]),
    pytest.param((4, 132 * 66, 64, 64), marks=[pytest.mark.gcuStress]),
    pytest.param((7, 132 * 66, 65, 65), marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.float32]

RTOL = {torch.float32: 5e-5, torch.bfloat16: 0.016}
ATOL = {torch.float32: 5e-5, torch.bfloat16: 1e-3}


class TestBatchNorm:

    @pytest.mark.parametrize("shape", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_batch_norm_forward_br200(self, shape, dtype):
        def test_batch_norm_forward(shape, dtype):
            c = shape[1]
            cpu_in, supa_in = create_random_tensor(shape, dtype=dtype)
            cpu_bn2d = nn.BatchNorm2d(c)
            supa_bn2d = nn.BatchNorm2d(c)
            supa_bn2d = supa_bn2d.to("supa")
            cpu_out = cpu_bn2d(cpu_in)
            supa_out = supa_bn2d(supa_in)
            assert_allclose(cpu_out, supa_out.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])

        test_batch_norm_forward(shape, dtype)

    @pytest.mark.parametrize("shape", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_batch_norm_backward_br200(self, shape, dtype):
        def test_batch_norm_backward(shape, dtype):
            torch.manual_seed(0)
            c = shape[1]
            cpu_in, supa_in = create_random_tensor(
                shape, dtype=dtype, requires_grad=True
            )
            cpu_grad, supa_grad = create_random_tensor(
                shape, dtype=dtype, requires_grad=True
            )
            cpu_weight, supa_weight = create_random_tensor(
                [c], dtype=dtype, requires_grad=True
            )
            cpu_bias, supa_bias = create_random_tensor(
                [c], dtype=dtype, requires_grad=True
            )
            cpu_running_mean, supa_running_mean = create_random_tensor(
                [c], dtype=dtype, requires_grad=True
            )
            cpu_running_var, supa_running_var = create_random_tensor(
                [c], dtype=dtype, requires_grad=True
            )

            cpu_bn2d = torch.nn.BatchNorm2d(c)
            cpu_bn2d.weight.data = cpu_weight
            cpu_bn2d.bias.data = cpu_bias
            cpu_bn2d.running_mean.data = cpu_running_mean
            cpu_bn2d.running_var.data = cpu_running_var

            supa_bn2d = torch.nn.BatchNorm2d(c)
            supa_bn2d.requires_grad_(True)
            supa_bn2d = supa_bn2d.to("supa")
            supa_bn2d.weight.data = supa_weight
            supa_bn2d.bias.data = supa_bias
            supa_bn2d.running_mean.data = supa_running_mean
            supa_bn2d.running_var.data = supa_running_var

            cpu_output = cpu_bn2d(cpu_in)
            cpu_running_mean = cpu_bn2d.running_mean
            cpu_running_var = cpu_bn2d.running_var
            cpu_output.backward(cpu_grad)
            cpu_in_grad = cpu_in.grad
            cpu_weight_grad = cpu_bn2d.weight.grad
            cpu_bias_grad = cpu_bn2d.bias.grad

            supa_output = supa_bn2d(supa_in)
            supa_running_mean = supa_bn2d.running_mean
            supa_running_var = supa_bn2d.running_var
            supa_output.backward(supa_grad)
            supa_in_grad = supa_in.grad
            supa_weight_grad = supa_bn2d.weight.grad
            supa_bias_grad = supa_bn2d.bias.grad

            assert_allclose(cpu_output, supa_output, atol=5 * 1e-5, rtol=1 * 1e-5)
            assert_allclose(
                cpu_bias_grad, supa_bias_grad.cpu(), atol=5 * 1e-2, rtol=1 * 1e-2
            )
            assert_allclose(
                cpu_running_mean, supa_running_mean.cpu(), atol=5 * 1e-5, rtol=1 * 1e-5
            )
            assert_allclose(
                cpu_running_var, supa_running_var.cpu(), atol=5 * 1e-5, rtol=1 * 1e-5
            )
            assert_allclose(
                cpu_weight_grad, supa_weight_grad.cpu(), atol=5 * 1e-2, rtol=1 * 1e-2
            )
            assert_allclose(
                cpu_in_grad, supa_in_grad.cpu(), atol=5 * 1e-5, rtol=1 * 1e-5
            )

        test_batch_norm_backward(shape, dtype)
