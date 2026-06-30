# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

params = [
    pytest.param(
        (2, 10, 4, 4),
        (2, 2),
        (1, 1),
        (0, 0),
        (2, 2),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 10, 256, 256),
        (2, 2),
        (1, 1),
        (0, 0),
        (2, 2),
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
]

dtypes = [torch.float32, torch.bfloat16, torch.float16]


class TestUnfold:

    @pytest.mark.parametrize("shape, kernel_size, dilation, padding, stride", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_unfold(self, shape, kernel_size, dilation, padding, stride, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        unfold = torch.nn.Unfold(kernel_size, dilation, padding, stride)

        y_cpu = unfold(x_cpu)
        y_supa = unfold(x_supa)
        assert_allclose(y_cpu, y_supa.cpu(), atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("shape, kernel_size, dilation, padding, stride", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_unfold_backward(
        self, shape, kernel_size, dilation, padding, stride, dtype
    ):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)
        unfold = torch.nn.Unfold(kernel_size, dilation, padding, stride)

        y_cpu = unfold(x_cpu)
        grad_cpu, grad_supa = create_random_tensor(y_cpu.shape, dtype=dtype)
        y_cpu.backward(grad_cpu)
        y_supa = unfold(x_supa)
        y_supa.backward(grad_supa)
        assert_allclose(y_cpu, y_supa.cpu(), atol=1e-6, rtol=1e-6)
        assert_allclose(x_cpu.grad, x_supa.grad.cpu(), atol=1e-6, rtol=1e-6)
