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
        (2, 96, 72, 72),
        (3, 3),
        (1, 1),
        (0, 0),
        (1, 1),
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (1, 32 * 66, 64, 64),
        (3, 3),
        (1, 1),
        (1, 1),
        (2, 2),
        marks=[pytest.mark.gcuStress],
    ),
    pytest.param(
        (4, 32 * 66, 64, 64),
        (3, 3),
        (1, 1),
        (1, 1),
        (2, 2),
        marks=[pytest.mark.gcuStress],
    ),
    pytest.param(
        (9, 32 * 66, 64, 64),
        (3, 3),
        (1, 1),
        (1, 1),
        (2, 2),
        marks=[pytest.mark.gcuStress],
    ),
]

dtypes = [torch.float32, torch.float16, torch.bfloat16]
RTOL = {torch.float32: 1e-6, torch.bfloat16: 6e-1, torch.float16: 1e-1}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-1, torch.float16: 1e-2}


@pytest.mark.parametrize("shape, kernel_size, dilation, padding, stride", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_im2col(shape, kernel_size, dilation, padding, stride, dtype):
    x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)
    unfold = torch.nn.Unfold(kernel_size, dilation, padding, stride)

    y_cpu = unfold(x_cpu)
    grad_cpu, grad_supa = create_random_tensor(y_cpu.shape, dtype=dtype)
    y_cpu.backward(grad_cpu)
    y_supa = unfold(x_supa)
    y_supa.backward(grad_supa)
    assert_allclose(y_cpu, y_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])
    assert_allclose(x_cpu.grad, x_supa.grad.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])
