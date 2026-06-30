# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

params = [
    pytest.param(
        (2, 128, 49),
        (14, 14),
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
        (64, 32, 16),
        (8, 8),
        (2, 2),
        (1, 1),
        (0, 0),
        (2, 2),
        marks=[pytest.mark.gcuStress],
    ),
    pytest.param(
        (53, 32, 16),
        (8, 8),
        (2, 2),
        (1, 1),
        (0, 0),
        (2, 2),
        marks=[pytest.mark.gcuStress],
    ),
    pytest.param(
        (127, 32, 16),
        (8, 8),
        (2, 2),
        (1, 1),
        (0, 0),
        (2, 2),
        marks=[pytest.mark.gcuStress],
    ),
]

torch.manual_seed(0)
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}
dtypes = [torch.float32, torch.bfloat16, torch.float16]


@pytest.mark.parametrize(
    "shape, output_size, kernel_size, dilation, padding, stride", params
)
@pytest.mark.parametrize("dtype", dtypes)
def test_col2im(shape, output_size, kernel_size, dilation, padding, stride, dtype):
    x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)
    fold = torch.nn.Fold(output_size, kernel_size, dilation, padding, stride)

    y_cpu = fold(x_cpu)
    grad_cpu, grad_supa = create_random_tensor(y_cpu.shape, dtype=dtype)
    y_cpu.backward(grad_cpu)
    y_supa = fold(x_supa)
    y_supa.backward(grad_supa)
    assert_allclose(y_cpu, y_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])
    assert_allclose(x_cpu.grad, x_supa.grad.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])
