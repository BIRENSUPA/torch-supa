# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


params = [
    pytest.param(
        (1, 4, 8, 8),
        (3, 3),
        2,
        1,
        1,
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
        (16, 32, 32, 32),
        (3, 3),
        2,
        1,
        1,
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (33 * 2, 32, 32, 32),
        (3, 3),
        2,
        1,
        1,
        torch.float32,
        marks=[pytest.mark.gcuStress],
    ),
    pytest.param(
        (33 * 3, 63, 31, 31),
        (3, 3),
        2,
        1,
        1,
        torch.float32,
        marks=[pytest.mark.gcuStress],
    ),
    pytest.param(
        (33 * 2 * 3, 63, 32, 32),
        (3, 3),
        2,
        1,
        1,
        torch.float32,
        marks=[pytest.mark.gcuStress],
    ),
]

RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}

torch.manual_seed(0)


class TestDilatedMaxPool2dMethod:

    @pytest.mark.parametrize(
        "input_shape, kernel_shape, stride, padding, dilation, dtype", params
    )
    def test_dilated_maxpool2d(
        self, input_shape, kernel_shape, stride, padding, dilation, dtype
    ):
        x_cpu, x_supa = create_random_tensor(
            input_shape, dtype=dtype, requires_grad=True
        )
        y_cpu = torch.nn.functional.max_pool2d_with_indices(
            x_cpu,
            kernel_size=kernel_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        y_supa = torch.nn.functional.max_pool2d_with_indices(
            x_supa,
            kernel_size=kernel_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        assert_allclose(y_cpu[0], y_supa[0], atol=ATOL[dtype], rtol=RTOL[dtype])
        assert_allclose(y_cpu[1], y_supa[1], atol=ATOL[dtype], rtol=RTOL[dtype])

        cpu_grad, supa_grad = create_random_tensor(
            y_cpu[0].shape, dtype=dtype, requires_grad=True
        )
        y_cpu[0].backward(cpu_grad)
        y_supa[0].backward(supa_grad)
        assert_allclose(x_cpu.grad, x_supa.grad, atol=ATOL[dtype], rtol=RTOL[dtype])
