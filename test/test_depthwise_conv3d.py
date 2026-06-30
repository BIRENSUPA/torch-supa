# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_supa.testing.common_utils as test_util

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


shapes = [
    # input_shape, output_channels, kernel_size, stride, padding, groups
    pytest.param(
        (1, 2, 14, 14, 14),
        2,
        (3, 3, 3),
        2,
        1,
        2,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (16, 64, 7, 7, 7),
        64,
        (3, 3, 3),
        1,
        1,
        64,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
]

bias_params = [
    False,
    True,
]

is_backwards = [False, True]

dtypes = [torch.float32]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-4, torch.bfloat16: 1e-5, torch.float16: 1e-5}


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, groups", shapes
)
@pytest.mark.parametrize("bias", bias_params)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("is_backward", is_backwards)
def test_dwc3d(
    input_shape,
    out_channels,
    kernel_size,
    stride,
    padding,
    groups,
    bias,
    dtype,
    is_backward,
):
    torch.manual_seed(0)
    dilation = 1

    np.testing.assert_equal(len(input_shape), 5)
    in_channels = input_shape[1]

    conv_cpu = nn.Conv3d(
        in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
    ).to(dtype)
    conv_supa = copy.deepcopy(conv_cpu).to(dtype).to(supa_device)

    filter_cpu, filter_supa = test_util.create_random_tensor(
        conv_cpu.weight.shape,
        dtype=dtype,
        min_value=-1,
        max_value=1,
        requires_grad=True,
    )
    conv_cpu.weight.data = filter_cpu
    conv_supa.weight.data = filter_supa

    x_cpu, x_supa = test_util.create_random_tensor(
        input_shape, dtype=dtype, min_value=-1, max_value=1, requires_grad=True
    )
    y_cpu = conv_cpu(x_cpu)
    y_supa = conv_supa(x_supa)

    test_util.assert_allclose(y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])

    if not is_backward:
        test_util.assert_allclose(y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])
    else:
        grad_out, grad_out_supa = test_util.create_random_tensor(
            y_cpu.shape, dtype=dtype, min_value=-1, max_value=1
        )
        y_cpu.backward(grad_out)
        y_supa.backward(grad_out_supa)

        test_util.assert_allclose(
            x_cpu.grad, x_supa.grad, atol=ATOL[dtype], rtol=RTOL[dtype]
        )
        test_util.assert_allclose(
            conv_cpu.weight.grad,
            conv_supa.weight.grad,
            atol=ATOL[dtype],
            rtol=RTOL[dtype],
        )
        if bias:
            test_util.assert_allclose(
                conv_cpu.bias.grad,
                conv_supa.bias.grad,
                atol=ATOL[dtype],
                rtol=RTOL[dtype],
            )
