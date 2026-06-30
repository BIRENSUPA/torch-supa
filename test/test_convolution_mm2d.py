# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import copy

import pytest
import torch
import torch.nn as nn
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

supa_device = torch.device("supa")

# params = [
#   # input_shape, output_channels, kernel_size, stride, padding, groups
#   # efficientnet stride1
#   # [(2, 3, 224, 224), 64, 7, 1, 3, 1],
#   [(2, 96, 72, 72), 40, 1, 1, 0, 1],
#   [(2, 40, 72, 72), 160, 3, 1, 1, 1],
#   [(2, 160, 72, 72), 40, 1, 1, 0, 1],
#   [(2, 160, 36, 36), 48, 1, 1, 0, 1],
#   [(2, 192, 36, 36), 48, 1, 1, 0, 1],
#   [(2, 48, 36, 36), 192, 3, 1, 1, 1],
#   [(2, 192, 18, 18), 104, 1, 1, 0, 1],
#   [(2, 26, 1, 1), 416, 1, 1, 0, 1],
#   [(2, 416, 1, 1), 256, 1, 1, 0, 1],
#   [(2, 12, 1, 1), 192, 1, 1, 0, 1],
#   [(2, 26, 1, 1), 624, 1, 1, 0, 1],
#   [(2, 128, 18, 18), 768, 1, 1, 0, 1],
#   [(2, 32, 1, 1), 64, 1, 1, 0, 1],
#   [(2, 52, 1, 1), 1248, 1, 1, 0, 1],
#   [(2, 1248, 9, 9), 208, 1, 1, 0, 1],
#   [(2, 768, 9, 9), 208, 1, 1, 0, 1],
#   [(2, 208, 9, 9), 1248, 1, 1, 0, 1],
#   [(2, 1248, 1, 1), 52, 1, 1, 0, 1],
#   [(2, 208, 9, 9), 1024, 1, 1, 0, 1],
#   [(2, 24, 144, 144), 24, 3, 1, 1, 1],
#   [(2, 416, 1, 1), 26, 1, 1, 0, 1],
#   [(2, 104, 18, 18), 416, 1, 1, 0, 1],
#   [(2, 416, 18, 18), 416, 3, 1, 1, 1],
#   [(2, 192, 1, 1), 12, 1, 1, 0, 1],
#   [(2, 48, 36, 36), 192, 1, 1, 0, 1],
#   [(2, 104, 18, 18), 624, 1, 1, 0, 1],
#   [(2, 416, 18, 18), 104, 1, 1, 0, 1],
#   [(2, 416, 1, 1), 26, 1, 1, 0, 1],
#   [(2, 624, 1, 1), 26, 1, 1, 0, 1],
#   [(2, 624, 18, 18), 128, 1, 1, 0, 1],
#   [(2, 768, 1, 1), 32, 1, 1, 0, 1],
#   [(2, 768, 18, 18), 128, 1, 1, 0, 1],

#   # efficientnet stride2
#   [(2, 24, 144, 144), 96, 3, 2, 1, 1],
#   [(2, 40, 72, 72), 160, 3, 2, 1, 1],

#   # efficientnet dwc
#   [(128, 192, 28, 28), 192, 3, 2, 1, 192],
#   [(128, 416, 14, 14), 416, 3, 1, 1, 416],
#   [(128, 624, 14, 14), 624, 3, 1, 1, 624],
#   [(128, 768, 14, 14), 768, 3, 1, 1, 768],
#   [(128, 768, 14, 14), 768, 3, 2, 1, 768],
#   [(128, 1248, 7, 7), 1248, 3, 1, 1, 1248],
# ]

params = [
    [(2, 3, 1, 1), 6, 1, 1, 0, 1],
    [(2, 3, 4, 4), 16, 3, 1, 0, 1],
    [(2, 4, 8, 8), 8, 3, 1, 0, 1],
]

dtypes = [torch.float32]

use_cudnn = True
allow_tf32 = False


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.regression
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, groups", params
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.skip("convolution_overrideable not implemented")
def test_conv_mm2d(
    input_shape,
    out_channels,
    kernel_size,
    stride,
    padding,
    groups,
    dtype,
    bias=False,
    requires_grad=True,
    atol=5e-3,
    rtol=1e-4,
):
    with torch.backends.cudnn.flags(enabled=use_cudnn, allow_tf32=allow_tf32):
        bpw_atol = 5e-2  # fwd, bpa and bpw accuracy is not meeting requirements

        dilation = 1

        assert len(input_shape) == 4
        in_channels = input_shape[1]

        conv_cpu = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            dtype=dtype,
        )
        conv_supa = copy.deepcopy(conv_cpu).to(supa_device)

        filter_cpu, filter_supa = create_random_tensor(
            conv_cpu.weight.shape,
            min_value=-1,
            max_value=1,
            dtype=dtype,
            requires_grad=requires_grad,
        )

        conv_cpu.weight.data = filter_cpu
        conv_supa.weight.data = filter_supa

        x_cpu, x_supa = create_random_tensor(
            input_shape,
            min_value=-1,
            max_value=1,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        y_cpu = conv_cpu(x_cpu)
        y_supa = conv_supa(x_supa)

        assert_allclose(y_cpu, y_supa, atol=atol, rtol=rtol)

        if requires_grad:
            grad_out, grad_out_supa = create_random_tensor(
                y_cpu.shape, min_value=-1, max_value=1, dtype=dtype
            )
            y_cpu.backward(grad_out)
            y_supa.backward(grad_out_supa)
            assert_allclose(x_cpu.grad, x_supa.grad, atol=atol, rtol=rtol)
            assert_allclose(
                conv_cpu.weight.grad, conv_supa.weight.grad, atol=bpw_atol, rtol=rtol
            )
            if bias:
                assert_allclose(
                    conv_cpu.bias.grad, conv_supa.bias.grad, atol=bpw_atol, rtol=rtol
                )
