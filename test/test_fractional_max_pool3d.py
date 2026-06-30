# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# noqa

import pytest
import torch
import torch.nn.functional as F
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


br200_shapes = [
    (1, 1, 4, 4, 4),
]


kernel = [
    (2),
]

output_ratio = [
    (0.5, 0.5, 0.5),
]

dtypes = [torch.float32, torch.float16, torch.bfloat16]

RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


class TestNNMethod:

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("kernel", kernel)
    @pytest.mark.parametrize("output_ratio", output_ratio)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_fractional_maxpool3d_fwd(self, shape, kernel, output_ratio, dtype):
        cpu_input, supa_input = create_random_tensor(
            shape, dtype=dtype, requires_grad=False
        )
        cpu_output = F.fractional_max_pool3d(
            cpu_input, kernel_size=kernel, output_ratio=output_ratio
        )
        supa_output = F.fractional_max_pool3d(
            supa_input, kernel_size=kernel, output_ratio=output_ratio
        )
        assert_allclose(
            cpu_output, supa_output.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype]
        )

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("kernel", kernel)
    @pytest.mark.parametrize("output_ratio", output_ratio)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_fractional_maxpool3d_bwd(self, shape, kernel, output_ratio, dtype):
        cpu_input, supa_input = create_random_tensor(
            shape, dtype=dtype, requires_grad=True
        )
        cpu_output = F.fractional_max_pool3d(
            cpu_input, kernel_size=kernel, output_ratio=output_ratio
        )
        supa_output = F.fractional_max_pool3d(
            supa_input, kernel_size=kernel, output_ratio=output_ratio
        )
        cpu_grad, supa_grad = create_random_tensor(
            cpu_output.shape, dtype=dtype, requires_grad=False
        )
        cpu_output.backward(cpu_grad)
        cpu_inputs_grad = cpu_input.grad
        supa_output.backward(supa_grad)
        supa_inputs_grad = supa_input.grad
        assert_allclose(
            cpu_inputs_grad, supa_inputs_grad.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype]
        )

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("kernel", kernel)
    @pytest.mark.parametrize("output_ratio", output_ratio)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_fractional_maxpool3d_fwd_br200(self, shape, kernel, output_ratio, dtype):
        self.test_fractional_maxpool3d_fwd(
            shape=shape, kernel=kernel, output_ratio=output_ratio, dtype=dtype
        )

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("kernel", kernel)
    @pytest.mark.parametrize("output_ratio", output_ratio)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_fractional_maxpool3d_bwd_br200(self, shape, kernel, output_ratio, dtype):
        self.test_fractional_maxpool3d_bwd(
            shape=shape, kernel=kernel, output_ratio=output_ratio, dtype=dtype
        )
