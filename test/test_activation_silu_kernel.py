# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
import torch.nn.functional as F
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

ATOL = 5 * 1e-5
RTOL = 1 * 1e-5


br200_shapes = [
    pytest.param(
        (16,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((1024, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.float32, torch.bfloat16, torch.float16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}

requires_grads = [False, True]


class TestSilu:

    @pytest.mark.parametrize("in_shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("requires_grad", requires_grads)
    def test_silu_op(self, in_shape, dtype, requires_grad):
        cpu_in, supa_in = create_random_tensor(
            in_shape, dtype=dtype, requires_grad=requires_grad
        )
        cpu_out = F.silu(cpu_in, False)
        supa_out = F.silu(supa_in, False)

        if not requires_grad:
            assert_allclose(cpu_out, supa_out.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])
        else:
            cpu_grad, supa_grad = create_random_tensor(
                in_shape, dtype=dtype, requires_grad=False
            )
            cpu_out.backward(cpu_grad)
            cpu_in_grad = cpu_in.grad

            supa_out.backward(supa_grad)
            supa_in_grad = supa_in.grad
            assert_allclose(
                cpu_in_grad, supa_in_grad.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype]
            )
