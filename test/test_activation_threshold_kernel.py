# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


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

thresholds = [0.5, 1.0]

values = [
    20,
]

dtypes = [torch.float32, torch.bfloat16, torch.float16]

requires_grads = [True]


class TestPrelu:

    @pytest.mark.parametrize("input_shape", br200_shapes)
    @pytest.mark.parametrize("threshold", thresholds)
    @pytest.mark.parametrize("value", values)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("requires_grad", requires_grads)
    def test_threshold(self, input_shape, threshold, value, dtype, requires_grad):
        cpu_input, supa_input = create_random_tensor(
            input_shape, dtype=dtype, requires_grad=requires_grad
        )

        cpu_out = torch.nn.functional.threshold(cpu_input, threshold, value)
        supa_out = torch.nn.functional.threshold(supa_input, threshold, value)

        assert_allclose(cpu_out, supa_out, rtol=0, atol=0)

        if requires_grad:
            cpu_grad, supa_grad = create_random_tensor(cpu_out.shape, dtype=dtype)
            cpu_out.backward(cpu_grad)
            cpu_res = cpu_input.grad

            supa_out.backward(supa_grad)
            supa_res = supa_input.grad

            assert_allclose(cpu_res, supa_res, rtol=0, atol=0)

    @pytest.mark.parametrize("input_shape", br200_shapes)
    @pytest.mark.parametrize("threshold", thresholds)
    @pytest.mark.parametrize("value", values)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("requires_grad", requires_grads)
    def test_threshold_inplace(
        self, input_shape, threshold, value, dtype, requires_grad
    ):
        cpu_input, supa_input = create_random_tensor(input_shape, dtype=dtype)

        cpu_out = torch.nn.functional.threshold_(cpu_input, threshold, value)
        supa_out = torch.nn.functional.threshold_(supa_input, threshold, value)

        assert_allclose(cpu_out, supa_out, rtol=0, atol=0)
