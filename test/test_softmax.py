# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
import torch.nn as nn

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

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
    pytest.param((512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.float32]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


class TestSoftmax:

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_softmax_forward_br200(self, shape, dtype):
        # For bf16 prec issue
        if dtype == torch.bfloat16:
            return

        def test_softmax_forward(shape, dtype):
            cpu_input, supa_input = create_random_tensor(shape, dtype=dtype)
            cpu_output = nn.functional.softmax(cpu_input, -1)
            supa_output = nn.functional.softmax(supa_input, -1)

            assert_allclose(
                cpu_output,
                supa_output,
                atol=ATOL[dtype],
                rtol=RTOL[dtype],
                equal_nan=True,
            )

        test_softmax_forward(shape, dtype)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_softmax_backward_br200(self, shape, dtype):
        if dtype == torch.bfloat16:
            return

        def test_softmax_backward(shape, dtype):
            cpu_input, supa_input = create_random_tensor(
                shape, dtype=dtype, requires_grad=True
            )
            cpu_grad, supa_grad = create_random_tensor(shape, dtype=dtype)
            cpu_output = nn.functional.softmax(cpu_input, -1)
            cpu_output.backward(cpu_grad)
            cpu_res = cpu_input.grad

            supa_output = nn.functional.softmax(supa_input, -1)
            supa_output.backward(supa_grad)
            supa_res = supa_input.grad

            assert_allclose(
                cpu_res, supa_res, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
            )

        test_softmax_backward(shape, dtype)


class TestLogSoftmax:

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_log_softmax_forward_br200(self, shape, dtype):
        if dtype == torch.bfloat16:
            return

        def test_log_softmax_forward(shape, dtype):
            cpu_input, supa_input = create_random_tensor(shape, dtype=dtype)
            cpu_output = nn.functional.log_softmax(cpu_input, -1)
            supa_output = nn.functional.log_softmax(supa_input, -1)

            assert_allclose(
                cpu_output,
                supa_output,
                atol=ATOL[dtype],
                rtol=RTOL[dtype],
                equal_nan=True,
            )

        test_log_softmax_forward(shape, dtype)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_log_softmax_backward_br200(self, shape, dtype):
        if dtype == torch.bfloat16:
            return

        def test_log_softmax_backward(shape, dtype):
            cpu_input, supa_input = create_random_tensor(
                shape, dtype=dtype, requires_grad=True
            )
            cpu_grad, supa_grad = create_random_tensor(shape, dtype=dtype)
            cpu_output = nn.functional.log_softmax(cpu_input, -1)
            cpu_output.backward(cpu_grad)
            cpu_res = cpu_input.grad

            supa_output = nn.functional.log_softmax(supa_input, -1)
            supa_output.backward(supa_grad)
            supa_res = supa_input.grad

            assert_allclose(
                cpu_res, supa_res, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
            )

        test_log_softmax_backward(shape, dtype)
