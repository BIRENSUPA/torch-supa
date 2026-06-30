# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import math
from functools import reduce

import numpy as np

# noqa
import pytest
import torch
import torch.nn as nn
from torch_supa.testing.common_utils import assert_allclose, create_torch_tensor_from_np

supa_device = torch.device("supa")
cpu_device = torch.device("cpu")
shapes = [
    (1280,),
    (32, 10000, 32),
    (74, 256, 256),
    (512, 200),
    (181, 290),
    (64, 128),
    (1, 64, 128),
    (12, 64, 128),
    (16, 64, 128),
    (19, 64, 128),
    (23, 64, 128),
    (32, 64, 128),
    (32, 290, 183),
    (1, 3, 64, 128),
    (12, 3, 64, 128),
    (19, 3, 64, 128),
    (25, 3, 64, 128),
    (32, 3, 64, 128),
    (59, 3, 64, 128),
    # conformer case
    (30, 325, 256),
    (25, 150, 2048),
    (25, 4, 150, 150),
    (1, 123, 256),
    (40, 4, 13, 13),
    (52, 77, 2048),
    (52, 4, 77, 77),
    (52, 14, 2048),
    (28, 22, 2048),
    (15, 4, 355, 355),
    (31, 4, 21, 140),
    (50, 79, 256),
    (65, 62, 2048),
    (40, 4, 111, 111),
    (16, 311, 256),
    (1, 311, 256),
    # GPT7B case
    (16, 12, 512, 512),
    (16, 512, 768),
    (512, 16, 768),
    (4, 1024, 4096),
    (4, 2048, 4096),
    (1024, 4, 4096),
    (4, 8, 1024, 1024),
    (16, 12, 512, 512),
    (32, 2, 137, 137),
    (16, 12, 512, 512),
    (32, 2, 137, 137),
]

br200_shapes = [
    pytest.param(
        (1000,),
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

br200_feature_shapes = [
    pytest.param(
        (1000, 2, 4, 2),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (1000, 16, 32, 16), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((1023, 511, 15, 15), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513, 32, 31), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 213, 16, 15), marks=[pytest.mark.gcuStress]),
]

shapes_2d = [
    (2, 512, 64, 128),
    (4, 512, 128),
    (512, 128),
]

# probs =[0.1, 0.3, 0.7, 0.9]
probs = [0.1]
# only test prop 0 and 1.
probs1 = [0.0, 1.0]

dtypes = [torch.float32, torch.bfloat16]


def create_ones_tensor(shape, dtype):
    cpu_input = torch.ones((shape), dtype=dtype, requires_grad=True)
    supa_input = torch.ones(
        (shape), dtype=dtype, requires_grad=True, device=torch.device("supa")
    )
    cpu_grad = torch.ones((shape), dtype=dtype)
    supa_grad = torch.ones((shape), dtype=dtype).supa()

    return cpu_input, supa_input, cpu_grad, supa_grad


def create_dropout_tensor(shape, dtype, requires_grad=False):
    if dtype == torch.bool:
        x = np.ones(shape).astype(bool)
    elif dtype == torch.half:
        x = np.ones(shape).astype(np.float16)
    elif dtype == torch.float:
        return create_ones_tensor(shape, dtype)
    elif dtype == torch.float64:
        x = np.ones(shape).astype(np.float64)
    elif dtype == torch.int64:
        x = np.ones(shape).astype(np.int64)
    elif dtype == torch.int32:
        x = np.ones(shape).astype(np.int32)
    elif dtype == torch.bfloat16:
        return create_ones_tensor(shape, dtype)
    else:
        raise TypeError("Unsupported type!")

    return create_torch_tensor_from_np(x, requires_grad)


def need_check_oob(input_shape: tuple, dtype) -> bool:
    check_oob_shape = [
        (512, 200),
        (181, 290),
        (64, 128),
        (32, 290, 183),
        (1, 3, 64, 128),
        (52, 77, 2048),
        (52, 4, 77, 77),
    ]

    check_oob_dtype = [torch.float32, torch.bfloat16]
    if input_shape in check_oob_shape and dtype in check_oob_dtype:
        return


class Pt2Dropout(nn.Module):
    def __init__(self, dropout_probability) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, input):
        output = self.dropout(input)
        return output


def check_output_ratio(
    prob,
    shape,
    cpu_fwd_count,
    supa_fwd_count,
    cpu_grad_count,
    supa_grad_count,
    num_iter=1,
    rel_error=0.3,
):
    expected_count = np.prod(shape) * (1 - prob) * num_iter
    cpu_rel_error = math.fabs(cpu_grad_count - expected_count) / expected_count
    supa_rel_error = math.fabs(supa_grad_count - expected_count) / expected_count
    cpu_fwd_rel_error = math.fabs(cpu_fwd_count - expected_count) / expected_count
    supa_fwd_rel_error = math.fabs(supa_fwd_count - expected_count) / expected_count
    length = reduce(lambda acc, curr: acc * curr, shape)
    if length < 3000:
        assert cpu_fwd_rel_error <= rel_error, "need less %s error range." % (rel_error)
        assert supa_fwd_rel_error <= rel_error, "need less %s error range." % (
            rel_error
        )
        assert cpu_rel_error <= rel_error, "need less %s error range." % (rel_error)
        assert supa_rel_error <= rel_error, "need less %s error range." % (rel_error)
    else:
        assert cpu_fwd_rel_error <= (rel_error / 2), "need less %s error range." % (
            rel_error / 2
        )
        assert supa_fwd_rel_error <= (rel_error / 2), "need less %s error range." % (
            rel_error / 2
        )
        assert cpu_rel_error <= (rel_error / 2), "need less %s error range." % (
            rel_error / 2
        )
        assert supa_rel_error <= (rel_error / 2), "need less %s error range." % (
            rel_error / 2
        )


def checkout_dropout_compu(input, shape, dtype, prob):
    fwd_final_count = 0
    fwd_final_count += torch.count_nonzero(input)
    sorted_value = torch.unique(input)
    zero_tensor = sorted_value[0].new_tensor(0)
    assert zero_tensor.equal(sorted_value[0])
    prob_tensor = sorted_value[1].new_tensor(1 / (1 - prob))
    assert_allclose(prob_tensor, sorted_value[1], atol=5 * 1e-5, rtol=1 * 1e-5)

    return fwd_final_count, sorted_value


class TestDropout:
    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("prob", probs)
    def test_dropout_forward_backward(self, shape, dtype, prob):
        cpu_final_count = 0
        supa_final_count = 0
        cpu_fwd_final_count = 0
        supa_fwd_final_count = 0
        num_iter = 10
        m = nn.Dropout(p=prob)

        for _ in range(0, num_iter):
            cpu_input, supa_input, cpu_grad, supa_grad = create_dropout_tensor(
                shape, dtype
            )
            cpu_output = m(cpu_input)
            cpu_fwd_count, sorted_value_cpu = checkout_dropout_compu(
                cpu_output, shape, dtype, prob
            )
            cpu_fwd_final_count += cpu_fwd_count
            cpu_output.backward(cpu_grad)
            cpu_res = cpu_input.grad
            cpu_final_count += torch.count_nonzero(cpu_res)

            supa_output = m(supa_input)
            supa_fwd_count, sorted_value_gpu = checkout_dropout_compu(
                supa_output.cpu(), shape, dtype, prob
            )
            supa_fwd_final_count += supa_fwd_count
            supa_output.backward(supa_grad)
            supa_res = supa_input.grad
            supa_final_count += torch.count_nonzero(supa_res.cpu())
            assert_allclose(
                sorted_value_cpu, sorted_value_gpu, atol=5 * 1e-5, rtol=1 * 1e-5
            )
        check_output_ratio(
            prob,
            shape,
            cpu_fwd_final_count,
            supa_fwd_final_count,
            cpu_final_count,
            supa_final_count,
            num_iter,
        )

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("prob", probs)
    def test_alpha_dropout(self, shape, dtype, prob):

        m = nn.AlphaDropout(p=prob)
        input = torch.randn(shape)
        mean = input.mean()
        std = input.std()
        supa_input = input.to(supa_device)
        supa_output = m(supa_input)
        supa_out_cpu = supa_output.to(cpu_device)
        assert_allclose(mean, supa_out_cpu.mean(), atol=0.1, rtol=0.01)
        assert_allclose(std, supa_out_cpu.std(), atol=0.1, rtol=0.01)

    @pytest.mark.parametrize("shape", br200_feature_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("prob", probs)
    def test_feature_alpha_dropout(self, shape, dtype, prob):
        m = nn.FeatureAlphaDropout(p=prob)
        input = torch.randn(shape)
        mean = input.mean()
        std = input.std()
        supa_input = input.to(supa_device)
        supa_output = m(supa_input)
        supa_out_cpu = supa_output.to(cpu_device)
        assert_allclose(mean, supa_out_cpu.mean(), atol=0.1, rtol=0.01)
        assert_allclose(std, supa_out_cpu.std(), atol=0.1, rtol=0.01)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("prob", probs1)
    def test_dropout_forward_backward_p01(self, shape, dtype, prob):
        cpu_input, supa_input, cpu_grad, supa_grad = create_dropout_tensor(shape, dtype)
        m = nn.Dropout(p=prob)
        cpu_output = m(cpu_input)
        cpu_output.backward(cpu_grad)
        cpu_res = cpu_input.grad
        supa_output = m(supa_input)
        supa_output.backward(torch.ones_like(supa_output))

        torch.testing.assert_allclose(cpu_output, supa_output.cpu(), rtol=0, atol=0)
        torch.testing.assert_allclose(cpu_res, supa_input.grad.cpu(), rtol=0, atol=0)

    @pytest.mark.parametrize("shape", shapes_2d)
    @pytest.mark.parametrize("prob", probs)
    def test_dropout2d_forward_backward(self, shape, prob):
        cpu_final_count = 0
        supa_final_count = 0
        cpu_fwd_final_count = 0
        supa_fwd_final_count = 0
        m = nn.Dropout2d(p=prob)  # aten::bernoulli_.float
        num_iter = 10

        for i_ in range(0, num_iter):
            cpu_input, supa_input, cpu_grad, supa_grad = create_dropout_tensor(
                shape, torch.float32
            )
            # cpu dropout2d fwd and bwd
            cpu_output = m(cpu_input)
            cpu_fwd_count, sorted_value_cpu = checkout_dropout_compu(
                cpu_output, shape, torch.float32, prob
            )
            cpu_fwd_final_count += cpu_fwd_count
            cpu_output.backward(cpu_grad)
            cpu_res = cpu_input.grad
            cpu_final_count += torch.count_nonzero(cpu_res)

            # supa dropout2d fwd and bwd
            supa_output = m(supa_input)
            supa_fwd_count, sorted_value_gpu = checkout_dropout_compu(
                supa_output.cpu(), shape, torch.float32, prob
            )
            supa_fwd_final_count += supa_fwd_count
            supa_output.backward(supa_grad)
            supa_res = supa_input.grad
            supa_final_count += torch.count_nonzero(supa_res.cpu())

        # Check that we are in the 15 % error range
        same_point_num = 1
        if len(shape) == 4:
            same_point_num = shape[2] * shape[3]
            shape = (shape[0], shape[1])
        elif len(shape) == 3:
            same_point_num = shape[2]
            shape = (shape[0], shape[1])
        check_output_ratio(
            prob,
            shape,
            cpu_fwd_final_count / same_point_num,
            supa_fwd_final_count / same_point_num,
            cpu_final_count / same_point_num,
            supa_final_count / same_point_num,
            num_iter=num_iter,
            rel_error=0.3,
        )

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("prob", probs)
    def test_dropout_forward_backward_perf(self, shape, dtype, prob):
        m = nn.Dropout(p=prob)
        if dtype == torch.bfloat16:
            supa_input = torch.ones(
                (shape),
                dtype=torch.bfloat16,
                requires_grad=True,
                device=torch.device("supa"),
            )
            supa_output = m(supa_input)
            supa_output.backward(torch.ones_like(supa_output))

        elif dtype == torch.float32:
            supa_input = torch.ones(
                (shape), dtype=dtype, requires_grad=True, device=torch.device("supa")
            )
            supa_grad = torch.ones((shape), dtype=dtype).supa()

            # test supa implement dropout.
            supa_output = m(supa_input)
            supa_output.backward(supa_grad)
