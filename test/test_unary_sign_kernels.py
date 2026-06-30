# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    RandomMode,
)

params = [
    pytest.param(
        (1, 2, 3, 4),
        True,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (1, 2, 4, 4),
        False,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (16, 2, 256, 256), True, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param(
        (8, 2, 512, 512), False, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((16, 2, 511, 511), True, marks=[pytest.mark.gcuStress]),
    pytest.param((2, 31, 513, 513), False, marks=[pytest.mark.gcuStress]),
    pytest.param((3, 63, 1021, 1021), False, marks=[pytest.mark.gcuStress]),
]


shapes = [
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
    pytest.param(
        (32,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (1, 15),
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


class TestLogicalOps:

    @pytest.mark.parametrize("shape, in_place", params)
    def test_logical_not(self, shape, in_place):
        RTOL = 0
        ATOL = 0
        dtype = torch.bool

        input_cpu, input_supa = create_random_tensor(shape, dtype)
        if in_place:
            output_cpu = input_cpu.logical_not_()
            output_supa = input_supa.logical_not_()
        else:
            output_cpu = torch.logical_not(input_cpu)
            output_supa = torch.logical_not(input_supa)

        assert_allclose(output_cpu, output_supa, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("shape, in_place", params)
    def test_logical_and(self, shape, in_place):
        RTOL = 0
        ATOL = 0
        dtype = torch.bool

        input_cpu, input_supa = create_random_tensor(shape, dtype)
        input2_cpu, input2_supa = create_random_tensor(shape, dtype)
        if in_place:
            output_cpu = input_cpu.logical_and_(input2_cpu)
            output_supa = input_supa.logical_and_(input2_supa)
        else:
            output_cpu = torch.logical_and(input_cpu, input2_cpu)
            output_supa = torch.logical_and(input_supa, input2_supa)

        assert_allclose(output_cpu, output_supa, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("shape, in_place", params)
    def test_logical_or(self, shape, in_place):
        RTOL = 0
        ATOL = 0
        dtype = torch.bool

        input_cpu, input_supa = create_random_tensor(shape, dtype)
        input2_cpu, input2_supa = create_random_tensor(shape, dtype)
        if in_place:
            output_cpu = input_cpu.logical_or_(input2_cpu)
            output_supa = input_supa.logical_or_(input2_supa)
        else:
            output_cpu = torch.logical_or(input_cpu, input2_cpu)
            output_supa = torch.logical_or(input_supa, input2_supa)

        assert_allclose(output_cpu, output_supa, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("shape, in_place", params)
    def test_logical_xor(self, shape, in_place):
        RTOL = 0
        ATOL = 0
        dtype = torch.bool

        input_cpu, input_supa = create_random_tensor(shape, dtype)
        input2_cpu, input2_supa = create_random_tensor(shape, dtype)
        if in_place:
            output_cpu = input_cpu.logical_xor_(input2_cpu)
            output_supa = input_supa.logical_xor_(input2_supa)
        else:
            output_cpu = torch.logical_xor(input_cpu, input2_cpu)
            output_supa = torch.logical_xor(input_supa, input2_supa)

        assert_allclose(output_cpu, output_supa, atol=ATOL, rtol=RTOL)


scalar_values = [0.666, -0.666, 0.0]

scalar_grad_values = [0.888, -0.888, 0.0]

dtypes = [torch.float32, torch.bfloat16, torch.float16]

# data distribution, min_value, max_value
data_distributions = [(RandomMode.norm, None, None), (RandomMode.uniform, 0, 0)]

requires_grads = [
    False,
    True,
]


class TestNeg:

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("mode, max_value, min_value", data_distributions)
    @pytest.mark.parametrize("requires_grad", requires_grads)
    def test_neg(self, shape, mode, min_value, max_value, dtype, requires_grad):
        x_cpu, x_supa = create_random_tensor(
            shape,
            dtype=dtype,
            min_value=min_value,
            max_value=max_value,
            requires_grad=requires_grad,
            mode=mode,
        )
        y_cpu = torch.neg(x_cpu)
        y_supa = torch.neg(x_supa)
        if not requires_grad:
            assert_allclose(y_cpu, y_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        else:
            cpu_grad, supa_grad = create_random_tensor(
                y_cpu.shape,
                dtype=dtype,
                min_value=min_value,
                max_value=max_value,
                requires_grad=False,
                mode=mode,
            )
            y_cpu.backward(cpu_grad)
            x_cpu_grad = x_cpu.grad

            y_supa.backward(supa_grad)
            x_supa_grad = x_supa.grad
            assert_allclose(
                x_cpu_grad, x_supa_grad, rtol=1e-5, atol=5e-5, equal_nan=True
            )

    @pytest.mark.parametrize("scalar_value", scalar_values)
    @pytest.mark.parametrize("scalar_grad_value", scalar_grad_values)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("requires_grad", [False])
    def test_scalar_neg(self, scalar_value, scalar_grad_value, dtype, requires_grad):
        cpu_device = torch.device("cpu")
        supa_device = torch.device("supa")
        x_cpu = torch.tensor(scalar_value, requires_grad=requires_grad).to(cpu_device)
        x_supa = torch.tensor(scalar_value, requires_grad=requires_grad).to(supa_device)
        y_cpu = torch.neg(x_cpu)
        y_supa = torch.neg(x_supa)
        if not requires_grad:
            assert_allclose(y_cpu, y_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        else:
            cpu_grad = torch.tensor(scalar_grad_value, requires_grad=False).to(
                cpu_device
            )
            supa_grad = torch.tensor(scalar_grad_value, requires_grad=False).to(
                supa_device
            )
            y_cpu.backward(cpu_grad)
            x_cpu_grad = x_cpu.grad

            y_supa.backward(supa_grad)
            x_supa_grad = x_supa.grad
            assert_allclose(
                x_cpu_grad, x_supa_grad, rtol=1e-5, atol=5e-5, equal_nan=True
            )


class TestSign:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_sign(self):
        b_cpu = torch.sign(
            torch.tensor(
                [
                    -5.0,
                    -3.4028235e-38,
                    -0.0,
                    3.4028235e-38,
                    5.0,
                    float("nan"),
                    float("inf"),
                    float("-inf"),
                ]
            )
        )
        b_supa = torch.sign(
            torch.tensor(
                [
                    -5.0,
                    -3.4028235e-38,
                    -0.0,
                    3.4028235e-38,
                    5.0,
                    float("nan"),
                    float("inf"),
                    float("-inf"),
                ]
            ).to("supa")
        )

        assert_allclose(b_cpu, b_supa, rtol=1e-5, atol=5e-5)


class TestSignBit:

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_signbit(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        y_cpu = torch.signbit(x_cpu)
        y_supa = torch.signbit(x_supa)

        assert_allclose(y_cpu, y_supa, rtol=1e-5, atol=5e-5, equal_nan=True)


class TestSgn:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_sgn(self):
        x_cpu = torch.tensor([3 + 4j, 7 - 24j, 0, 1 + 2j])
        x_supa = torch.tensor([3 + 4j, 7 - 24j, 0, 1 + 2j])
        y_cpu = torch.sgn(x_cpu)
        y_supa = torch.sgn(x_supa)

        assert_allclose(y_cpu, y_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
