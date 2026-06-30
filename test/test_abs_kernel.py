# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    RandomMode,
)

br200_shapes = [
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
    pytest.param((1024, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]

scalar_values = [
    pytest.param(
        0.666,
        marks=[
            pytest.mark.ci_mini,
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        -0.666,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
]

scalar_grad_values = [
    pytest.param(
        0.888,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        -0.888,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
]

# data distribution, min_value, max_value
data_distributions = [(RandomMode.norm, None, None), (RandomMode.uniform, 0, 0)]

dtypes = [torch.float32, torch.bfloat16, torch.float16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}

requires_grads = [
    False,
]


class TestNNMethod:
    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("mode, max_value, min_value", data_distributions)
    @pytest.mark.parametrize("requires_grad", requires_grads)
    def test_abs(self, shape, dtype, mode, max_value, min_value, requires_grad):
        x_cpu, x_supa = create_random_tensor(
            shape,
            dtype=dtype,
            min_value=min_value,
            max_value=max_value,
            requires_grad=requires_grad,
            mode=mode,
        )
        y_cpu = torch.abs(x_cpu)
        y_supa = torch.abs(x_supa)
        if not requires_grad:
            assert_allclose(
                y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
            )
        else:
            cpu_grad, supa_grad = create_random_tensor(
                y_cpu.shape,
                dtype=dtype,
                min_value=min_value,
                max_value=max_value,
                requires_grad=requires_grad,
                mode=mode,
            )
            y_cpu.backward(cpu_grad)
            x_cpu_grad = x_cpu.grad

            y_supa.backward(supa_grad)
            x_supa_grad = x_supa.grad
            assert_allclose(
                x_cpu_grad,
                x_supa_grad,
                atol=ATOL[dtype],
                rtol=RTOL[dtype],
                equal_nan=True,
            )

    @pytest.mark.parametrize("scalar_value", scalar_values)
    @pytest.mark.parametrize("scalar_grad_value", scalar_grad_values)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("requires_grad", [False])
    def test_scalar_abs(self, scalar_value, scalar_grad_value, dtype, requires_grad):
        cpu_device = torch.device("cpu")
        supa_device = torch.device("supa")
        x_cpu = torch.tensor(scalar_value, requires_grad=requires_grad).to(cpu_device)
        x_supa = torch.tensor(scalar_value, requires_grad=requires_grad).to(supa_device)
        y_cpu = torch.abs(x_cpu)
        y_supa = torch.abs(x_supa)
        if not requires_grad:
            assert_allclose(
                y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
            )
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
                x_cpu_grad,
                x_supa_grad,
                atol=ATOL[dtype],
                rtol=RTOL[dtype],
                equal_nan=True,
            )

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("mode, max_value, min_value", data_distributions)
    def test_abs_out(self, shape, dtype, mode, max_value, min_value):
        x_cpu, x_supa = create_random_tensor(
            shape,
            dtype=dtype,
            min_value=min_value,
            max_value=max_value,
            requires_grad=False,
            mode=mode,
        )
        y_cpu = torch.abs(x_cpu)
        y_supa = torch.abs(x_supa)

        x_cpu.abs_()
        x_supa.abs_()

        assert_allclose(
            x_cpu, y_cpu, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
        assert_allclose(
            x_cpu, x_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
        assert_allclose(
            x_supa.cpu(),
            y_supa.cpu(),
            atol=ATOL[dtype],
            rtol=RTOL[dtype],
            equal_nan=True,
        )
