# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import copy

import pytest
import torch
import torch.nn as nn
from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    RandomMode,
)

shapes = [
    pytest.param(
        (2, 6),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 2, 3, 3),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param(
        (16, 2, 256, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]

br200_params = [
    pytest.param(
        [2, 4, 2],
        "sum",
        1.0,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [2, 4, 2],
        "mean",
        1.0,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [2, 4, 2],
        "none",
        1.0,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [128, 512, 50], "sum", 1.0, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param(
        [128, 512, 50],
        "mean",
        1.0,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        [128, 512, 50],
        "none",
        1.0,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        [127, 511, 99], "sum", 1.0, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param(
        [127, 511, 99],
        "mean",
        1.0,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        [127, 511, 99],
        "none",
        1.0,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
]


br200_shapes = [
    pytest.param(
        (4, 2),
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

dtypes = [torch.float32, torch.bfloat16, torch.float16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}

values = [0.5, 1.0, 3.0, 1000, 0.001]


class TestNNMethod:

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("value", values)
    def test_addcmul(self, shape, dtype, value):
        a_cpu, a_supa = create_random_tensor(shape, dtype)
        b_cpu, b_supa = create_random_tensor(shape, dtype)
        c_cpu, c_supa = create_random_tensor(shape, dtype)

        d_cpu = torch.addcmul(a_cpu, b_cpu, c_cpu, value=value)
        d_supa = torch.addcmul(a_supa, b_supa, c_supa, value=value)

        assert_allclose(
            d_cpu, d_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("value", values)
    def test_addcmul_inplace(self, shape, dtype, value):
        a_cpu, a_supa = create_random_tensor(shape, dtype)
        b_cpu, b_supa = create_random_tensor(shape, dtype)
        c_cpu, c_supa = create_random_tensor(shape, dtype)

        a_cpu.addcmul_(b_cpu, c_cpu, value=value)
        a_supa.addcmul_(b_supa, c_supa, value=value)

        assert_allclose(
            a_cpu, a_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("value", values)
    def test_addcdiv(self, shape, dtype, value):
        a_cpu, a_supa = create_random_tensor(
            shape,
            dtype=dtype,
            min_value=0,
            max_value=1,
            requires_grad=True,
            mode=RandomMode.uniform,
        )
        b_cpu, b_supa = create_random_tensor(
            shape,
            dtype=dtype,
            min_value=0,
            max_value=1,
            requires_grad=True,
            mode=RandomMode.uniform,
        )
        c_cpu, c_supa = create_random_tensor(
            shape,
            dtype=dtype,
            min_value=0,
            max_value=1,
            requires_grad=True,
            mode=RandomMode.uniform,
        )

        d_cpu = torch.addcdiv(a_cpu, b_cpu, c_cpu + 1, value=value)
        d_supa = torch.addcdiv(a_supa, b_supa, c_supa + 1, value=value)

        assert_allclose(
            d_cpu, d_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("value", values)
    def test_addcdiv_inplace(self, shape, dtype, value):
        a_cpu, a_supa = create_random_tensor(
            shape,
            dtype=dtype,
            min_value=0,
            max_value=1,
            requires_grad=False,
            mode=RandomMode.uniform,
        )
        b_cpu, b_supa = create_random_tensor(
            shape,
            dtype=dtype,
            min_value=0,
            max_value=1,
            requires_grad=False,
            mode=RandomMode.uniform,
        )
        c_cpu, c_supa = create_random_tensor(
            shape,
            dtype=dtype,
            min_value=0,
            max_value=1,
            requires_grad=False,
            mode=RandomMode.uniform,
        )

        a_cpu.addcdiv_(b_cpu, c_cpu + 1, value=value)
        a_supa.addcdiv_(b_supa, c_supa + 1, value=value)

        assert_allclose(
            a_cpu, a_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )


cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


class TestSmoothL1Loss:

    @pytest.mark.parametrize("shape, reduction, beta", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_smooth_l1_loss_br200(self, shape, reduction, beta, dtype):
        def test_smooth_l1_loss(shape, dtype, reduction, beta):
            x_cpu, x_supa = create_random_tensor(
                shape,
                dtype=dtype,
                min_value=-1,
                max_value=1,
                requires_grad=True,
                mode=RandomMode.uniform,
            )
            target_cpu = (torch.rand(shape) * 2 - 1).to(dtype)  # [-1, 1]
            loss = nn.SmoothL1Loss(reduction=reduction, beta=beta)
            y_cpu = loss(x_cpu, target_cpu)
            target_supa = target_cpu.to(supa_device)
            loss_supa = loss.to(supa_device)

            y_supa = loss_supa(x_supa, target_supa)

            assert_allclose(y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])

        test_smooth_l1_loss(shape, dtype, reduction, beta)

    dtypes = [torch.float32]

    @pytest.mark.parametrize("shape, reduction, beta", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_smooth_l1_loss_bwd_br200(self, shape, reduction, beta, dtype):
        def test_smooth_l1_loss_bwd(shape, dtype, reduction, beta):
            x_cpu, x_supa = create_random_tensor(
                shape,
                dtype=dtype,
                min_value=-1,
                max_value=1,
                requires_grad=True,
                mode=RandomMode.uniform,
            )
            target_cpu = torch.rand(shape) * 2 - 1  # [-1, 1]
            loss = nn.SmoothL1Loss(reduction=reduction, beta=beta).float()
            y_cpu = loss(x_cpu.float(), target_cpu)

            target_supa = target_cpu.to(dtype).to(supa_device)
            loss_supa = copy.deepcopy(loss).to(dtype).to(supa_device)

            y_supa = loss_supa(x_supa, target_supa)

            if reduction == "none":
                cpu_grad_data, supa_grad_data = create_random_tensor(
                    x_cpu.shape,
                    min_value=0,
                    max_value=1,
                    dtype=dtype,
                    mode=RandomMode.uniform,
                )
                y_cpu.backward(cpu_grad_data.float())
                y_supa.backward(supa_grad_data)
            else:
                y_cpu.backward()
                y_supa.backward()

            cpu_grad = x_cpu.grad.clone()
            supa_grad = x_supa.grad

            assert_allclose(
                cpu_grad.to(dtype), supa_grad, atol=ATOL[dtype], rtol=RTOL[dtype]
            )

        test_smooth_l1_loss_bwd(shape, dtype, reduction, beta)


class TestHuberLoss:

    @pytest.mark.parametrize("shape, reduce, beta", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_huber_loss_backward_br200(self, shape, reduce, beta, dtype):
        def test_huber_loss_bwd(shape, dtype, reduction):
            x_cpu, x_supa = create_random_tensor(
                shape,
                dtype=dtype,
                min_value=-1,
                max_value=1,
                requires_grad=True,
                mode=RandomMode.uniform,
            )
            target_cpu = (torch.rand(shape) * 2 - 1).to(dtype)  # [-1, 1]
            loss = nn.HuberLoss(reduction=reduction)
            y_cpu = loss(x_cpu, target_cpu)

            target_supa = target_cpu.to(supa_device)
            loss_supa = copy.deepcopy(loss).to(supa_device)
            y_supa = loss_supa(x_supa, target_supa)

            if reduction == "none":
                cpu_grad_data, supa_grad_data = create_random_tensor(
                    x_cpu.shape,
                    min_value=0,
                    max_value=1,
                    dtype=dtype,
                    mode=RandomMode.uniform,
                )
                y_cpu.backward(cpu_grad_data)
                y_supa.backward(supa_grad_data)
            else:
                y_cpu.backward()
                y_supa.backward()

            cpu_grad = x_cpu.grad.clone()
            supa_grad = x_supa.grad

            assert_allclose(
                cpu_grad.to(dtype), supa_grad, atol=ATOL[dtype], rtol=RTOL[dtype]
            )

        test_huber_loss_bwd(shape, dtype, reduce)

    @pytest.mark.parametrize("shape, reduce, beta", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_huber_loss_br200(self, shape, reduce, beta, dtype):
        def test_huber_loss(shape, dtype, reduction):
            x_cpu, x_supa = create_random_tensor(
                shape,
                dtype=dtype,
                min_value=-1,
                max_value=1,
                requires_grad=True,
                mode=RandomMode.uniform,
            )
            target_cpu = (torch.rand(shape) * 2 - 1).to(dtype)  # [-1, 1]
            loss = nn.HuberLoss(reduction=reduction)
            y_cpu = loss(x_cpu, target_cpu)
            target_supa = target_cpu.to(supa_device)
            loss_supa = copy.deepcopy(loss).to(supa_device)

            y_supa = loss_supa(x_supa, target_supa)

            assert_allclose(y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])

        test_huber_loss(shape, dtype, reduce)


reductions = [
    "none",
    "mean",
    "sum",
]
dtypes = [torch.float32]


class TestMSELoss:

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("reduction", reductions)
    def test_mse_loss_supa(self, shape, dtype, reduction):
        def mse_loss_single(shape, dtype, reduction):
            x_cpu, x_supa = create_random_tensor(
                shape,
                dtype=dtype,
                min_value=-1,
                max_value=1,
                requires_grad=True,
                mode=RandomMode.uniform,
            )
            target_cpu = torch.rand(shape) * 2 - 1  # [-1, 1]
            loss = nn.MSELoss(reduction=reduction).float()

            # NOTE: "mse_cpu" not implemented for 'BFloat16'
            y_cpu = loss(x_cpu.float(), target_cpu)

            target_supa = target_cpu.to(dtype).to(supa_device)
            loss_supa = copy.deepcopy(loss).to(dtype).to(supa_device)

            y_supa = loss_supa(x_supa, target_supa)

            assert_allclose(y_cpu.to(dtype), y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])

        mse_loss_single(shape, dtype, reduction)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("reduction", reductions)
    def test_mse_loss_backward(self, shape, dtype, reduction):
        def mse_loss_bwd_single(shape, dtype, reduction):
            x_cpu, x_supa = create_random_tensor(
                shape,
                dtype=dtype,
                min_value=-1,
                max_value=1,
                requires_grad=True,
                mode=RandomMode.uniform,
            )
            target_cpu = torch.rand(shape) * 2 - 1  # [-1, 1]
            loss = nn.MSELoss(reduction=reduction).float()
            y_cpu = loss(x_cpu.float(), target_cpu)

            target_supa = target_cpu.to(dtype).to(supa_device)
            loss_supa = copy.deepcopy(loss).to(dtype).to(supa_device)
            y_supa = loss_supa(x_supa, target_supa)

            if reduction == "none":
                cpu_grad_data, supa_grad_data = create_random_tensor(
                    y_cpu.shape,
                    min_value=0,
                    max_value=1,
                    dtype=dtype,
                    mode=RandomMode.uniform,
                )
                y_cpu.backward(cpu_grad_data.float())
                y_supa.backward(supa_grad_data)
            else:
                y_cpu.backward()
                y_supa.backward()

            cpu_grad = x_cpu.grad.clone().to(dtype)
            supa_grad = x_supa.grad

            assert_allclose(cpu_grad, supa_grad, atol=ATOL[dtype], rtol=RTOL[dtype])

        mse_loss_bwd_single(shape, dtype, reduction)
