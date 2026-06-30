# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    RandomMode,
)

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

shapes_4d = [
    pytest.param(
        (2, 8, 8, 8),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (16, 4, 4, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((16, 5, 4, 1024), marks=[pytest.mark.gcuStress]),
    pytest.param((16, 7, 4, 1024), marks=[pytest.mark.gcuStress]),
    pytest.param((15, 81, 23, 1023), marks=[pytest.mark.gcuStress]),
]

shapes_5d = [
    pytest.param(
        (2, 8, 8, 8, 2),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (16, 4, 4, 4, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((16, 5, 4, 3, 1024), marks=[pytest.mark.gcuStress]),
    pytest.param((16, 7, 4, 4, 1024), marks=[pytest.mark.gcuStress]),
    pytest.param((15, 81, 23, 5, 1023), marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.float32]

mode = ["bilinear"]

padding_mode = [
    "zeros",
]


class TestGridSamplerOp:

    @pytest.mark.parametrize("shape", shapes_4d)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("mode", mode)
    @pytest.mark.parametrize("padding_mode", padding_mode)
    def test_grid_sampler_2d_fwd(self, shape, dtype, mode, padding_mode):
        input_cpu, input_supa = create_random_tensor(
            shape,
            dtype=dtype,
            min_value=-1,
            max_value=1,
            requires_grad=False,
            mode=RandomMode.uniform,
        )
        grid_shape = (shape[0], shape[2], shape[3], 2)
        grid_cpu, grid_supa = create_random_tensor(
            grid_shape,
            dtype=dtype,
            min_value=-1,
            max_value=1,
            requires_grad=False,
            mode=RandomMode.uniform,
        )
        output_cpu = torch.nn.functional.grid_sample(
            input_cpu, grid_cpu, mode, padding_mode
        )
        output_supa = torch.nn.functional.grid_sample(
            input_supa, grid_supa, mode, padding_mode
        )

        assert_allclose(output_cpu, output_supa, rtol=1e-2, atol=1e-3, equal_nan=True)

    @pytest.mark.parametrize("shape", shapes_4d)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("mode", mode)
    @pytest.mark.parametrize("padding_mode", padding_mode)
    def test_grid_sampler_2d_bwd(self, shape, dtype, mode, padding_mode):
        input_cpu, input_supa = create_random_tensor(
            shape,
            dtype=dtype,
            min_value=-1,
            max_value=1,
            requires_grad=True,
            mode=RandomMode.uniform,
        )
        grid_shape = (shape[0], shape[2], shape[3], 2)
        grid_cpu, grid_supa = create_random_tensor(
            grid_shape,
            dtype=dtype,
            min_value=-1,
            max_value=1,
            requires_grad=False,
            mode=RandomMode.uniform,
        )
        output_cpu = torch.nn.functional.grid_sample(
            input_cpu, grid_cpu, mode, padding_mode
        )
        output_supa = torch.nn.functional.grid_sample(
            input_supa, grid_supa, mode, padding_mode
        )

        cpu_grad, supa_grad = create_random_tensor(
            output_cpu.shape, dtype=dtype, requires_grad=False
        )
        output_cpu.backward(cpu_grad)
        output_supa.backward(supa_grad)

        assert_allclose(cpu_grad, supa_grad, rtol=1e-2, atol=1e-3, equal_nan=True)

    @pytest.mark.parametrize("shape", shapes_5d)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("mode", mode)
    @pytest.mark.parametrize("padding_mode", padding_mode)
    def test_grid_sampler_3d_fwd(self, shape, dtype, mode, padding_mode):
        input_cpu, input_supa = create_random_tensor(
            shape,
            dtype=dtype,
            min_value=-1,
            max_value=1,
            requires_grad=False,
            mode=RandomMode.uniform,
        )
        grid_shape = (shape[0], shape[2], shape[3], shape[4], 3)
        grid_cpu, grid_supa = create_random_tensor(
            grid_shape,
            dtype=dtype,
            min_value=-1,
            max_value=1,
            requires_grad=False,
            mode=RandomMode.uniform,
        )
        output_cpu = torch.nn.functional.grid_sample(
            input_cpu, grid_cpu, mode, padding_mode
        )
        output_supa = torch.nn.functional.grid_sample(
            input_supa, grid_supa, mode, padding_mode
        )

        assert_allclose(output_cpu, output_supa, rtol=1e-2, atol=1e-3, equal_nan=True)

    @pytest.mark.parametrize("shape", shapes_5d)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("mode", mode)
    @pytest.mark.parametrize("padding_mode", padding_mode)
    def test_grid_sampler_3d_bwd(self, shape, dtype, mode, padding_mode):
        input_cpu, input_supa = create_random_tensor(
            shape,
            dtype=dtype,
            min_value=-1,
            max_value=1,
            requires_grad=True,
            mode=RandomMode.uniform,
        )
        grid_shape = (shape[0], shape[2], shape[3], shape[4], 3)
        grid_cpu, grid_supa = create_random_tensor(
            grid_shape,
            dtype=dtype,
            min_value=-1,
            max_value=1,
            requires_grad=False,
            mode=RandomMode.uniform,
        )
        output_cpu = torch.nn.functional.grid_sample(
            input_cpu, grid_cpu, mode, padding_mode
        )
        output_supa = torch.nn.functional.grid_sample(
            input_supa, grid_supa, mode, padding_mode
        )

        cpu_grad, supa_grad = create_random_tensor(
            output_cpu.shape, dtype=dtype, requires_grad=False
        )
        output_cpu.backward(cpu_grad)
        output_supa.backward(supa_grad)

        assert_allclose(cpu_grad, supa_grad, rtol=1e-2, atol=1e-3, equal_nan=True)
