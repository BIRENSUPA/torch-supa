# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import numpy as np
import pytest
import torch

from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    RandomMode,
)

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

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
    pytest.param(
        (1, 4, 16),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((511, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((513, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((533, 533), marks=[pytest.mark.gcuStress]),
]

# br200_dual_shapes = [
#     ((4, 1, 2), (1, 4, 2)),
# ]

br200_dual_shapes = [
    pytest.param(
        ((4, 1, 2), (1, 4, 2)),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        ((512, 1, 1024), (1, 512, 1024)),
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
]


dims = [0, 1, -1]

keep_dims = [True, False]

dtypes = [
    torch.float32,
    torch.bfloat16,
    torch.float16,
]

params_reduce_bf16_max = [
    pytest.param(
        (3, 4, 2),
        2,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (3, 4, 3),
        2,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (16, 512, 256), 2, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
]


br200_params = [
    pytest.param(
        (2, 8),
        -1,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 1024), -1, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
]

FLOAT_RTOL, FLOAT_ATOL = 1e-5, 5e-5
INT_RTOL, INT_ATOL = 1e-8, 1e-8

RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


class TestMax:

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_max_global_br200(self, shape, dtype):
        def test_maximum_function(shape, dtype):
            cpu_input1, supa_input1 = create_random_tensor(
                shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
            )
            cpu_input2, supa_input2 = create_random_tensor(
                shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
            )

            y_cpu = torch.max(cpu_input1, cpu_input2)
            y_supa = torch.max(supa_input1, supa_input2)

            assert_allclose(y_cpu, y_supa.cpu(), atol=FLOAT_ATOL, rtol=FLOAT_RTOL)

        def test_max_global_method(shape, dtype):
            x_cpu, x_supa = create_random_tensor(
                shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
            )

            y_cpu = x_cpu.max()
            y_supa = x_supa.max()

            assert_allclose(y_cpu, y_supa.cpu(), atol=FLOAT_ATOL, rtol=FLOAT_RTOL)

        def test_max_global_function(shape, dtype):
            x_cpu, x_supa = create_random_tensor(
                shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
            )

            y_cpu = torch.max(x_cpu)
            y_supa = torch.max(x_supa)

            assert_allclose(y_cpu, y_supa.cpu(), atol=FLOAT_ATOL, rtol=FLOAT_RTOL)

        def test_maximum_method(shape, dtype):
            cpu_input1, supa_input1 = create_random_tensor(
                shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
            )
            cpu_input2, supa_input2 = create_random_tensor(
                shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
            )

            y_cpu = cpu_input1.max(cpu_input2)
            y_supa = supa_input1.max(supa_input2)

            assert_allclose(y_cpu, y_supa.cpu(), atol=FLOAT_ATOL, rtol=FLOAT_RTOL)

        def test_maximum_out_function(shape, dtype):
            cpu_input1, supa_input1 = create_random_tensor(
                shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
            )
            cpu_input2, supa_input2 = create_random_tensor(
                shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
            )

            out_cpu, out_supa = create_random_tensor(
                shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
            )
            y_cpu = torch.max(cpu_input1, cpu_input2, out=out_cpu)
            y_supa = torch.max(supa_input1, supa_input2, out=out_supa)

            assert_allclose(out_cpu, out_supa.cpu(), atol=FLOAT_ATOL, rtol=FLOAT_RTOL)
            assert_allclose(y_cpu, y_supa.cpu(), atol=FLOAT_ATOL, rtol=FLOAT_RTOL)

        test_max_global_function(shape, dtype)
        test_max_global_method(shape, dtype)
        test_maximum_function(shape, dtype)
        test_maximum_method(shape, dtype)
        test_maximum_out_function(shape, dtype)

    @pytest.mark.parametrize("shapes", br200_dual_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_maximum_broadcast_function_200(self, shapes, dtype):
        def test_maximum_broadcast_function(shapes, dtype):
            cpu_input1, supa_input1 = create_random_tensor(
                shapes[0],
                dtype=dtype,
                min_value=-5,
                max_value=5,
                mode=RandomMode.uniform,
            )
            cpu_input2, supa_input2 = create_random_tensor(
                shapes[1],
                dtype=dtype,
                min_value=-5,
                max_value=5,
                mode=RandomMode.uniform,
            )

            y_cpu = torch.max(cpu_input1, cpu_input2)
            y_supa = torch.max(supa_input1, supa_input2)

            assert_allclose(y_cpu, y_supa.cpu(), atol=FLOAT_ATOL, rtol=FLOAT_RTOL)

        test_maximum_broadcast_function(shapes, dtype)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("dim", dims)
    @pytest.mark.parametrize("keep_dim", keep_dims)
    def test_max_reduce_function(self, shape, dtype, dim, keep_dim):
        pos_dim = dim if dim >= 0 else (len(shape) + dim)
        if pos_dim < 0 or pos_dim >= len(shape):
            return
        x_cpu, x_supa = create_random_tensor(
            shape, dtype=dtype, min_value=-500, max_value=500, mode=RandomMode.uniform
        )

        y_cpu_value, y_cpu_indices = torch.max(x_cpu, dim, keep_dim)
        y_supa_value, y_supa_indices = torch.max(x_supa, dim, keep_dim)

        y_supa_cpu = y_supa_value.cpu()
        assert_allclose(y_cpu_value, y_supa_cpu, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)
        # Disable indices check, since supa behaves differently from cpu when values tie.
        # assert_allclose(y_cpu_indices, y_supa_indices.cpu(), atol=INT_ATOL, rtol=INT_RTOL)

    @pytest.mark.parametrize("shape, dim", params_reduce_bf16_max)
    @pytest.mark.parametrize("keep_dim", keep_dims)
    def test_max_reduce_function_bf16inbf16out(self, shape, dim, keep_dim):
        pos_dim = dim if dim >= 0 else (len(shape) + dim)
        if pos_dim < 0 or pos_dim >= len(shape):
            return
        x_cpu, x_supa = create_random_tensor(
            shape,
            dtype=torch.bfloat16,
            min_value=-500,
            max_value=500,
            mode=RandomMode.uniform,
        )

        y_cpu_value, y_cpu_indices = torch.max(x_cpu, dim, keep_dim)
        y_supa_value, y_supa_indices = torch.max(x_supa, dim, keep_dim)

        y_supa_cpu = y_supa_value.cpu()
        assert_allclose(y_cpu_value, y_supa_cpu, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)


class TestMin:

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_minimum(self, shape, dtype):
        a_cpu, a_supa = create_random_tensor(
            shape, dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
        )
        b_cpu, b_supa = create_random_tensor(
            shape, dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
        )

        y_cpu = torch.min(a_cpu, b_cpu)
        y_supa = torch.min(a_supa, b_supa)

        assert_allclose(y_cpu, y_supa, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_minimum_br200(self, shape, dtype):
        self.test_minimum(shape, dtype)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_minimum_2(self, shape, dtype):
        cpu_input1, supa_input1 = create_random_tensor(
            shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
        )
        cpu_input2, supa_input2 = create_random_tensor(
            shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
        )

        y_cpu = cpu_input1.min(cpu_input2)
        y_supa = supa_input1.min(supa_input2)

        assert_allclose(y_cpu, y_supa.cpu(), rtol=FLOAT_RTOL, atol=FLOAT_ATOL)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_minimum_3(self, shape, dtype):
        cpu_input1, supa_input1 = create_random_tensor(
            shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
        )
        cpu_input2, supa_input2 = create_random_tensor(
            shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
        )

        out_cpu, out_supa = create_random_tensor(
            shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
        )

        y_cpu = torch.min(cpu_input1, cpu_input2, out=out_cpu)
        y_supa = torch.min(supa_input1, supa_input2, out=out_supa)

        assert_allclose(out_cpu, out_supa.cpu(), rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
        assert_allclose(y_cpu, y_supa.cpu(), rtol=FLOAT_RTOL, atol=FLOAT_ATOL)

    @pytest.mark.parametrize("shapes", br200_dual_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_minimum_broadcast(self, shapes, dtype):
        a_cpu, a_supa = create_random_tensor(
            shapes[0], dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
        )
        b_cpu, b_supa = create_random_tensor(
            shapes[1], dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
        )

        y_cpu = torch.min(a_cpu, b_cpu)
        y_supa = torch.min(a_supa, b_supa)

        assert_allclose(y_cpu, y_supa, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_min_global(self, shape, dtype):
        a_cpu, a_supa = create_random_tensor(
            shape, dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
        )

        y_cpu = torch.min(a_cpu)
        y_supa = torch.min(a_supa)

        assert_allclose(y_cpu, y_supa, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_min_br200(self, shape, dtype):
        self.test_min_global(shape, dtype)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_min_global_2(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(
            shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
        )

        y_cpu = x_cpu.min()
        y_supa = x_supa.min()

        assert_allclose(y_cpu, y_supa.cpu(), rtol=FLOAT_RTOL, atol=FLOAT_ATOL)

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("dim", dims)
    @pytest.mark.parametrize("keep_dim", keep_dims)
    def test_min_indices(self, shape, dtype, dim, keep_dim):
        pos_dim = dim if dim >= 0 else (len(shape) + dim)
        if pos_dim < 0 or pos_dim >= len(shape):
            return
        a_cpu, a_supa = create_random_tensor(
            shape, dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
        )

        y_cpu, indices_cpu = torch.min(a_cpu, dim, keep_dim)
        y_supa, indices_supa = torch.min(a_supa, dim, keep_dim)

        assert_allclose(y_cpu, y_supa, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
        # assert_allclose(indices_cpu, indices_supa.cpu(), rtol = INT_RTOL, atol = INT_ATOL)


class TestMean:

    @pytest.mark.parametrize("shape, dim", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("keep_dim", keep_dims)
    def test_mean(self, shape, dim, keep_dim, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)

        y_cpu = torch.mean(x_cpu, dim, keep_dim)
        y_supa = torch.mean(x_supa, dim, keep_dim)
        assert_allclose(y_cpu, y_supa, atol=5e-5, rtol=1e-5)

    @pytest.mark.parametrize("shape, dim", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("keep_dim", keep_dims)
    def test_mean_br200(self, shape, dim, keep_dim, dtype):
        self.test_mean(shape, dim, keep_dim, dtype)

    specialparam = [
        [[], []],
        [[1], []],
    ]

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize("shape, dim", specialparam)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("keep_dim", keep_dims)
    def test_mean_special(self, shape, dim, keep_dim, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)

        y_cpu = torch.mean(x_cpu, dim, keep_dim)
        y_supa = torch.mean(x_supa, dim, keep_dim)
        assert_allclose(y_cpu, y_supa, atol=5e-5, rtol=1e-5)


atol_norm = 4e-3


class TestNorm:

    l = [1.0, 2.0]

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_norm2(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        y_cpu = torch.norm(x_cpu, 0)
        y_supa = torch.norm(x_supa, 0)
        assert_allclose(y_cpu, y_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_norm_l2(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        y_cpu = torch.norm(x_cpu, 2.0)
        y_supa = torch.norm(x_supa, 2.0)
        assert_allclose(y_cpu, y_supa.cpu(), atol=atol_norm, rtol=RTOL[dtype])

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_norm_l2_bf16inbf16out(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        y_cpu = torch.norm(x_cpu, p=2).to(dtype)
        y_supa = torch.norm(x_supa, p=2, dtype=dtype)
        assert_allclose(y_cpu, y_supa.cpu(), atol=atol_norm, rtol=RTOL[dtype])

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_norm_l1(self, shape, dtype):
        torch.manual_seed(0)
        np.random.seed(0)
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        y_cpu = torch.norm(x_cpu, 1.0)
        y_supa = torch.norm(x_supa, 1.0)
        assert_allclose(y_cpu, y_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("l_value", l)
    def test_norm_backward(self, shape, dtype, l_value):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)
        y_cpu = torch.norm(x_cpu, l_value)
        y_supa = torch.norm(x_supa, l_value)
        assert_allclose(y_cpu, y_supa.cpu(), atol=atol_norm, rtol=RTOL[dtype])

        cpu_grad, supa_grad = create_random_tensor(
            y_cpu.shape, dtype=dtype, requires_grad=False
        )
        y_cpu.backward(cpu_grad)
        input_cpu_grad = x_cpu.grad

        y_supa.backward(supa_grad)
        input_supa_grad = x_supa.grad
        assert_allclose(
            input_cpu_grad,
            input_supa_grad,
            atol=ATOL[dtype],
            rtol=RTOL[dtype],
            equal_nan=False,
        )

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_norm_backward_l2(self, shape, dtype, l_value=2.0):
        x_cpu, x_supa = create_random_tensor((shape), dtype=dtype, requires_grad=True)
        y_cpu = torch.norm(x_cpu, l_value)
        y_supa = torch.norm(x_supa, l_value)
        assert_allclose(y_cpu, y_supa.cpu(), atol=atol_norm, rtol=RTOL[dtype])

        cpu_grad, supa_grad = create_random_tensor(
            y_cpu.shape, dtype=dtype, requires_grad=False
        )
        y_cpu.backward(cpu_grad)
        input_cpu_grad = x_cpu.grad

        y_supa.backward(supa_grad)
        input_supa_grad = x_supa.grad
        assert_allclose(
            input_cpu_grad,
            input_supa_grad,
            atol=ATOL[dtype],
            rtol=RTOL[dtype],
            equal_nan=False,
        )


class TestAMax:
    dims = [0, 1, (0, 1)]
    keep_dims = [True, False]

    @pytest.mark.parametrize("shape, dim", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("keep_dim", keep_dims)
    def test_amax(self, shape, dtype, dim, keep_dim):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        y_cpu = torch.amax(x_cpu, dim, keep_dim)
        y_supa = torch.amax(x_supa, dim, keep_dim)

        assert_allclose(y_cpu, y_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])


class TestAMin:

    keep_dims = [True, False]

    @pytest.mark.parametrize("shape, dim", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("keep_dim", keep_dims)
    def test_amin(self, shape, dtype, dim, keep_dim):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        y_cpu = torch.amin(x_cpu, dim, keep_dim)
        y_supa = torch.amin(x_supa, dim, keep_dim)

        assert_allclose(y_cpu, y_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])


class TestArgmax:

    @pytest.mark.parametrize("src_shape, dim", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_Argmax(self, src_shape, dim, dtype, keepdim=False):
        cpu_input, supa_input = create_random_tensor(src_shape, dtype=dtype)
        cpu_res = cpu_input.argmax(dim, keepdim)
        supa_res = supa_input.argmax(dim, keepdim)
        assert_allclose(
            cpu_res, supa_res, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )


class TestArgmin:

    @pytest.mark.parametrize("src_shape, dim", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_Argmin(self, src_shape, dim, dtype, keepdim=False):
        cpu_input, supa_input = create_random_tensor(src_shape, dtype=dtype)
        cpu_res = cpu_input.argmin(dim, keepdim)
        supa_res = supa_input.argmin(dim, keepdim)
        assert_allclose(
            cpu_res, supa_res, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )


class TestStdVar:

    scalar_grad_values = [0.888, -0.888]

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("scalar_grad_value", scalar_grad_values)
    def test_std(self, shape, scalar_grad_value, dtype, requires_grad=True):
        cpu_device = torch.device("cpu")
        supa_device = torch.device("supa")
        x_cpu, x_supa = create_random_tensor(
            shape, dtype=dtype, requires_grad=requires_grad
        )
        y_cpu = torch.std(x_cpu)
        y_supa = torch.std(x_supa)
        if not requires_grad:
            assert_allclose(y_cpu, y_supa, rtol=1.3 * 1e-6, atol=1e-5, equal_nan=True)
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
                x_cpu_grad, x_supa_grad, rtol=1.3 * 1e-6, atol=1e-5, equal_nan=True
            )

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("scalar_grad_value", scalar_grad_values)
    def test_var(self, shape, scalar_grad_value, dtype, requires_grad=True):
        cpu_device = torch.device("cpu")
        supa_device = torch.device("supa")
        x_cpu, x_supa = create_random_tensor(
            shape, dtype=dtype, requires_grad=requires_grad
        )
        y_cpu = torch.var(x_cpu)
        y_supa = torch.var(x_supa)
        if not requires_grad:
            assert_allclose(y_cpu, y_supa, rtol=1.3 * 1e-6, atol=1e-5, equal_nan=True)
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
                x_cpu_grad, x_supa_grad, rtol=1.3 * 1e-6, atol=1e-5, equal_nan=True
            )
