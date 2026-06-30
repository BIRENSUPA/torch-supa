# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import numpy as np
import pytest
import torch

from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    RandomMode,
)

top0_params = [
    pytest.param(
        0,
        (10,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(0, (512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param(0, (1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param(0, (1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param(0, (1028, 2135), marks=[pytest.mark.gcuStress]),
]

br200_params = [
    pytest.param(
        5,
        (2, 11),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(5, (512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param(5, (1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param(6, (1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param(100, (1028, 2135), marks=[pytest.mark.gcuStress]),
]

backward_params = [
    pytest.param(
        2,
        (10,),
        (2,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        2, (1024,), (2,), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param(2, (1025,), (2,), marks=[pytest.mark.gcuStress]),
]

general_shape_params = [
    pytest.param(
        2,
        (4, 5, 6),
        0,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        2, (4, 512, 1024), 0, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param(2, (2, 1023, 511), 0, marks=[pytest.mark.gcuStress]),
    pytest.param(2, (6, 1025, 513), 0, marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.float32, torch.bfloat16, torch.float16]
# NOTE: Avoid CI fail if there are same value in topk
np.random.seed(0)
torch.manual_seed(0)


class TestTopK:

    @pytest.mark.parametrize("k, shape", top0_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_top0(self, k, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        value_cpu, indices_cpu = torch.topk(x_cpu, k)
        value_supa, indices_supa = torch.topk(x_supa, k)

        assert_allclose(value_cpu, value_supa, atol=1e-5, rtol=1e-5)
        assert_allclose(
            indices_cpu.sort()[0], indices_supa.cpu().sort()[0], atol=0, rtol=0
        )

    @pytest.mark.parametrize("k, shape", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_topk(self, k, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        value_cpu, indices_cpu = torch.topk(x_cpu, k)
        value_supa, indices_supa = torch.topk(x_supa, k)

        assert_allclose(value_cpu, value_supa, atol=1e-5, rtol=1e-5)
        # This test may cause CI to fail if there are same values, especially when dim-0 is very big.
        # assert_allclose(indices_cpu.sort()[0], indices_supa.cpu().sort()[0], atol=0, rtol=0)

    @pytest.mark.parametrize("k, shape, bd_shape", backward_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_topk_backward(self, k, shape, bd_shape, dtype):
        x_cpu, x_supa = create_random_tensor(
            shape,
            min_value=0,
            max_value=shape[0] - 1,
            dtype=dtype,
            requires_grad=True,
            mode=RandomMode.range,
        )
        dx_cpu, dx_supa = create_random_tensor(bd_shape, dtype=dtype)

        value_cpu, indices_cpu = torch.topk(x_cpu, k)
        value_cpu.backward(dx_cpu)

        value_supa, indices_supa = torch.topk(x_supa, k)
        value_supa.backward(dx_supa)

        assert_allclose(x_cpu.grad, x_supa.grad.cpu(), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("k, shape, dim", general_shape_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_topk_general(self, k, shape, dim, dtype):
        np.random.seed(0)
        torch.manual_seed(0)
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)

        value_cpu, indices_cpu = torch.topk(x_cpu, k, dim=dim)
        value_supa, indices_supa = torch.topk(x_supa, k, dim=dim)

        assert_allclose(value_cpu, value_supa, atol=1e-5, rtol=1e-5)
        # This test may cause CI to fail if there are same values
        # assert_allclose(indices_cpu.sort()[0], indices_supa.cpu().sort()[0], atol=0, rtol=0)
