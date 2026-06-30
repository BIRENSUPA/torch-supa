# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}

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

keepdims = [True, False]

dtypes = [torch.float32, torch.bfloat16, torch.float16, torch.bool]

torch.manual_seed(10)


@pytest.mark.parametrize("shape", br200_shapes)
@pytest.mark.parametrize("keepdim", keepdims)
@pytest.mark.parametrize("dtype", dtypes)
def test_prod_br200(shape, keepdim, dtype):
    def test_prod(shape, reduce_dim, keepdim, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        y_cpu = torch.prod(x_cpu, reduce_dim, keepdim)
        y_supa = torch.prod(x_supa, reduce_dim, keepdim)

        # bool input dtype will got a long output dtype
        if dtype == torch.bool:
            assert y_cpu.dtype == y_supa.dtype
        else:
            assert_allclose(y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])

    test_prod(shape, -1, keepdim, dtype)


@pytest.mark.parametrize("shape", br200_shapes)
@pytest.mark.parametrize("keepdim", keepdims)
@pytest.mark.parametrize("dtype", dtypes)
def test_nansum_br200(shape, keepdim, dtype):
    def test_nansum(shape, reduce_dim, keepdim, dtype):
        x_cpu = torch.tensor([1.0, 2.0, float("nan"), 4.0])
        x_supa = x_cpu.supa()
        y_cpu = torch.nansum(x_cpu)
        y_supa = torch.nansum(x_supa)

        # bool input dtype will got a long output dtype
        if dtype == torch.bool:
            assert y_cpu.dtype == y_supa.dtype
        else:
            assert_allclose(y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])

    test_nansum(shape, -1, keepdim, dtype)
