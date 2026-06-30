# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.


# noqa
import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


br200_shapes = [
    pytest.param(
        (2, 6, 12, 12),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (16, 32, 128, 128), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((17, 54, 256, 256), marks=[pytest.mark.gcuStress]),
]

br200_shapes_5d = [
    pytest.param(
        (2, 6, 12, 12, 12),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (16, 32, 64, 64, 64), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
]

kernel = [
    (2),
]

stride = [
    (2),
]


dtypes = [torch.float32, torch.bfloat16, torch.float16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


class TestMaxUnpoolMethod:

    @pytest.mark.parametrize("shape", br200_shapes)
    @pytest.mark.parametrize("kernel", kernel)
    @pytest.mark.parametrize("stride", stride)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_max_unpool2d(self, shape, kernel, stride, dtype):

        in_cpu, in_supa = create_random_tensor(shape, dtype=dtype, requires_grad=False)
        pool = torch.nn.MaxPool2d(kernel, stride=stride, return_indices=True)
        unpool = torch.nn.MaxUnpool2d(kernel, stride=stride)

        pool_cpu, indices = pool(in_cpu.float())
        pool_supa, indices_supa = pool(in_supa)

        assert_allclose(
            pool_cpu.to(dtype), pool_supa, rtol=1e-5, atol=5e-5, equal_nan=True
        )
        assert_allclose(indices, indices_supa, rtol=1e-5, atol=5e-5, equal_nan=True)

        if dtype != torch.bfloat16:
            out_cpu = unpool(pool_cpu.float(), indices)
            out_supa = unpool(pool_supa, indices_supa)

            assert_allclose(
                out_cpu.to(dtype), out_supa, rtol=1e-5, atol=5e-5, equal_nan=True
            )

    @pytest.mark.parametrize("shape", br200_shapes_5d)
    @pytest.mark.parametrize("kernel", kernel)
    @pytest.mark.parametrize("stride", stride)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_max_unpool3d(self, shape, kernel, stride, dtype):

        in_cpu, in_supa = create_random_tensor(shape, dtype=dtype, requires_grad=False)
        pool = torch.nn.MaxPool3d(kernel, stride=stride, return_indices=True)
        unpool = torch.nn.MaxUnpool3d(kernel, stride=stride)

        pool_cpu, indices = pool(in_cpu.float())
        pool_supa, indices_supa = pool(in_supa)

        assert_allclose(
            pool_cpu.to(dtype), pool_supa, rtol=1e-5, atol=5e-5, equal_nan=True
        )
        assert_allclose(indices, indices_supa, rtol=1e-5, atol=5e-5, equal_nan=True)

        if dtype != torch.bfloat16:
            out_cpu = unpool(pool_cpu.float(), indices)
            out_supa = unpool(pool_supa, indices_supa)

            assert_allclose(
                out_cpu.to(dtype), out_supa, rtol=1e-5, atol=5e-5, equal_nan=True
            )
