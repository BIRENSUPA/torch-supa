# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import numpy as np
import pytest
import torch

from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    create_torch_tensor_from_np,
    RandomMode,
)

params_2D = [
    # 2D
    [(32, 64), (32, 64), (1)],
    [(32, 64), (32, 64), (0)],
    [(32, 8000), (32, 7000), (0)],
    [(32, 8000), (32, 7000), (1)],
    [(408, 4233), (408, 4233), (0)],
    [(408, 4233), (408, 4233), (1)],
    [(408, 4233), (408, 1), (1)],
    # large
    [(3, 16380), (3, 8193), (1)],
    [(32, 16380), (32, 8193), (1)],
]

params_3D_4D = [
    # 3D
    [(2, 2, 2), (2, 2, 2), (0)],
    [(128, 120, 200), (128, 120, 100), (2)],
    [(8, 160, 80), (8, 160, 70), (2)],
    [(32, 160, 80), (32, 163, 80), (1)],
    [(32, 160, 80), (32, 160, 70), (2)],
    # large
    [(2, 26, 8193), (2, 26, 8193), (0)],
    [(4, 8193, 45), (4, 8193, 45), (1)],
    [(6, 8193, 31), (6, 8193, 31), (1)],
    [(8, 8193, 25), (8, 8193, 31), (2)],
    # 4D
    [(2, 2, 2, 2), (2, 2, 2, 2), (0)],
    [(2, 2, 2, 2), (2, 2, 2, 2), (1)],
    [(3, 4, 5, 6), (3, 4, 5, 6), (1)],
    [(128, 3, 64, 64), (128, 3, 64, 64), (0)],
    [(512, 64, 8, 16), (256, 64, 8, 16), (0)],
    [(16, 64, 20, 71), (16, 129, 20, 71), (1)],
    [(32, 64, 100, 30), (32, 64, 80, 30), (2)],
    [(64, 64, 20, 64), (64, 64, 20, 71), (3)],
]

br200_params_2d = [
    pytest.param(
        (2, 4),
        (2, 4),
        (1),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (32, 8000),
        (32, 7000),
        (1),
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (408, 4233),
        (408, 4233),
        (0),
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param((234, 4233), (234, 4233), (0), marks=[pytest.mark.gcuStress]),
    pytest.param((616, 5136), (616, 5136), (0), marks=[pytest.mark.gcuStress]),
    pytest.param((616, 5136), (616, 5136), (1), marks=[pytest.mark.gcuStress]),
]

br200_params = [
    pytest.param(
        (5,),
        (2,),
        0,
        torch.float32,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (500,),
        (200,),
        0,
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param((2343,), (423,), 0, torch.float32, marks=[pytest.mark.gcuStress]),
    pytest.param((800,), (300,), 0, torch.float32, marks=[pytest.mark.gcuStress]),
    pytest.param((4325,), (255,), 0, torch.float32, marks=[pytest.mark.gcuStress]),
]

params_2D_unique_index = [
    pytest.param(
        (3, 4),
        (3, 1),
        1,
        torch.float32,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (512, 1024),
        (300, 100),
        1,
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (1023, 511), (300, 100), 1, torch.float32, marks=[pytest.mark.gcuStress]
    ),
    pytest.param(
        (1025, 513), (300, 100), 0, torch.float32, marks=[pytest.mark.gcuStress]
    ),
    pytest.param(
        (1028, 2135), (300, 100), 1, torch.float32, marks=[pytest.mark.gcuStress]
    ),
]

params_2D_unique_index = [
    pytest.param(
        (3, 4),
        (3, 1),
        1,
        torch.float32,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (512, 1024),
        (300, 100),
        1,
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (1023, 511), (300, 100), 0, torch.float32, marks=[pytest.mark.gcuStress]
    ),
    pytest.param(
        (1025, 513), (300, 100), 1, torch.float32, marks=[pytest.mark.gcuStress]
    ),
    pytest.param(
        (1028, 2135), (300, 100), 1, torch.float32, marks=[pytest.mark.gcuStress]
    ),
]

params_2D_3D = [
    pytest.param(
        (3, 4),
        (3, 2),
        1,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (512, 1024), (300, 200), 1, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((1023, 511), (300, 200), 1, marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), (300, 200), 1, marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), (300, 200), 1, marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.float32, torch.int32]


@pytest.mark.parametrize("input_shape, output_shape, dim", br200_params_2d)
@pytest.mark.parametrize("dtype", dtypes)
def test_gather_2D_br200(input_shape, output_shape, dim, dtype):
    def test_gather_2D(input_shape, output_shape, dim, dtype):
        input_cpu, input_supa = create_random_tensor(input_shape, dtype=dtype)
        index = np.random.randint(0, input_shape[dim], output_shape)
        index_cpu, index_supa = create_torch_tensor_from_np(index)

        output_cpu = torch.gather(input_cpu, dim, index_cpu)
        output_supa = torch.gather(input_supa, dim, index_supa)

        assert_allclose(output_cpu, output_supa.cpu(), rtol=0, atol=0)

    test_gather_2D(input_shape, output_shape, dim, dtype)


class TestScatter:

    @pytest.mark.parametrize("src_shape, idx_shape, dim, dtype", br200_params)
    def test_scatter(self, src_shape, idx_shape, dim, dtype):
        cpu_input, supa_input = create_random_tensor(idx_shape, dtype=dtype)
        # If index is duplicated, data could be mismatch due to the parallel scatter in GPU.
        cpu_index, supa_index = create_random_tensor(
            idx_shape,
            dtype=torch.int64,
            min_value=0,
            max_value=min(idx_shape) - 1,
            mode=RandomMode.range,
        )
        cpu_res, supa_res = create_random_tensor(src_shape, dtype=dtype)

        cpu_res = cpu_res.scatter_(dim, cpu_index, cpu_input)
        supa_res = supa_res.scatter_(dim, supa_index, supa_input)

        assert_allclose(cpu_res, supa_res, rtol=1e-5, atol=5e-5, equal_nan=True)

    @pytest.mark.parametrize("src_shape, idx_shape, dim, dtype", br200_params)
    def test_scatter_scale(self, src_shape, idx_shape, dim, dtype):
        idx_shape = (1,)
        cpu_input, supa_input = create_random_tensor(idx_shape, dtype=dtype)
        cpu_index, supa_index = create_random_tensor(
            idx_shape, dtype=torch.int64, min_value=0, max_value=min(src_shape) - 1
        )
        cpu_res, supa_res = create_random_tensor(src_shape, dtype=dtype)

        cpu_res = cpu_res.scatter_(dim, cpu_index, cpu_input.item())
        supa_res = supa_res.scatter_(dim, supa_index, supa_input.item())

        assert_allclose(cpu_res, supa_res, rtol=1e-5, atol=1e-5, equal_nan=True)


ATOL_M = {torch.int32: 5e-5, torch.float32: 5e-5, torch.bfloat16: 1e-3}
RTOL_M = {torch.int32: 1e-5, torch.float32: 1e-5, torch.bfloat16: 0.016}


def CreateMatrix2DIndex(idx_shape, src_shape, dim):
    H_INDEX = idx_shape[0]
    W_INDEX = idx_shape[1]
    H_OUTPUT = src_shape[0]
    W_OUTPUT = src_shape[1]
    index_high = 0
    if dim < 0:
        dim += 2
    if dim == 0:
        index_high = H_INDEX
        assert H_INDEX <= H_OUTPUT
    elif dim == 1:
        index_high = W_INDEX
        assert W_INDEX <= W_OUTPUT
    else:
        print("wrong dim:", dim)
        assert 0
    torchindex = np.zeros(idx_shape, dtype=np.int64)
    idx = np.arange(index_high)
    np.random.shuffle(idx)
    if dim == 0:
        for i in range(W_INDEX):
            for j in range(H_INDEX):
                torchindex[j][i] = idx[j]
    elif dim == 1:
        for i in range(H_INDEX):
            for j in range(W_INDEX):
                torchindex[i][j] = idx[j]
    return torchindex


class TestScatterAdd:

    @pytest.mark.parametrize("src_shape, idx_shape, dim, dtype", params_2D_unique_index)
    def test_scatter_add_inplace_unique_index(self, src_shape, idx_shape, dim, dtype):
        torchindex = CreateMatrix2DIndex(idx_shape, src_shape, dim)
        cpu_input, supa_input = create_random_tensor(idx_shape, dtype=dtype)
        # When indices are not unique, the behavior is non-deterministic (one of the values from src will be picked arbitrarily)
        cpu_index, supa_index = create_torch_tensor_from_np(torchindex, False)
        cpu_res, supa_res = create_random_tensor(src_shape, dtype=dtype)
        cpu_res.scatter_add_(dim, cpu_index, cpu_input)
        supa_res.scatter_add_(dim, supa_index, supa_input)
        assert_allclose(
            cpu_res, supa_res, rtol=RTOL_M[dtype], atol=ATOL_M[dtype], equal_nan=True
        )

    @pytest.mark.parametrize("src_shape, idx_shape, dim", params_2D_3D)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_scatter_add_inplace(self, src_shape, idx_shape, dim, dtype):
        cpu_input, supa_input = create_random_tensor(idx_shape, dtype=dtype)
        cpu_index, supa_index = create_random_tensor(
            idx_shape, dtype=torch.int64, min_value=0, max_value=src_shape[dim] - 1
        )
        cpu_res, supa_res = create_random_tensor(src_shape, dtype=dtype)

        if dtype == torch.bfloat16:
            cpu_input = cpu_input.to(torch.float32)
            cpu_res = cpu_res.to(torch.float32)

        cpu_res.scatter_add_(dim, cpu_index, cpu_input)
        supa_res.scatter_add_(dim, supa_index, supa_input)

        if dtype == torch.bfloat16:
            cpu_res = cpu_res.to(torch.bfloat16)
            supa_res = supa_res.to(torch.bfloat16)

        assert_allclose(
            cpu_res, supa_res, rtol=RTOL_M[dtype], atol=ATOL_M[dtype], equal_nan=True
        )

    @pytest.mark.parametrize("src_shape, idx_shape, dim", params_2D_3D)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_scatter_add(self, src_shape, idx_shape, dim, dtype):
        cpu_input, supa_input = create_random_tensor(idx_shape, dtype=dtype)
        cpu_index, supa_index = create_random_tensor(
            idx_shape, dtype=torch.int64, min_value=0, max_value=src_shape[dim] - 1
        )
        cpu_res, supa_res = create_random_tensor(src_shape, dtype=dtype)

        if dtype == torch.bfloat16:
            cpu_input = cpu_input.to(torch.float32)
            cpu_res = cpu_res.to(torch.float32)

        cpu_res = torch.scatter_add(cpu_res, dim, cpu_index, cpu_input)
        supa_res = torch.scatter_add(supa_res, dim, supa_index, supa_input)

        if dtype == torch.bfloat16:
            cpu_res = cpu_res.to(torch.bfloat16)
            supa_res = supa_res.to(torch.bfloat16)

        assert_allclose(
            cpu_res, supa_res, rtol=RTOL_M[dtype], atol=ATOL_M[dtype], equal_nan=True
        )

    @pytest.mark.parametrize("src_shape, idx_shape, dim", params_2D_3D)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_scatter_add_out(self, src_shape, idx_shape, dim, dtype):
        cpu_input, supa_input = create_random_tensor(idx_shape, dtype=dtype)
        cpu_index, supa_index = create_random_tensor(
            idx_shape, dtype=torch.int64, min_value=0, max_value=src_shape[dim] - 1
        )
        cpu_res, supa_res = create_random_tensor(src_shape, dtype=dtype)

        if dtype == torch.bfloat16:
            cpu_input = cpu_input.to(torch.float32)
            cpu_res = cpu_res.to(torch.float32)

        cpu_res = cpu_res.scatter_add(dim, cpu_index, cpu_input)
        supa_res = supa_res.scatter_add(dim, supa_index, supa_input)

        if dtype == torch.bfloat16:
            cpu_res = cpu_res.to(torch.bfloat16)
            supa_res = supa_res.to(torch.bfloat16)

        assert_allclose(
            cpu_res, supa_res, rtol=RTOL_M[dtype], atol=ATOL_M[dtype], equal_nan=True
        )


reduce_mode = [
    "sum",
    "amax",
    "amin",
    "mean",
    "prod",
]


class TestScatterReduce:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize("reduce", reduce_mode)
    def test_scatter_reduce(self, reduce):
        cpu_src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        cpu_index = torch.tensor([0, 1, 0, 1, 2, 1])
        cpu_input = torch.tensor([1.0, 2.0, 3.0, 4.0])
        supa_src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).supa()
        supa_index = torch.tensor([0, 1, 0, 1, 2, 1]).supa()
        supa_input = torch.tensor([1.0, 2.0, 3.0, 4.0]).supa()
        supa_input.scatter_reduce(0, supa_index, supa_src, reduce=reduce)
        cpu_input.scatter_reduce(0, cpu_index, cpu_src, reduce=reduce)
        assert_allclose(cpu_input, supa_input, rtol=1e-5, atol=5e-5, equal_nan=True)
