# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    RandomMode,
)

shapes = [
    pytest.param(
        (36, 1),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (4, 36, 1),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (16, 36, 2),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((4, 36, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
]

shape_repeat = [
    pytest.param(
        (3, 3),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 1023), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 1025), marks=[pytest.mark.gcuStress]),
]

dim_repeats = [
    (2,),
    (2, 3),
]
repeats = [
    1,
    2,
]
dim = [0, 1]

dtypes = [torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.bool]

FLOAT_RTOL, FLOAT_ATOL = 1e-5, 5e-5
INT_RTOL, INT_ATOL = 1e-6, 1e-6


class TestRepeat:

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dim_repeat", dim_repeats)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_repeat_function(self, shape, dim_repeat, dtype):
        if len(dim_repeat) < len(shape):
            return
        x_cpu, x_supa = create_random_tensor(
            shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
        )

        y_cpu = x_cpu.repeat(dim_repeat)
        y_supa = x_supa.repeat(dim_repeat)

        if dtype == torch.int32:
            assert_allclose(
                y_cpu.to(torch.int64),
                y_supa.cpu().to(torch.int64),
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        elif dtype == torch.bool:
            assert_allclose(
                y_cpu.to(torch.int64),
                y_supa.cpu().to(torch.int64),
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        else:
            assert_allclose(y_cpu, y_supa, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_repeat_multi_dims(self):
        x_cpu = torch.randn([2, 2, 2, 2])
        x_supa = x_cpu.supa()

        x_cpu = x_cpu.unsqueeze(3)
        x_supa = x_supa.unsqueeze(3)

        y_cpu = x_cpu.repeat(1, 1, 1, 4, 1)
        y_supa = x_supa.repeat(1, 1, 1, 4, 1)
        assert_allclose(y_cpu, y_supa, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_repeat(self):
        x_cpu = torch.randn([10])
        x_supa = x_cpu.supa()

        y_cpu = x_cpu.repeat(10)
        y_supa = x_supa.repeat(10)
        assert_allclose(y_cpu, y_supa, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)


class TestRepeatInterleave:

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dim_repeat", repeats)
    @pytest.mark.parametrize("dim", dim)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_repeat_interleave(self, shape, dim_repeat, dim, dtype):
        if dim >= len(shape):
            return
        x_cpu, x_supa = create_random_tensor(
            shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
        )

        y_cpu = x_cpu.repeat_interleave(dim_repeat, dim=dim)
        y_supa = x_supa.repeat_interleave(dim_repeat, dim=dim)

        if dtype == torch.int32:
            assert_allclose(
                y_cpu.to(torch.int64),
                y_supa.cpu().to(torch.int64),
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        elif dtype == torch.bool:
            assert_allclose(
                y_cpu.to(torch.int64),
                y_supa.cpu().to(torch.int64),
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        else:
            assert_allclose(y_cpu, y_supa, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_repeat_interleave_function(self):
        x_cpu = torch.randn([10])
        x_supa = x_cpu.supa()

        y_cpu = x_cpu.repeat_interleave(10)
        y_supa = x_supa.repeat_interleave(10)
        assert_allclose(y_cpu, y_supa, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)

    @pytest.mark.parametrize("shape", shape_repeat)
    @pytest.mark.parametrize("dim", dim)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_repeat_interleave_func(self, shape, dim, dtype):
        if dim >= len(shape):
            return
        x_cpu, x_supa = create_random_tensor(
            shape, dtype=dtype, min_value=-5, max_value=5, mode=RandomMode.uniform
        )

        dim_repeat = torch.arange(1, shape[0] + 1)
        dim_repeat_supa = dim_repeat.clone().supa()
        y_cpu = torch.repeat_interleave(x_cpu, dim_repeat, dim=dim)
        y_supa = torch.repeat_interleave(x_supa, dim_repeat_supa, dim=dim)

        if dtype == torch.int32:
            assert_allclose(
                y_cpu.to(torch.int64),
                y_supa.cpu().to(torch.int64),
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        elif dtype == torch.bool:
            assert_allclose(
                y_cpu.to(torch.int64),
                y_supa.cpu().to(torch.int64),
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        else:
            assert_allclose(y_cpu, y_supa, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)
