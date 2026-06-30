# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
)

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


br200_input_shape_dim_index = [
    pytest.param(
        (50,),
        0,
        20,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (10050,),
        0,
        5001,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (1024 * 512,),
        0,
        1024 * 256,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
]

dtypes = [torch.float32, torch.bfloat16, torch.float16]

dtypes_acc = [
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.int32,
    torch.int64,
]

masked_fill_br200_params = [
    pytest.param(
        (4,),
        (4,),
        torch.int64,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (4,),
        (4,),
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
        (4,),
        (4,),
        torch.bfloat16,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (4,),
        (4,),
        torch.float16,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (
            512,
            1024,
        ),
        (
            512,
            1024,
        ),
        torch.int64,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (
            512,
            1024,
        ),
        (
            512,
            1024,
        ),
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (
            512,
            1024,
        ),
        (
            512,
            1024,
        ),
        torch.bfloat16,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (
            512,
            1024,
        ),
        (
            512,
            1024,
        ),
        torch.float16,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (
            512,
            1023,
        ),
        (
            512,
            1023,
        ),
        torch.int64,
        marks=[pytest.mark.gcuStress],
    ),
    pytest.param(
        (
            512,
            1023,
        ),
        (
            512,
            1023,
        ),
        torch.float32,
        marks=[pytest.mark.gcuStress],
    ),
    pytest.param(
        (
            512,
            1023,
        ),
        (
            512,
            1023,
        ),
        torch.bfloat16,
        marks=[pytest.mark.gcuStress],
    ),
    pytest.param(
        (
            512,
            1023,
        ),
        (
            512,
            1023,
        ),
        torch.float16,
        marks=[pytest.mark.gcuStress],
    ),
]

mask_types = [torch.bool]

values = [8, 4, 2]


class TestIndexing:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_index_add_(self):
        x = torch.ones(5, 3)
        x_supa = x.to(supa_device)
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        t_supa = t.to(supa_device)
        index = torch.tensor([0, 4, 2])
        index_supa = index.to(supa_device)
        x.index_add_(0, index, t)
        x_supa.index_add_(0, index_supa, t_supa)
        assert_allclose(x, x_supa, rtol=0, atol=0)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_index_reduce(self):
        x = torch.empty(5, 3).fill_(2)
        x_supa = x.to(supa_device)
        t = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float
        )
        t_supa = t.to(supa_device)
        index = torch.tensor([0, 4, 2, 0])
        index_supa = index.to(supa_device)

        x.index_reduce_(0, index, t, "prod")
        x_supa.index_reduce_(0, index_supa, t_supa, "prod")
        assert_allclose(x, x_supa, rtol=0, atol=0)

    @pytest.mark.parametrize("shape, dim, index", br200_input_shape_dim_index)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_index_select(self, shape, dim, index, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype)
        i_cpu = torch.randint(shape[dim], [index], dtype=torch.int32, device=cpu_device)
        y_cpu = torch.index_select(x_cpu, dim, i_cpu)
        i_supa = i_cpu.to(supa_device)

        y_supa = torch.index_select(x_supa, dim, i_supa)
        assert_allclose(y_cpu, y_supa, atol=0, rtol=0)

    @pytest.mark.parametrize("shape, mask_shape, dtype", masked_fill_br200_params)
    @pytest.mark.parametrize("mask_dtype", mask_types)
    @pytest.mark.parametrize("value", values)
    def test_masked_fill(self, shape, mask_shape, dtype, mask_dtype, value):
        cpu_input, supa_input = create_random_tensor(shape, dtype=dtype)
        cpu_mask, supa_mask = create_random_tensor(
            mask_shape, dtype=mask_dtype, min_value=0, max_value=1
        )

        cpu_input.requires_grad_(False)
        supa_input.requires_grad_(False)

        cpu_input.masked_fill_(cpu_mask, value)
        # call out-of-place version
        supa_output = supa_input.masked_fill(supa_mask, value)
        # call in-place version.
        supa_input.masked_fill_(supa_mask, value)

        # verify result
        assert_allclose(cpu_input, supa_output.cpu(), rtol=0, atol=0)
        assert_allclose(cpu_input, supa_input.cpu(), rtol=0, atol=0)
