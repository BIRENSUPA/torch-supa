# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

shifts = [
    1,
    -1,
]
dims = [0, 1]


class TestRoll:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize("shifts", shifts)
    @pytest.mark.parametrize("dims", dims)
    def test_roll(self, shifts, dims):
        cpu_x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
        cpu_out = torch.roll(cpu_x, shifts=shifts, dims=dims)

        supa_x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
        supa_out = torch.roll(supa_x, shifts=shifts, dims=dims)
        assert_allclose(cpu_out, supa_out, atol=0, rtol=0)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_roll_multidim(self):
        cpu_x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
        cpu_out = torch.roll(cpu_x, shifts=(2, 1), dims=(0, 1))

        supa_x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
        supa_out = torch.roll(supa_x, shifts=(2, 1), dims=(0, 1))
        assert_allclose(cpu_out, supa_out, atol=0, rtol=0)


br200_params = [
    pytest.param(
        (2,),
        (0,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (512, 1024), (0,), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((1023, 511), (0,), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), (0,), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), (0,), marks=[pytest.mark.gcuStress]),
]
dtypes = [torch.float32, torch.bfloat16, torch.float16]


class TestFlip:

    @pytest.mark.parametrize("input_shape, dim_list", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_flip_br200(self, input_shape, dim_list, dtype):
        def test_flip(input_shape, dim_list, dtype):
            input_cpu, input_supa = create_random_tensor(input_shape, dtype=dtype)

            output_cpu = torch.flip(input_cpu, dim_list)
            output_supa = torch.flip(input_supa, dim_list)

            assert_allclose(output_cpu, output_supa.cpu(), rtol=0, atol=0)

        test_flip(input_shape, dim_list, dtype)
