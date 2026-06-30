# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

br200_shape = [
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
    pytest.param((1024, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.float32, torch.bfloat16, torch.float16]


class TestFmax:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_fmax(self):
        input_a = torch.tensor([9.7, float("nan"), 3.1, float("nan")])
        input_b = torch.tensor([-2.2, 0.5, float("nan"), float("nan")])
        input_a_supa = input_a.to(supa_device)
        input_b_supa = input_b.to(supa_device)
        out_cpu = torch.fmax(input_a, input_b)
        out_supa = torch.fmax(input_a_supa, input_b_supa)
        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_fmin(self):
        input_a = torch.tensor([9.7, float("nan"), 3.1, float("nan")])
        input_b = torch.tensor([-2.2, 0.5, float("nan"), float("nan")])
        input_a_supa = input_a.to(supa_device)
        input_b_supa = input_b.to(supa_device)
        out_cpu = torch.fmin(input_a, input_b)
        out_supa = torch.fmin(input_a_supa, input_b_supa)
        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_maximum(self, shape, dtype):
        in_cpu, in_supa = create_random_tensor(shape, dtype=dtype, requires_grad=False)
        in_cpu2, in_supa2 = create_random_tensor(
            shape, dtype=dtype, requires_grad=False
        )

        output_cpu = torch.maximum(in_cpu, in_cpu2)
        output_supa = torch.maximum(in_supa, in_supa2)

        assert_allclose(output_cpu, output_supa, rtol=1e-5, atol=5e-5, equal_nan=True)

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_minimum(self, shape, dtype):
        in_cpu, in_supa = create_random_tensor(shape, dtype=dtype, requires_grad=False)
        in_cpu2, in_supa2 = create_random_tensor(
            shape, dtype=dtype, requires_grad=False
        )

        output_cpu = torch.minimum(in_cpu, in_cpu2)
        output_supa = torch.minimum(in_supa, in_supa2)

        assert_allclose(output_cpu, output_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
