# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

br200_params = [
    pytest.param(
        (2, 4),
        1,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 1024), 1, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), 0, marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), 1, marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), 0, marks=[pytest.mark.gcuStress]),
]

dtypes = [
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]


class TestCat:

    @pytest.mark.parametrize("shape, dim", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_cat_br200(self, shape, dtype, dim):
        def test_cat(shape, dtype, dim):
            cpu_in_a, supa_in_a = create_random_tensor(shape, dtype=dtype)
            cpu_in_b, supa_in_b = create_random_tensor(shape, dtype=dtype)

            cpu_out = torch.cat((cpu_in_a, cpu_in_b), dim)

            supa_out = torch.cat((supa_in_a, supa_in_b), dim)
            assert_allclose(cpu_out, supa_out, atol=0, rtol=0)

        test_cat(shape, dtype, dim)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_view_as_complex(self):
        x_cpu = torch.randn(4, 2)
        y_cpu = torch.view_as_complex(x_cpu)

        x_cuda = x_cpu.supa()
        y_cuda = torch.view_as_complex(x_cuda)

        assert_allclose(y_cpu, y_cuda, atol=0, rtol=0)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_view_as_real(self):
        x_cpu = torch.randn(4, 2)
        y_cpu = torch.view_as_complex(x_cpu)
        y_cpu = torch.view_as_real(y_cpu)

        x_cuda = x_cpu.supa()
        y_cuda = torch.view_as_complex(x_cuda)
        y_cuda = torch.view_as_real(y_cuda)

        assert_allclose(y_cpu, y_cuda, atol=0, rtol=0)
