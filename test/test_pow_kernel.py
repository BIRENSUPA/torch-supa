# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

br200_params = [
    pytest.param(
        (1, 16),
        True,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (1, 16),
        False,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (512, 1024), True, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param(
        (512, 1024), False, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((1023, 511), True, marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), True, marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), True, marks=[pytest.mark.gcuStress]),
    pytest.param((1023, 511), False, marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), False, marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), False, marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.float32]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


class TestPow:

    @pytest.mark.parametrize("shape, scalar", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_pow_ptwise_br200(self, shape, scalar, dtype):
        def test_pow_ptwise(shape, dtype, scalar):
            torch.manual_seed(0)
            torch.set_printoptions(sci_mode=False)
            x_cpu, x_supa = create_random_tensor(shape, dtype)
            if not scalar:
                y_cpu, y_supa = create_random_tensor(shape, dtype)
            else:
                y_cpu = torch.tensor(2)
                y_supa = torch.tensor(2)

            out_cpu = torch.pow(x_cpu, y_cpu)
            out_supa = torch.pow(x_supa, y_supa)
            x_cpu.pow_(y_cpu)
            x_supa.pow_(y_supa)

            assert_allclose(
                x_cpu, x_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
            )
            assert_allclose(
                out_cpu, out_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
            )
            assert_allclose(
                x_supa.cpu(),
                out_supa,
                atol=ATOL[dtype],
                rtol=RTOL[dtype],
                equal_nan=True,
            )

        test_pow_ptwise(shape, dtype, scalar)
