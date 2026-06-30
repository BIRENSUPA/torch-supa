# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

params = [
    pytest.param(
        (1, 4),
        (3, 4),
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
        (1, 4),
        (3, 4),
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
        (512, 1024),
        (512, 1024),
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (512, 1024),
        (512, 1024),
        torch.float16,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (1023, 511), (1023, 511), torch.float32, marks=[pytest.mark.gcuStress]
    ),
    pytest.param(
        (1025, 513), (1025, 513), torch.float32, marks=[pytest.mark.gcuStress]
    ),
    pytest.param(
        (1028, 2135), (1028, 2135), torch.float32, marks=[pytest.mark.gcuStress]
    ),
]


@pytest.mark.parametrize("real, imag, dtype", params)
def test_complex(real, imag, dtype):
    real_cpu, real_supa = create_random_tensor(real, dtype=dtype, requires_grad=True)
    imag_cpu, imag_supa = create_random_tensor(imag, dtype=dtype, requires_grad=True)

    y_cpu = torch.complex(real_cpu, imag_cpu)
    y_supa = torch.complex(real_supa, imag_supa)

    assert_allclose(y_cpu.real, y_supa.cpu().real, atol=0, rtol=0)
    assert_allclose(y_cpu.imag, y_supa.cpu().imag, atol=0, rtol=0)


@pytest.mark.parametrize("real, imag, dtype", params)
def test_polar(real, imag, dtype):
    # dtype must be float or double
    dtype = torch.float32
    real_cpu, real_supa = create_random_tensor(real, dtype=dtype, requires_grad=True)
    imag_cpu, imag_supa = create_random_tensor(imag, dtype=dtype, requires_grad=True)

    y_cpu = torch.polar(real_cpu, imag_cpu)
    y_supa = torch.polar(real_supa, imag_supa)

    assert_allclose(y_cpu, y_supa.cpu(), atol=1e-6, rtol=1e-6)
