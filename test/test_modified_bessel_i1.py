# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor


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
dtypes = [torch.float32]

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


class TestModifiedBesselI1Method:

    @pytest.mark.parametrize("shape", br200_shape)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_modified_bessel_i1(self, shape, dtype):
        in_cpu, in_supa = create_random_tensor(shape, dtype=dtype, requires_grad=False)

        output_cpu = torch.special.modified_bessel_i1(in_cpu)
        output_supa = torch.special.modified_bessel_i1(in_supa)

        assert_allclose(output_cpu, output_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
