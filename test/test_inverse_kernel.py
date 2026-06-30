# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

input_shapes = [
    pytest.param((3, 3), marks=[
        pytest.mark.sanity,
        pytest.mark.gcuSmoke,
        pytest.mark.regression,
        pytest.mark.gcuSanity,
        pytest.mark.gcuStress,]),
    pytest.param((1, 6, 3, 3), marks=[
        pytest.mark.sanity,
        pytest.mark.gcuSmoke,
        pytest.mark.regression,
        pytest.mark.gcuSanity,
        pytest.mark.gcuStress,]),
]

pytest.mark.sanity,
pytest.mark.regression,
class TestInverseKernel:
    @pytest.mark.parametrize("shape", input_shapes)
    def test_linalg_inv(self, shape):
        input_cpu, input_supa = create_random_tensor(shape, dtype=torch.float32, requires_grad=False)
        out_cpu = torch.linalg.inv(input_cpu)
        out_supa = torch.linalg.inv(input_supa)
        assert_allclose(out_cpu, out_supa, rtol=1e-4, atol=1e-4, equal_nan=True)

    @pytest.mark.parametrize("shape", input_shapes)
    def test_inverse(self, shape):
        input_cpu, input_supa = create_random_tensor(shape, dtype=torch.float32, requires_grad=False)
        out_cpu = torch.empty_like(input_cpu)
        out_supa = torch.empty_like(input_supa)
        torch.inverse(input_cpu, out=out_cpu)
        torch.inverse(input_supa, out=out_supa)
        assert_allclose(out_cpu, out_supa, rtol=1e-4, atol=1e-4, equal_nan=True)
