# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

RTOL = {
    torch.float32: 1e-5,
    torch.bfloat16: 1.6e-2,
    torch.float16: 1e-3,
}
ATOL = {
    torch.float32: 1e-5,
    torch.bfloat16: 1e-5,
    torch.float16: 1e-5,
}

VECTOR_NORM_CASES = [
    ((1, 384, 1, 8, 8), 2, 1, True, torch.float32),
    ((1, 384, 8, 8), 2, 1, True, torch.float32),
    ((128, 5120), 2, 1, False, torch.float32),
    ((32, 128, 64), 2, -1, False, torch.float32),
]


class TestLinalgVectorNorm:
    @pytest.mark.parametrize("shape, ord_value, dim, keepdim, acc_dtype", VECTOR_NORM_CASES)
    def test_linalg_vector_norm(self, shape, ord_value, dim, keepdim, acc_dtype):
        x_cpu, x_supa = create_random_tensor(shape, torch.bfloat16)

        y_cpu = torch.linalg.vector_norm(
            x_cpu,
            ord=ord_value,
            dim=dim,
            keepdim=keepdim,
            dtype=acc_dtype,
        )

        y_supa = torch.linalg.vector_norm(
            x_supa,
            ord=ord_value,
            dim=dim,
            keepdim=keepdim,
            dtype=acc_dtype,
        )

        assert_allclose(y_cpu, y_supa, atol=ATOL[acc_dtype], rtol=RTOL[acc_dtype])
