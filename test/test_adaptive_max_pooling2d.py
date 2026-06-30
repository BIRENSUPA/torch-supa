# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
import torch.nn as nn
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

params = [
    pytest.param(
        (1, 32, 3, 3),
        (1, 32, 1, 1),
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
        (2, 16, 7, 7),
        (2, 16, 1, 1),
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
        (1, 32, 3, 3),
        (1, 32, 1, 1),
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
        (2, 16, 7, 7),
        (2, 16, 1, 1),
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
        (1, 32, 3, 3),
        (1, 32, 1, 1),
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
        (2, 16, 7, 7),
        (2, 16, 1, 1),
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
        (8, 32, 17, 17),
        (8, 32, 9, 9),
        torch.float32,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (15, 64, 13, 13), (15, 64, 5, 5), torch.float32, marks=[pytest.mark.gcuStress]
    ),
    pytest.param(
        (15, 63, 15, 15), (15, 63, 7, 7), torch.float32, marks=[pytest.mark.gcuStress]
    ),
    pytest.param(
        (15, 65, 19, 19), (15, 65, 10, 10), torch.float32, marks=[pytest.mark.gcuStress]
    ),
]

dtypes = [torch.float32, torch.bfloat16, torch.float16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


class TestAdaptiveMaxPool2dMethod:
    @pytest.mark.parametrize("input_shape, output_shape, dtype", params)
    def test_adaptive_maxpool2d(self, input_shape, output_shape, dtype):
        x_cpu, x_supa = create_random_tensor(
            input_shape, dtype=dtype, requires_grad=True
        )
        cpu_grad, supa_grad = create_random_tensor(
            output_shape, dtype=dtype, requires_grad=True
        )

        adaptive_maxpool2d_cpu = nn.AdaptiveMaxPool2d(
            (output_shape[2], output_shape[3])
        )
        y_cpu = adaptive_maxpool2d_cpu(x_cpu)
        y_cpu.backward(cpu_grad)

        adp_avg_pool_supa = adaptive_maxpool2d_cpu.to(supa_device)
        y_supa = adp_avg_pool_supa(x_supa)
        y_supa.backward(supa_grad)

        assert_allclose(y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])
        assert_allclose(x_cpu.grad, x_supa.grad, atol=ATOL[dtype], rtol=RTOL[dtype])
