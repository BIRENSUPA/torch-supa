# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose


RTOL = {torch.float32: 1e-5}
ATOL = {torch.float32: 1e-6}


NORMALIZE_FUSED_CASES = [
    pytest.param((7, 48), marks=[pytest.mark.sanity, pytest.mark.gcuSmoke]),
    pytest.param((8, 64), marks=[pytest.mark.sanity, pytest.mark.gcuSmoke]),
    pytest.param((9, 96), marks=[pytest.mark.gcuSanity]),
    pytest.param((31, 128), marks=[pytest.mark.gcuSanity]),
    pytest.param((5, 192), marks=[pytest.mark.gcuSanity]),
    pytest.param((8, 256), marks=[pytest.mark.gcuSanity]),
    pytest.param((11, 384), marks=[pytest.mark.gcuStress]),
    pytest.param((8, 512), marks=[pytest.mark.gcuStress]),
]


@pytest.mark.parametrize("shape", NORMALIZE_FUSED_CASES)
def test_normalize_fused(shape):
    torch.manual_seed(20260604)
    x_cpu = torch.randn(shape, device="cpu", dtype=torch.float32)
    x_supa = x_cpu.to("supa").contiguous()

    y_cpu = torch.nn.functional.normalize(x_cpu, p=2.0, dim=-1, eps=1.0e-12)
    y_supa = torch.normalize(x_supa, p=2.0, dim=-1, eps=1.0e-12)

    assert_allclose(y_cpu, y_supa.cpu(), atol=ATOL[torch.float32], rtol=RTOL[torch.float32])
