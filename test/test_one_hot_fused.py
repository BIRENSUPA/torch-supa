# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose


# Correctness is mode-agnostic: both BRTB_ENABLE_NATIVE_OP=0 (fused SUPA op) and
# =1 (native zeros+scatter composite) must match the CPU reference. Run this file
# under both switch values (separate processes) to cover both dispatch paths.
ONE_HOT_FUSED_CASES = [
    pytest.param((8,), 4, marks=[pytest.mark.sanity, pytest.mark.gcuSmoke]),
    pytest.param((64,), 32, marks=[pytest.mark.sanity, pytest.mark.gcuSmoke]),
    pytest.param((1000,), 10, marks=[pytest.mark.gcuSanity]),
    pytest.param((4, 16), 7, marks=[pytest.mark.gcuSanity]),
    pytest.param((1024,), 128, marks=[pytest.mark.gcuSanity]),
    pytest.param((262144,), 32, marks=[pytest.mark.gcuStress]),
]


@pytest.mark.parametrize("shape, num_classes", ONE_HOT_FUSED_CASES)
def test_one_hot_fused(shape, num_classes):
    torch.manual_seed(20260605)
    idx_cpu = torch.randint(0, num_classes, shape, dtype=torch.int64)
    idx_supa = idx_cpu.to("supa").contiguous()

    out_cpu = torch.nn.functional.one_hot(idx_cpu, num_classes=num_classes)
    out_supa = torch.nn.functional.one_hot(idx_supa, num_classes=num_classes)

    assert out_supa.dtype == torch.int64
    assert list(out_supa.shape) == list(shape) + [num_classes]
    assert_allclose(out_cpu, out_supa.cpu(), atol=0, rtol=0)
