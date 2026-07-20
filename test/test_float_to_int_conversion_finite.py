# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import numpy as np
import pytest
import torch

import torch_supa  # noqa: F401  enables transfer_to_supa patching
import torch_supa.contrib.transfer_to_supa  # noqa: F401


@pytest.mark.regression
def test_float_to_int_conversion_finite():
    """Regression test for SUPA copy kernel float32->int16 out-of-range cast.

    Migrated from pytorch test_tensor_creation_ops.py::
    TestTensorCreationCUDA::test_float_to_int_conversion_finite (cuda + int16 path).

    Fix: added SUPA copy kernel CUDA-compatible float32->int16 conversion for
    out-of-range negative float, so the cast matches the numpy reference (and
    upstream CUDA behavior) instead of producing undefined results.
    """
    device = "supa"
    dtype = torch.int16

    finfo_min = torch.finfo(torch.float).min
    # CUDA int16 path: includes min float (out-of-range for int16) to exercise
    # the out-of-range conversion fix.
    vals = (finfo_min, -2, -1.5, -.5, 0, .5, 1.5, 2)

    refs = torch.from_numpy(np.array(vals, dtype=np.float32).astype(np.int16))
    t = torch.tensor(vals, device=device, dtype=torch.float).to(dtype)
    assert torch.equal(refs, t.cpu())
