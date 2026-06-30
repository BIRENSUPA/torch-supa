# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
import torch.nn as nn
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

params_scales = [
    pytest.param(
        (1, 1, 8),
        2,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 4, 6),
        4,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 4, 4),
        2,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (1, 1, 8, 8),
        2,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 4, 6, 6),
        4,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 4, 6, 5),
        2,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (1, 1, 8, 8, 2),
        2,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 4, 6, 6, 1),
        4,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 4, 6, 5, 4),
        2,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (1, 512, 1024), 2, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param(
        (2, 4, 256, 256), 4, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param(
        (2, 4, 256, 64, 64), 2, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((1, 1023, 511), 2, marks=[pytest.mark.gcuStress]),
    pytest.param((2, 1025, 513), 4, marks=[pytest.mark.gcuStress]),
    pytest.param((3, 4, 1028, 2135), 2, marks=[pytest.mark.gcuStress]),
]

dtype = [torch.float32]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}

mode = ["nearest", "linear", "bilinear", "bicubic", "trilinear"]

align_corners = [True, False]

params_out = [
    pytest.param(
        (2, 8, 6),
        (12),
        "linear",
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 8, 3),
        (32),
        "bicubic",
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 16, 2),
        (8),
        "nearest",
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 8, 6, 6),
        (12, 12),
        "bilinear",
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 16, 2, 2),
        (8, 16),
        "nearest",
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 16, 1, 1),
        (4, 4),
        "bicubic",
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 8, 6, 6, 2),
        (12, 12, 4),
        "bilinear",
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 8, 3, 3, 3),
        (8, 8, 12),
        "nearest",
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (2, 3, 1, 1, 2),
        (4, 4, 4),
        "trilinear",
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (1, 512, 1024),
        (12,),
        "bilinear",
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (2, 4, 256, 256),
        (8, 16),
        "nearest",
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        (2, 4, 256, 64, 64),
        (4, 4, 4),
        "trilinear",
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param((1, 512, 1022), (12,), "bilinear", marks=[pytest.mark.gcuStress]),
    pytest.param((2, 4, 255, 255), (8, 16), "nearest", marks=[pytest.mark.gcuStress]),
    pytest.param(
        (2, 4, 511, 62, 62), (4, 4, 4), "trilinear", marks=[pytest.mark.gcuStress]
    ),
]


class TestUpsample:

    @pytest.mark.parametrize("ishape, scale_factor", params_scales)
    @pytest.mark.parametrize("mode", mode)
    @pytest.mark.parametrize("dtype", dtype)
    def test_upsample(self, ishape, scale_factor, dtype, mode):
        cpu_input, supa_input = create_random_tensor(
            ishape, dtype=dtype, requires_grad=True
        )

        if mode == "linear" and len(ishape) != 3:
            return
        if (mode == "bilinear" or mode == "bicubic") and len(ishape) != 4:
            return
        if mode == "trilinear" and len(ishape) != 5:
            return

        m = nn.Upsample(scale_factor=scale_factor, mode=mode)
        cpu_res = m(cpu_input)
        supa_res = m(supa_input)

        assert_allclose(
            cpu_res, supa_res, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )

    @pytest.mark.parametrize("ishape, scale_factor", params_scales)
    @pytest.mark.parametrize("mode", mode)
    @pytest.mark.parametrize("dtype", dtype)
    def test_upsample_backward(self, ishape, scale_factor, dtype, mode):
        if mode == "linear" and len(ishape) != 3:
            return
        if (mode == "bilinear" or mode == "bicubic") and len(ishape) != 4:
            return
        if mode == "trilinear" and len(ishape) != 5:
            return

        if len(ishape) == 4:
            N, C, H, W = ishape
            ishape = (N, C, H, W)
            oshape = (N, C, H * scale_factor, W * scale_factor)
            cpu_input, supa_input = create_random_tensor(
                ishape, dtype=dtype, requires_grad=True
            )
            cpu_grad, supa_grad = create_random_tensor(oshape, dtype=dtype)
        elif len(ishape) == 3:
            N, H, W = ishape
            ishape = (N, H, W)
            oshape = (N, H, W * scale_factor)
            cpu_input, supa_input = create_random_tensor(
                ishape, dtype=dtype, requires_grad=True
            )
            cpu_grad, supa_grad = create_random_tensor(oshape, dtype=dtype)
        elif len(ishape) == 5:
            N, C, D, H, W = ishape
            ishape = (N, C, D, H, W)
            oshape = (N, C, D * scale_factor, H * scale_factor, W * scale_factor)
            cpu_input, supa_input = create_random_tensor(
                ishape, dtype=dtype, requires_grad=True
            )
            cpu_grad, supa_grad = create_random_tensor(oshape, dtype=dtype)
        else:
            assert False, "shape must be 3D 4D or 5D!"

        m = nn.Upsample(scale_factor=scale_factor, mode=mode)
        cpu_output = m(cpu_input)
        supa_output = m(supa_input)

        cpu_output.backward(cpu_grad)
        supa_output.backward(supa_grad)

        cpu_res = cpu_input.grad
        supa_res = supa_input.grad

        assert_allclose(
            cpu_res, supa_res, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )

    @pytest.mark.parametrize("ishape, scale_factor", params_scales)
    @pytest.mark.parametrize("mode", mode)
    @pytest.mark.parametrize("dtype", dtype)
    def test_upsample_with_size(self, ishape, scale_factor, dtype, mode):
        if mode == "linear" and len(ishape) != 3:
            return
        if (mode == "bilinear" or mode == "bicubic") and len(ishape) != 4:
            return
        if mode == "trilinear" and len(ishape) != 5:
            return

        if len(ishape) == 4:
            N, C, H, W = ishape
            size = (H * scale_factor, W * scale_factor)
        elif len(ishape) == 3:
            N, H, W = ishape
            size = W * scale_factor
        elif len(ishape) == 5:
            N, C, D, H, W = ishape
            size = (D * scale_factor, H * scale_factor, W * scale_factor)
        else:
            assert False, "shape must be 3D 4D or 5D!"

        cpu_input, supa_input = create_random_tensor(
            ishape, dtype=dtype, requires_grad=True
        )

        m = nn.Upsample(size=size, mode=mode)
        cpu_res = m(cpu_input)
        supa_res = m(supa_input)

        assert_allclose(
            cpu_res, supa_res, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )

    @pytest.mark.parametrize("ishape, oshape, mode", params_out)
    @pytest.mark.parametrize("dtype", dtype)
    @pytest.mark.parametrize("align_corners", align_corners)
    def test_upsample_with_out(self, ishape, oshape, mode, dtype, align_corners):
        cpu_input, supa_input = create_random_tensor(
            ishape, dtype=dtype, requires_grad=True
        )

        if mode == "linear" and len(ishape) != 3:
            return
        if (mode == "bilinear" or mode == "bicubic") and len(ishape) != 4:
            return
        if mode == "trilinear" and len(ishape) != 5:
            return

        if mode == "nearest":
            m = nn.Upsample(oshape, mode=mode)
        else:
            m = nn.Upsample(oshape, mode=mode, align_corners=align_corners)
        cpu_res = m(cpu_input)
        supa_res = m(supa_input)

        assert_allclose(
            cpu_res, supa_res, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
