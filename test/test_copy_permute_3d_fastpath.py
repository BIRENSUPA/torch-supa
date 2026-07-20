# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

_ci_marks = [
    pytest.mark.sanity,
    pytest.mark.gcuSmoke,
    pytest.mark.regression,
    pytest.mark.gcuSanity,
    pytest.mark.gcuStress,
]

shapes = [
    pytest.param((4, 8, 16), marks=_ci_marks),
    pytest.param((3, 5, 7), marks=_ci_marks),
]

dtypes = [
    pytest.param(torch.bfloat16, marks=_ci_marks),
    pytest.param(torch.float16, marks=_ci_marks),
]


def _contiguous_strides(shape):
    stride = 1
    strides = [0] * len(shape)
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = stride
        stride *= shape[i]
    return strides


def _permute_021_strides(shape):
    # Input strides for a dense tensor after permute(0, 2, 1); logical shape unchanged.
    a, b, c = shape
    return [b * c, 1, b]


def _as_strided_pair(base_cpu, base_supa, shape, stride):
    return (
        torch.as_strided(base_cpu, shape, stride),
        torch.as_strided(base_supa, shape, stride),
    )


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
def test_copy_permute_021_small(shape, dtype):
    """copy_ with permute(0,2,1) stride layout -> transpose_tile_big_kernel fast path."""
    src_stride = _permute_021_strides(shape)
    dst_stride = _contiguous_strides(shape)

    dst_cpu, dst_supa = create_random_tensor(shape, dtype)
    src_cpu, src_supa = create_random_tensor(shape, dtype)
    dst_cpu, dst_supa = _as_strided_pair(dst_cpu, dst_supa, shape, dst_stride)
    src_cpu, src_supa = _as_strided_pair(src_cpu, src_supa, shape, src_stride)

    torch.ops.aten.copy_(dst_cpu, src_cpu)
    torch.ops.aten.copy_(dst_supa, src_supa)
    assert_allclose(dst_cpu, dst_supa, atol=0, rtol=0)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
def test_contiguous_permute_021_small(shape, dtype):
    """contiguous on permute(0,2,1) stride layout -> transpose_tile_big_kernel fast path."""
    src_stride = _permute_021_strides(shape)

    src_cpu, src_supa = create_random_tensor(shape, dtype)
    src_cpu, src_supa = _as_strided_pair(src_cpu, src_supa, shape, src_stride)

    out_cpu = torch.ops.aten.contiguous(src_cpu)
    out_supa = torch.ops.aten.contiguous(src_supa)
    assert_allclose(out_cpu, out_supa, atol=0, rtol=0)


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.regression
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
def test_copy_float32_permute_021_bypass_fastpath():
    """Float32 3D permute layout must bypass BF16/FP16 fast path but stay correct."""
    shape = (4, 8, 16)
    src_stride = _permute_021_strides(shape)
    dst_stride = _contiguous_strides(shape)

    dst_cpu, dst_supa = create_random_tensor(shape, torch.float32)
    src_cpu, src_supa = create_random_tensor(shape, torch.float32)
    dst_cpu, dst_supa = _as_strided_pair(dst_cpu, dst_supa, shape, dst_stride)
    src_cpu, src_supa = _as_strided_pair(src_cpu, src_supa, shape, src_stride)

    torch.ops.aten.copy_(dst_cpu, src_cpu)
    torch.ops.aten.copy_(dst_supa, src_supa)
    assert_allclose(dst_cpu, dst_supa, atol=0, rtol=0)


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.regression
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
def test_copy_non_permute_stride_bypass_fastpath():
    """Arbitrary non-permute strides must bypass 3D permute fast path but stay correct."""
    shape = (4, 8, 16)
    src_stride = (100, 10, 1)
    dst_stride = _contiguous_strides(shape)

    dst_cpu, dst_supa = create_random_tensor(shape, torch.bfloat16)
    src_cpu, src_supa = create_random_tensor(shape, torch.bfloat16)
    dst_cpu, dst_supa = _as_strided_pair(dst_cpu, dst_supa, shape, dst_stride)
    src_cpu, src_supa = _as_strided_pair(src_cpu, src_supa, shape, src_stride)

    torch.ops.aten.copy_(dst_cpu, src_cpu)
    torch.ops.aten.copy_(dst_supa, src_supa)
    assert_allclose(dst_cpu, dst_supa, atol=0, rtol=0)


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.regression
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
def test_copy_contiguous_identity_bypass_fastpath():
    """Contiguous src/dst identity layout must bypass permute fast path but stay correct."""
    shape = (4, 8, 16)
    stride = _contiguous_strides(shape)

    dst_cpu, dst_supa = create_random_tensor(shape, torch.bfloat16)
    src_cpu, src_supa = create_random_tensor(shape, torch.bfloat16)
    dst_cpu, dst_supa = _as_strided_pair(dst_cpu, dst_supa, shape, stride)
    src_cpu, src_supa = _as_strided_pair(src_cpu, src_supa, shape, stride)

    torch.ops.aten.copy_(dst_cpu, src_cpu)
    torch.ops.aten.copy_(dst_supa, src_supa)
    assert_allclose(dst_cpu, dst_supa, atol=0, rtol=0)
