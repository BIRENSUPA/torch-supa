# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

# shapes = [
#     [6, 2, 3, 4],
#     [6, 2, 3],
#     [6, 2],
#     [6],
# ]

shapes = [
    pytest.param(
        (6, 2, 3, 4),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (6, 2, 3),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (6, 2),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (6,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (6, 2, 512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((4, 512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((2048,), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((5, 2046, 2135), marks=[pytest.mark.gcuStress]),
]


dtypes = [
    torch.float32,
    torch.float,
    torch.bfloat16,
    torch.float16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.short,
    torch.int32,
    torch.int,
    torch.int64,
    torch.long,
    torch.bool,
]

# src_shape, src_size, src_stride, src_offset, dst_shape, dst_size, dst_stride, dst_offset
shapes_for_strided = [
    ([16, 192, 778], [192, 32], [778, 1], 1643552, [16, 192, 32], [192, 32], [32, 1], 67584),
]

dtypes_for_strided = [
    torch.float32,
    torch.bfloat16,
    torch.int32,
    # torch.double,
    torch.long,
]

def create_strided_tensor(shape, size, stride, offset, dtype):
    x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
    x_cpu = torch.as_strided(x_cpu, size, stride, storage_offset=offset)
    x_supa = torch.as_strided(x_supa, size, stride, storage_offset=offset)
    return x_cpu, x_supa

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


class TestCopy:

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_copy(self, shape, dtype):
        x_cpu, _ = create_random_tensor(shape, dtype=torch.float)
        if dtype == torch.bool:
            x_cpu = x_cpu > torch.median(x_cpu)
        else:
            x_cpu *= 17
            x_cpu = x_cpu.to(dtype)

        x_supa = x_cpu.clone()
        x_supa = x_supa.to(supa_device)

        if dtype == torch.bfloat16:
            # Promote to fp32 for assertion as numpy does not support bf16
            x_cpu = x_cpu.to(torch.float)
            x_supa = x_supa.cpu().to(torch.float)
        assert_allclose(x_cpu, x_supa, rtol=0, atol=0, equal_nan=True)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_copyh2d(self):
        len = 4
        x_cpu = torch.zeros([len * 2], dtype=torch.float32)
        x_cpu.narrow(0, 0, len).copy_(torch.ones([len], dtype=torch.float32).cpu())
        x_supa = x_cpu.supa()
        x_supa.narrow(0, 0, len).copy_(torch.ones([len], dtype=torch.float32).cpu())
        assert_allclose(x_cpu, x_supa, rtol=0, atol=0, equal_nan=True)

    @pytest.mark.parametrize("dtype", dtypes_for_strided)
    def test_empty_strided_copyd2d(self, dtype):
        shape = (2, 4, 2, 2)
        strides = (32, 4, 2, 1)
        y_cpu = torch.empty_strided(shape, strides, dtype=dtype)
        y_supa = torch.empty_strided(shape, strides, dtype=dtype, device=torch.device("supa"))
        x_cpu = torch.randn(shape)
        x_supa = x_cpu.supa()
        y_cpu.copy_(x_cpu)
        y_supa.copy_(x_supa)
        assert_allclose(y_cpu, y_supa, rtol=1e-8, atol=1e-8, equal_nan=True)

    @pytest.mark.parametrize(
        "src_shape, src_size, src_stride, src_offset, dst_shape, dst_size, dst_stride, dst_offset",
        shapes_for_strided,
    )
    @pytest.mark.parametrize("dtype", dtypes_for_strided)
    def test_general_strided_copyd2d(
        self, src_shape, src_size, src_stride, src_offset, dst_shape, dst_size, dst_stride, dst_offset, dtype
    ):
        x_src_cpu, x_src_supa = create_strided_tensor(src_shape, src_size, src_stride, src_offset, dtype)
        x_dst_cpu, x_dst_supa = create_strided_tensor(dst_shape, dst_size, dst_stride, dst_offset, dtype)

        x_dst_cpu.copy_(x_src_cpu)
        x_dst_supa.copy_(x_src_supa)

        assert_allclose(x_dst_cpu, x_dst_supa, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_tensor_shallow_copy(self):
        cpu_a, supa_a = create_random_tensor((3, 2, 1), dtype=torch.float)
        supa_a.data = cpu_a
        assert_allclose(supa_a, cpu_a, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_depromotion(self):

        with pytest.raises(RuntimeError):
            """no exception if BRTB_ENABLE_DTYPE_DEMOTION=1"""
            a = torch.zeros(2, 2, dtype=torch.float64)
            a_supa = a.supa()

        """normal printing must pass"""
        s = torch.zeros(2, 2).supa()
        print(s)
        """explicit conversion must pass"""
        s2 = s.double()
        print(s2)
