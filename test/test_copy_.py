# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch
from torch_supa.testing.common_utils import (
    create_random_tensor,
    assert_allclose,
    generate_uncontiguous_tensor,
)
import pytest


d2d_typecast_params = [
    [
        (2, 16, 2048),
        torch.bfloat16,
        (32, 2048),
        torch.float32,
    ],  # uncontiguous + typecast
    [(1, 2, 3), torch.bfloat16, (2, 3), torch.float],  # contiguous + typecast
]


d2d_broadcast_params = [
    [
        (2, 1),
        torch.bfloat16,
        (2, 3),
        torch.float32,
    ],  # uncontiguous + typecast + broadcast
]


d2d_broadcast_activation = [
    [(6, 23, 45, 34), 3, 6, torch.bfloat16, (3, 23, 45, 34)],
]


class TestD2D:
    @pytest.mark.parametrize(
        "shape1, dtype1, shape2, dtype2", d2d_typecast_params
    )
    def test_typecast(self, shape1, dtype1, shape2, dtype2):
        x_cpu, x_supa = create_random_tensor(shape1, dtype1)

        x_cpu = x_cpu.reshape(shape2)
        x_supa = x_supa.reshape(shape2)

        y_cpu, y_supa = generate_uncontiguous_tensor(shape2, dtype=dtype2)
        y_cpu.copy_(x_cpu)
        y_supa.copy_(x_supa)
        assert_allclose(y_cpu, y_supa, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "shape1, dtype1, shape2, dtype2", d2d_broadcast_params
    )
    def test_broadcast(self, shape1, dtype1, shape2, dtype2):
        x_cpu, x_supa = create_random_tensor(shape1, dtype1)
        y_cpu, y_supa = create_random_tensor(shape2, dtype2)
        y_cpu.copy_(x_cpu)
        y_supa.copy_(x_supa)
        assert_allclose(y_cpu, y_supa, atol=0, rtol=0)

    @pytest.mark.parametrize("in_shape, start, end, dtype, out_shape", d2d_broadcast_activation)
    def test_broadcast_activation(self, in_shape, start, end, dtype, out_shape):
        x_cpu, x_supa = create_random_tensor(in_shape, dtype)
        x_cpu = x_cpu[start::end, :, :, :]
        x_supa = x_supa[start::end, :, :, :]
        y_cpu, y_supa = create_random_tensor(out_shape, dtype)

        y_cpu.copy_(x_cpu)
        y_supa.copy_(x_supa)
        assert_allclose(y_cpu, y_supa, atol=0, rtol=0)

    def test_permute_021_bfloat16(self):
        dst_cpu, dst_supa = create_random_tensor([1, 12288, 2048], torch.bfloat16)
        src_cpu, src_supa = create_random_tensor([1, 12288, 2048], torch.bfloat16)

        dst_cpu = torch.as_strided(dst_cpu, [1, 12288, 2048], [25165824, 2048, 1])
        dst_supa = torch.as_strided(dst_supa, [1, 12288, 2048], [25165824, 2048, 1])
        src_cpu = torch.as_strided(src_cpu, [1, 12288, 2048], [25165824, 1, 12288])
        src_supa = torch.as_strided(src_supa, [1, 12288, 2048], [25165824, 1, 12288])

        torch.ops.aten.copy_(dst_cpu, src_cpu)
        torch.ops.aten.copy_(dst_supa, src_supa)

        assert_allclose(dst_cpu, dst_supa, atol=0, rtol=0)
