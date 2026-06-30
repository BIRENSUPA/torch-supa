# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import numpy as np
import pytest
import torch
from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    create_torch_tensor_from_np,
)

# params_dim0 = [
#     [(4700,), [[1]]],
#     [(364800,), [[1], [2], [3]]],
#     [(512, 3, 2), [[1], [2], [3]]], # fastercnn
#     [(3, 384, 768), [0, 1]],
#     [(2, 2), [0,]],
#     [(3, 384), [0, 2]],
#     [(5, 384, 5), [0, 2, 4]],
#     [(32, 3, 224, 224), [0, 2]],
#     [(5, 3, 224, 224), [0, 2, 3]]
# ]

# index_br200_params = [
#     [(2, 2), [0,]],
# ]

index_br200_params = [
    pytest.param(
        (2, 2),
        [
            0,
        ],
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (32, 3, 224, 224), [0, 2], marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((32, 3, 256, 256), [0, 2], marks=[pytest.mark.gcuStress]),
    pytest.param((32, 3, 256, 257), [0, 2], marks=[pytest.mark.gcuStress]),
    pytest.param((32, 3, 256, 257), [0, 2], marks=[pytest.mark.gcuStress]),
]


accumulates = [True, False]
index_put_1d_input_dtypes = [torch.float32, torch.int32, torch.int64, torch.uint8]
index_put_br200_params = [
    pytest.param(
        (32,),
        (6,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((16 * 8,), (5,), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((11143,), (32,), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((11143 * 4,), (31,), marks=[pytest.mark.gcuStress]),
    pytest.param((11143 * 7,), (31,), marks=[pytest.mark.gcuStress]),
]

masked_select_br200_params = [
    pytest.param(
        (16,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 1023), marks=[pytest.mark.gcuStress]),
    pytest.param((1024, 1024), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 1025), marks=[pytest.mark.gcuStress]),
]

masked_scatter_br200_params = [
    pytest.param(
        (3, 3),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 1023), marks=[pytest.mark.gcuStress]),
    pytest.param((1024, 1024), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 1025), marks=[pytest.mark.gcuStress]),
]

test_index_fill_params = [
    pytest.param(
        (16, 16, 16),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 512, 16), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 1023, 16), marks=[pytest.mark.gcuStress]),
    pytest.param((1024, 1024, 16), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 1025, 16), marks=[pytest.mark.gcuStress]),
]

params = [
    pytest.param(
        (16,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 512), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1024, 1024), marks=[pytest.mark.gcuStress]),
    pytest.param((2048, 2048), marks=[pytest.mark.gcuStress]),
    pytest.param((4096, 4096), marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.float32]

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


class TestIndexKernelOps:
    @pytest.mark.parametrize("shape, indices", index_br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_index_dim0(self, shape, indices, dtype):
        cpu_input, supa_input = create_random_tensor(
            shape, dtype=dtype, requires_grad=False
        )
        ind = torch.tensor(indices)

        cpu_output = cpu_input[ind]
        supa_output = supa_input[ind.to(supa_device)]

        assert_allclose(cpu_output, supa_output, rtol=1e-5, atol=5e-5)

    @pytest.mark.parametrize("shape, index_num", index_put_br200_params)
    @pytest.mark.parametrize("accumulate", accumulates)
    @pytest.mark.parametrize("dtype", index_put_1d_input_dtypes)
    def test_index_put_1d(self, shape, index_num, accumulate, dtype):
        if accumulate and (dtype == torch.uint8 or dtype == torch.bfloat16):
            pass
        else:
            x_cpu, x_supa = create_random_tensor(shape=shape, dtype=dtype)
            indices = np.random.choice(shape[0], index_num, replace=accumulate)
            indices_cpu, indices_supa = create_torch_tensor_from_np(indices)
            values_cpu, values_supa = create_random_tensor(shape=index_num, dtype=dtype)
            y_cpu = x_cpu.index_put(
                indices=[indices_cpu], values=values_cpu, accumulate=accumulate
            )
            y_supa = x_supa.index_put(
                indices=[indices_supa], values=values_supa, accumulate=accumulate
            )
            assert_allclose(y_cpu, y_supa, atol=1e-5, rtol=5e-5)

    @pytest.mark.parametrize("shape", masked_select_br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_masked_select(self, shape, dtype):
        cpu_input, supa_input = create_random_tensor(
            shape, dtype=dtype, requires_grad=False
        )
        masks_cpu, masks_supa = create_random_tensor(shape, dtype=torch.bool)

        cpu_output = torch.masked_select(cpu_input, masks_cpu)
        supa_output = torch.masked_select(supa_input, masks_supa)

        assert_allclose(cpu_output, supa_output.cpu(), rtol=0, atol=0)

    @pytest.mark.parametrize("shape", masked_scatter_br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_masked_scatter(self, shape, dtype):
        np.random.seed(1)
        cpu_input, supa_input = create_random_tensor(
            shape, dtype=dtype, requires_grad=False
        )

        masks_cpu, masks_supa = create_random_tensor(shape, dtype=torch.bool)

        source_shape = torch.ne(masks_cpu, 0).sum().item()
        cpu_source, supa_source = create_random_tensor([source_shape], dtype=dtype)

        cpu_output = torch.masked_scatter(cpu_input, masks_cpu, source=cpu_source)
        supa_output = torch.masked_scatter(supa_input, masks_supa, source=supa_source)

        assert_allclose(cpu_output, supa_output, rtol=1e-5, atol=5e-5)

    @pytest.mark.parametrize("shape", test_index_fill_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_index_fill_br200(self, shape, dtype):
        in_cpu, in_supa = create_random_tensor(shape, dtype)
        index = torch.tensor([0, 2])
        index_supa = index.to(supa_device)
        out_cpu = in_cpu.index_fill(1, index, -1)
        out_supa = in_supa.index_fill(1, index_supa, -1)
        assert_allclose(out_cpu, out_supa.cpu(), rtol=1e-5, atol=5e-5)

    @pytest.mark.parametrize("shape", test_index_fill_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_put_br200(self, shape, dtype):
        in_cpu, in_supa = create_random_tensor(shape, dtype)
        out_cpu = in_cpu.put(torch.tensor([1, 3]), torch.tensor([9.0, 10.0]))
        out_supa = in_supa.put(
            torch.tensor([1, 3]).to(supa_device),
            torch.tensor([9.0, 10.0]).to(supa_device),
        )
        assert_allclose(out_cpu, out_supa.cpu(), rtol=1e-5, atol=5e-5)

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_take_br200(self, shape, dtype):
        in_cpu, in_supa = create_random_tensor(shape, dtype)
        index = torch.tensor([0, 2, 5])
        index_supa = index.to(supa_device)
        out_cpu = torch.take(in_cpu, index)
        out_supa = torch.take(in_supa, index_supa)
        assert_allclose(out_cpu, out_supa.cpu(), rtol=1e-5, atol=5e-5)

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_flip_br200(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype)

        y_cpu = torch.flip(x_cpu.view(2, 2, -1), [0, 1])
        y_supa = torch.flip(x_supa.view(2, 2, -1), [0, 1])
        assert_allclose(y_cpu, y_supa, rtol=1e-5, atol=5e-5)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_index_copy(self):
        x = torch.ones(5, 3)
        x_supa = x.to(supa_device)
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        t_supa = t.to(supa_device)
        index = torch.tensor([0, 4, 2])
        index_supa = index.to(supa_device)
        x.index_copy(0, index, t)
        x_supa.index_copy(0, index_supa, t_supa)
        assert_allclose(x, x_supa, rtol=1e-5, atol=5e-5)


    def test_index_expand_idx(self):
        shape = [4, 256]
        dtype = torch.float
        cpu_input, supa_input = create_random_tensor(
            shape, dtype=dtype, requires_grad=False
        )
        idx1 = torch.tensor([0, 1], dtype=torch.int32)
        idx2 = torch.tensor([0], dtype=torch.int64)
        cpu_output = cpu_input[idx1, idx2]
        supa_output = supa_input[idx1.to(supa_device), idx2.to(supa_device)]
        assert_allclose(cpu_output, supa_output, rtol=1e-5, atol=5e-5)
