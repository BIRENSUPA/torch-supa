# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


br200_params = [
    pytest.param(
        (20, 30),
        (3, 5),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (8193, 257), (3, 11), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param(
        (30522, 768), (16, 384), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((30522, 768 * 3), (16, 384), marks=[pytest.mark.gcuStress]),
    pytest.param((30522 * 2, 768), (16, 384), marks=[pytest.mark.gcuStress]),
    pytest.param((30522 * 3, 768 * 5), (16, 384), marks=[pytest.mark.gcuStress]),
]


params_renorm_ = [
    pytest.param(
        (1, 64),
        (32, 4),
        1.0,
        2.0,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (1, 128),
        (64, 32),
        1.0,
        2.0,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param((1, 768 * 3), (16, 33), 1.0, 2.0, marks=[pytest.mark.gcuStress]),
    pytest.param((1 * 2, 768), (16, 24), 1.0, 2.0, marks=[pytest.mark.gcuStress]),
    pytest.param((1 * 3, 768 * 5), (16, 52), 1.0, 2.0, marks=[pytest.mark.gcuStress]),
]

idx = [-1, 0]

dtypes = [torch.float32, torch.bfloat16, torch.float16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


@pytest.mark.parametrize("wshape, ishape", br200_params)
@pytest.mark.parametrize("dtype", dtypes)
def test_embedding(wshape, ishape, dtype):
    origin_i = np.random.randint(0, wshape[0], ishape).astype(np.int32).tolist()
    i_cpu = torch.tensor(origin_i, dtype=torch.int32, device=cpu_device)
    i_supa = torch.tensor(origin_i, dtype=torch.int32, device=supa_device)

    embedding_op = nn.Embedding(wshape[0], wshape[1])

    embedding_op.weight.data = embedding_op.weight.data.to(dtype)
    embedding_supa = copy.deepcopy(embedding_op).to(supa_device)
    embedding_op.weight.data = embedding_op.weight.data.float()

    y_cpu = embedding_op(i_cpu)
    y_supa = embedding_supa(i_supa)
    assert_allclose(y_cpu.to(dtype), y_supa, atol=5e-5, rtol=1e-5)


@pytest.mark.parametrize("wshape, ishape", br200_params)
@pytest.mark.parametrize("padding_idx", idx)
@pytest.mark.parametrize("dtype", dtypes)
def test_embedding_bwd(wshape, ishape, padding_idx, dtype):
    origin_i = np.random.randint(0, wshape[0], ishape).astype(np.int32).tolist()
    i_cpu = torch.tensor(origin_i, dtype=torch.int32, device=cpu_device)
    i_supa = torch.tensor(origin_i, dtype=torch.int32, device=supa_device)

    embedding_op = nn.Embedding(wshape[0], wshape[1], padding_idx=padding_idx)

    embedding_op.weight.data = embedding_op.weight.data.to(dtype)
    embedding_supa = copy.deepcopy(embedding_op).to(supa_device)
    embedding_op.weight.data = embedding_op.weight.data.float()

    y_cpu = embedding_op(i_cpu)
    y_supa = embedding_supa(i_supa)

    g_cpu, g_supa = create_random_tensor(y_cpu.shape, dtype)
    y_cpu.backward(g_cpu.float())
    y_supa.backward(g_supa)

    assert_allclose(y_cpu.to(dtype), y_supa, atol=5e-5, rtol=1e-5)
    assert_allclose(
        embedding_op.weight.grad,
        embedding_supa.weight.grad.cpu().float(),
        atol=ATOL[dtype],
        rtol=RTOL[dtype],
    )


@pytest.mark.parametrize("wshape, ishape, max_norm, norm_type", params_renorm_)
@pytest.mark.parametrize("dtype", dtypes)
def test_embedding_renorm_(wshape, ishape, max_norm, norm_type, dtype):
    origin_i = np.random.randint(0, wshape[0], ishape).astype(np.int32).tolist()
    i_cpu = torch.tensor(origin_i, dtype=torch.int32, device=cpu_device)
    i_supa = torch.tensor(origin_i, dtype=torch.int32, device=supa_device)

    embedding_op = nn.Embedding(
        wshape[0], wshape[1], max_norm=max_norm, norm_type=norm_type
    )

    embedding_op.weight.data = embedding_op.weight.data.to(dtype)
    embedding_supa = copy.deepcopy(embedding_op).to(supa_device)
    embedding_op.weight.data = embedding_op.weight.data.float()

    embedding_op(i_cpu)
    embedding_supa(i_supa)

    assert_allclose(i_cpu, i_supa, atol=5e-5, rtol=1e-5)
