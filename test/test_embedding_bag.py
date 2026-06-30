# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor


cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

params = [
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
        (1024, 20),
        (1, 50),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (30522, 768), (16, 384), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((30522, 768 * 3), (16, 384), marks=[pytest.mark.gcuStress]),
    pytest.param((30522 * 2, 768), (16, 384), marks=[pytest.mark.gcuStress]),
    pytest.param((30522 * 3, 768 * 5), (16, 384), marks=[pytest.mark.gcuStress]),
]

idx = [-1, 0]

modes = ["sum", "mean", "max"]

dtypes = [torch.float32, torch.bfloat16, torch.float16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 2e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


@pytest.mark.parametrize("wshape, ishape", params)
@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("dtype", dtypes)
def test_embedding_bag(wshape, ishape, mode, dtype):
    origin_i = np.random.randint(0, wshape[0], ishape).astype(np.int32).tolist()
    i_cpu = torch.tensor(origin_i, dtype=torch.int32, device=cpu_device)
    i_supa = torch.tensor(origin_i, dtype=torch.int32, device=supa_device)

    embedding_bag_op = nn.EmbeddingBag(wshape[0], wshape[1], mode=mode)

    embedding_bag_op.weight.data = embedding_bag_op.weight.data.to(dtype)
    embedding_bag_supa = copy.deepcopy(embedding_bag_op).to(supa_device)
    embedding_bag_op.weight.data = embedding_bag_op.weight.data.float()

    y_cpu = embedding_bag_op(i_cpu)
    y_supa = embedding_bag_supa(i_supa)
    assert_allclose(y_cpu.to(dtype), y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("wshape, ishape", params)
@pytest.mark.parametrize("dtype", dtypes)
def test_embedding_bag_per_sample_weight(wshape, ishape, dtype):
    origin_i = np.random.randint(0, wshape[0], ishape).astype(np.int32).tolist()
    i_cpu = torch.tensor(origin_i, dtype=torch.int32, device=cpu_device)
    i_supa = torch.tensor(origin_i, dtype=torch.int32, device=supa_device)

    w_cpu, w_supa = create_random_tensor(ishape, dtype)

    embedding_bag_op = nn.EmbeddingBag(wshape[0], wshape[1], mode="sum")

    embedding_bag_op.weight.data = embedding_bag_op.weight.data.to(dtype)
    embedding_bag_supa = copy.deepcopy(embedding_bag_op).to(supa_device)
    embedding_bag_op.weight.data = embedding_bag_op.weight.data.float()

    y_cpu = embedding_bag_op(i_cpu, per_sample_weights=w_cpu.float())
    y_supa = embedding_bag_supa(i_supa, per_sample_weights=w_supa)
    assert_allclose(y_cpu.to(dtype), y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("wshape, ishape", params)
@pytest.mark.parametrize("padding_idx", idx)
@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("dtype", dtypes)
def test_embedding_bag_bwd(wshape, ishape, padding_idx, mode, dtype):
    origin_i = np.random.randint(0, wshape[0], ishape).astype(np.int32).tolist()
    i_cpu = torch.tensor(origin_i, dtype=torch.int32, device=cpu_device)
    i_supa = torch.tensor(origin_i, dtype=torch.int32, device=supa_device)

    embedding_bag_op = nn.EmbeddingBag(
        wshape[0], wshape[1], padding_idx=padding_idx, mode=mode
    )

    embedding_bag_op.weight.data = embedding_bag_op.weight.data.to(dtype)
    embedding_bag_supa = copy.deepcopy(embedding_bag_op).to(supa_device)
    embedding_bag_op.weight.data = embedding_bag_op.weight.data.float()

    y_cpu = embedding_bag_op(i_cpu)
    y_supa = embedding_bag_supa(i_supa)
    assert_allclose(y_cpu.to(dtype), y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])
    if mode == "sum" or mode == "mean":
        g_cpu, g_supa = create_random_tensor(y_cpu.shape, dtype)
        y_cpu.backward(g_cpu.float())
        y_supa.backward(g_supa)
        assert_allclose(
            embedding_bag_op.weight.grad,
            embedding_bag_supa.weight.grad.cpu().float(),
            atol=ATOL[dtype],
            rtol=RTOL[dtype],
        )


@pytest.mark.parametrize("wshape, ishape", params)
@pytest.mark.parametrize("padding_idx", idx)
@pytest.mark.parametrize("dtype", dtypes)
def test_embedding_bag_per_sampel_weight_bwd(wshape, ishape, padding_idx, dtype):
    origin_i = np.random.randint(0, wshape[0], ishape).astype(np.int32).tolist()
    i_cpu = torch.tensor(origin_i, dtype=torch.int32, device=cpu_device)
    i_supa = torch.tensor(origin_i, dtype=torch.int32, device=supa_device)

    embedding_bag_op = nn.EmbeddingBag(
        wshape[0], wshape[1], padding_idx=padding_idx, mode="sum"
    )

    embedding_bag_op.weight.data = embedding_bag_op.weight.data.to(dtype)
    embedding_bag_supa = copy.deepcopy(embedding_bag_op).to(supa_device)
    embedding_bag_op.weight.data = embedding_bag_op.weight.data.float()

    w_cpu, w_supa = create_random_tensor(ishape, dtype)

    y_cpu = embedding_bag_op(i_cpu, per_sample_weights=w_cpu.float())
    y_supa = embedding_bag_supa(i_supa, per_sample_weights=w_supa)

    g_cpu, g_supa = create_random_tensor(y_cpu.shape, dtype)
    y_cpu.backward(g_cpu.float())
    y_supa.backward(g_supa)

    assert_allclose(y_cpu.to(dtype), y_supa, atol=ATOL[dtype], rtol=RTOL[dtype])
    assert_allclose(
        embedding_bag_op.weight.grad,
        embedding_bag_supa.weight.grad.cpu().float(),
        atol=ATOL[dtype],
        rtol=RTOL[dtype],
    )
