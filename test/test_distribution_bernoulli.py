# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import numpy as np
import pytest

import torch
from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
    RandomMode,
)

params = [
    pytest.param(
        (2, 6, 32, 64),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (16, 16, 256, 64), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]
    ),
    pytest.param((16, 16, 253, 62), marks=[pytest.mark.gcuStress]),
    pytest.param((16, 16, 256, 66), marks=[pytest.mark.gcuStress]),
    pytest.param((16, 15, 259, 69), marks=[pytest.mark.gcuStress]),
]

probability = [0.1, 0.3, 0.5, 0.7, 0.9]
SEED = [6, 10]


intern_vl_params = [
    pytest.param(
        (2, 1, 1),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (3, 1, 1),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (4, 1, 1),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (5, 1, 1),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (6, 1, 1),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (7, 1, 1),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((200, 1, 1), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((2200, 1, 1), marks=[pytest.mark.gcuStress]),
    pytest.param((2230, 1, 1), marks=[pytest.mark.gcuStress]),
]

intern_vl_probability = [0.2]


class TestBernoulli:

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("p", probability)
    def test_tensor(self, shape, p):
        _, x_supa = create_random_tensor(
            shape,
            dtype=torch.float32,
            min_value=p,
            max_value=p,
            requires_grad=False,
            mode=RandomMode.uniform,
        )
        y_supa = torch.bernoulli(x_supa)

        one_num_expected = np.prod(shape) * p
        one_num_supa = torch.sum(y_supa.cpu())
        factor = 0.15
        if np.prod(shape) < 1000:
            factor = 0.3
        assert one_num_supa <= one_num_expected * (
            1 + factor
        ) and one_num_supa >= one_num_expected * (
            1 - factor
        ), "burnoulli op error shape: %s probability: %s" % (
            shape,
            p,
        )

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("seed", SEED)
    def test_bernoulli_seed(self, shape, seed):
        _, supa_in = create_random_tensor(
            shape,
            dtype=torch.float32,
            requires_grad=False,
            min_value=0,
            max_value=1,
            mode=RandomMode.uniform,
        )

        torch.manual_seed(seed)
        output0 = torch.bernoulli(supa_in)

        torch.manual_seed(seed)
        output1 = torch.bernoulli(supa_in)

        assert_allclose(output0, output1, rtol=0, atol=0)

    @pytest.mark.parametrize("shape", intern_vl_params)
    @pytest.mark.parametrize("probability", intern_vl_probability)
    @pytest.mark.parametrize("seed", SEED)
    def test_seed_intern_vl_bf16(self, shape, probability, seed):
        _, x_supa = create_random_tensor(
            shape,
            dtype=torch.bfloat16,
            requires_grad=False,
            min_value=0,
            max_value=1,
            mode=RandomMode.uniform,
        )
        torch.manual_seed(seed)
        output0 = x_supa.clone()
        output0.bernoulli_(probability)
        torch.manual_seed(seed)
        output1 = x_supa.clone()
        output1.bernoulli_(probability)
        assert_allclose(output0, output1, rtol=0, atol=0)

        torch.manual_seed(seed)
        output0 = torch.bernoulli(x_supa)
        torch.manual_seed(seed)
        output1 = torch.bernoulli(x_supa)
        assert_allclose(output0, output1, rtol=0, atol=0)

        torch.manual_seed(seed)
        output0 = x_supa.bernoulli(probability)
        torch.manual_seed(seed)
        output1 = x_supa.bernoulli(probability)
        assert_allclose(output0, output1, rtol=0, atol=0)

    @pytest.mark.parametrize("shape", intern_vl_params)
    @pytest.mark.parametrize("probability", intern_vl_probability)
    @pytest.mark.parametrize("seed", SEED)
    def test_seed_intern_vl_fp16(self, shape, probability, seed):
        _, x_supa = create_random_tensor(
            shape,
            dtype=torch.float16,
            requires_grad=False,
            min_value=0,
            max_value=1,
            mode=RandomMode.uniform,
        )
        torch.manual_seed(seed)
        output0 = x_supa.clone()
        output0.bernoulli_(probability)
        torch.manual_seed(seed)
        output1 = x_supa.clone()
        output1.bernoulli_(probability)
        assert_allclose(output0, output1, rtol=0, atol=0)

        torch.manual_seed(seed)
        output0 = torch.bernoulli(x_supa)
        torch.manual_seed(seed)
        output1 = torch.bernoulli(x_supa)
        assert_allclose(output0, output1, rtol=0, atol=0)

        torch.manual_seed(seed)
        output0 = x_supa.bernoulli(probability)
        torch.manual_seed(seed)
        output1 = x_supa.bernoulli(probability)
        assert_allclose(output0, output1, rtol=0, atol=0)
