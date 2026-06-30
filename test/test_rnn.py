# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import copy

import pytest
import torch
import torch.nn as nn

from torch_supa.testing.common_utils import (
    assert_allclose,
    create_random_tensor,
)

crnn_shapes = [
    # input shape, grad shape, input size, hidden size, bidirectional
    # CRNN
    pytest.param(
        [3, 4, 5],
        [3, 4, 4],
        5,
        2,
        True,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [3, 4, 5],
        [3, 4, 3],
        5,
        3,
        False,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [30, 10, 5],
        [30, 10, 4],
        5,
        2,
        True,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        [30, 10, 5],
        [30, 10, 3],
        5,
        3,
        False,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        [300, 100, 5], [300, 100, 4], 5, 2, True, marks=[pytest.mark.gcuStress]
    ),
    pytest.param(
        [300, 100, 5], [300, 100, 3], 5, 3, False, marks=[pytest.mark.gcuStress]
    ),
]

gru_shapes = [
    # input shape, grad shape, input size, hidden size, bidirectional
    # CRNN
    pytest.param(
        [3, 4, 5],
        [3, 4, 4],
        5,
        4,
        True,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [3, 4, 5],
        [3, 4, 3],
        5,
        3,
        False,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [30, 10, 5],
        [30, 10, 4],
        5,
        4,
        True,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        [30, 10, 5],
        [30, 10, 3],
        5,
        3,
        False,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        [300, 100, 5],
        [300, 100, 4],
        5,
        4,
        True,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
    pytest.param(
        [300, 100, 5],
        [300, 100, 3],
        5,
        3,
        False,
        marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress],
    ),
]

use_cudnn = False


@pytest.mark.parametrize(
    "shape, grad_shape, input_size, hidden_size, bidirectional", crnn_shapes
)
def test_lstm(shape, grad_shape, input_size, hidden_size, bidirectional):
    with torch.backends.cudnn.flags(enabled=use_cudnn):
        cpu_input, supa_input = create_random_tensor(
            shape, dtype=torch.float32, requires_grad=True
        )
        cpu_grad, supa_grad = create_random_tensor(grad_shape, dtype=torch.float32)

        cpu_rnn = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional)
        cpu_rnn.weight_ih_l0 = nn.Parameter(torch.ones(4 * hidden_size, input_size))
        cpu_rnn.weight_hh_l0 = nn.Parameter(torch.ones(4 * hidden_size, hidden_size))
        cpu_rnn.bias_ih_l0 = nn.Parameter(torch.ones(4 * hidden_size))
        cpu_rnn.bias_hh_l0 = nn.Parameter(torch.ones(4 * hidden_size))
        cpu_rnn.weight_ih_l0_reverse = nn.Parameter(
            torch.ones(4 * hidden_size, input_size)
        )
        cpu_rnn.weight_hh_l0_reverse = nn.Parameter(
            torch.ones(4 * hidden_size, hidden_size)
        )
        cpu_rnn.bias_ih_l0_reverse = nn.Parameter(torch.ones(4 * hidden_size))
        cpu_rnn.bias_hh_l0_reverse = nn.Parameter(torch.ones(4 * hidden_size))

        supa_rnn = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional)
        supa_rnn.weight_ih_l0 = nn.Parameter(torch.ones(4 * hidden_size, input_size))
        supa_rnn.weight_hh_l0 = nn.Parameter(torch.ones(4 * hidden_size, hidden_size))
        supa_rnn.bias_ih_l0 = nn.Parameter(torch.ones(4 * hidden_size))
        supa_rnn.bias_hh_l0 = nn.Parameter(torch.ones(4 * hidden_size))
        supa_rnn.weight_ih_l0_reverse = nn.Parameter(
            torch.ones(4 * hidden_size, input_size)
        )
        supa_rnn.weight_hh_l0_reverse = nn.Parameter(
            torch.ones(4 * hidden_size, hidden_size)
        )
        supa_rnn.bias_ih_l0_reverse = nn.Parameter(torch.ones(4 * hidden_size))
        supa_rnn.bias_hh_l0_reverse = nn.Parameter(torch.ones(4 * hidden_size))

        supa_rnn = supa_rnn.supa()
        cpu_output = cpu_rnn(cpu_input)
        supa_output = supa_rnn(supa_input)

        cpu_output[0].backward(cpu_grad)
        supa_output[0].backward(supa_grad)

        cpu_ht, supa_ht = cpu_output[1][0], supa_output[1][0]
        cpu_ct, supa_ct = cpu_output[1][1], supa_output[1][1]

        assert_allclose(cpu_output[0], supa_output[0], rtol=5e-4, atol=1e-3)
        assert_allclose(cpu_ht, supa_ht, rtol=5e-4, atol=1e-3)
        assert_allclose(cpu_ct, supa_ct, rtol=5e-4, atol=1e-3)
        assert_allclose(cpu_grad, supa_grad, rtol=5e-4, atol=1e-3)
        assert_allclose(
            cpu_rnn.weight_ih_l0.grad, supa_rnn.weight_ih_l0.grad, rtol=5e-4, atol=1e-3
        )
        assert_allclose(
            cpu_rnn.weight_hh_l0.grad, supa_rnn.weight_hh_l0.grad, rtol=5e-4, atol=1e-3
        )
        assert_allclose(
            cpu_rnn.bias_ih_l0.grad, supa_rnn.bias_ih_l0.grad, rtol=5e-4, atol=1e-2
        )
        assert_allclose(
            cpu_rnn.bias_hh_l0.grad, supa_rnn.bias_hh_l0.grad, rtol=5e-4, atol=1e-2
        )


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.regression
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
def test_gru():
    class GruNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mygru = torch.nn.GRU(7, 3, 1, bidirectional=False)

        def forward(self, input, initial_state):
            return self.mygru(input, initial_state)

    with torch.backends.cudnn.flags(enabled=use_cudnn):
        input = torch.randn(5, 3, 7)
        h0 = torch.randn(1, 3, 3)

        input_supa = input.clone().supa()
        h0_supa = h0.clone().supa()

        cpu_rnn = GruNet()
        supa_rnn = copy.deepcopy(cpu_rnn).supa()

        cpu_output = cpu_rnn(input, h0)
        supa_output = supa_rnn(input_supa, h0_supa)

        cpu_grad = torch.randn(5, 3, 3)
        supa_grad = cpu_grad.clone().supa()

        cpu_output[0].backward(cpu_grad)
        supa_output[0].backward(supa_grad)

        assert_allclose(cpu_output[0], supa_output[0], rtol=5e-4, atol=1e-3)
        assert_allclose(cpu_grad, supa_grad, rtol=5e-4, atol=1e-3)
