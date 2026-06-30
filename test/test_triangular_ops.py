# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

params = [
    pytest.param(
        [12, 12],
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [32, 32],
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [2, 2],
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param([512, 512], marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param([1024, 1024], marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param([256, 256], marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param([255, 255], marks=[pytest.mark.gcuStress]),
    pytest.param([1023, 1023], marks=[pytest.mark.gcuStress]),
    pytest.param([513, 513], marks=[pytest.mark.gcuStress]),
]

params_triu = [
    pytest.param(
        [12, 12],
        0,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [23, 23],
        0,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [32, 32],
        0,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        [2, 2],
        0,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param([512, 512], 0, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param([1024, 1024], 0, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param([256, 256], 0, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param([255, 255], 0, marks=[pytest.mark.gcuStress]),
    pytest.param([1023, 1023], 0, marks=[pytest.mark.gcuStress]),
    pytest.param([513, 513], 0, marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.float32, torch.bfloat16, torch.float16]
cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


class TestTril:

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_tril_out(self, shape, dtype):
        cpu_input, supa_input = create_random_tensor(
            shape, dtype=dtype, requires_grad=False
        )
        cpu_output = torch.empty(shape).to(dtype)
        supa_output = torch.empty(shape).to(dtype).to("supa")
        torch.tril(cpu_input, out=cpu_output)
        torch.tril(supa_input, out=supa_output)

        assert_allclose(cpu_output, supa_output, rtol=1e-5, atol=5e-5)

    @pytest.mark.parametrize("shape", params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_tril(self, shape, dtype):
        cpu_input, supa_input = create_random_tensor(
            shape, dtype=dtype, requires_grad=False
        )

        cpu_output = torch.tril(cpu_input)
        supa_output = torch.tril(supa_input)

        assert_allclose(cpu_output, supa_output, rtol=1e-5, atol=5e-5)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    def test_tril_bf16type(self):
        cpu_input, supa_input = create_random_tensor(
            [1, 512, 512], dtype=torch.bfloat16, requires_grad=False
        )

        cpu_output = torch.tril(cpu_input)
        supa_output = torch.tril(supa_input)

        assert_allclose(cpu_output, supa_output, rtol=1e-2, atol=5e-2)


class TestTriu:

    @pytest.mark.parametrize("shape, diagonal", params_triu)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_triu_out(self, shape, diagonal, dtype):
        cpu_input, supa_input = create_random_tensor(
            shape, dtype=dtype, requires_grad=False
        )
        cpu_output = torch.empty(shape).to(dtype)
        supa_output = torch.empty(shape).to(dtype).to("supa")
        torch.triu(cpu_input, diagonal=diagonal, out=cpu_output)
        torch.triu(supa_input, diagonal=diagonal, out=supa_output)

        assert_allclose(cpu_output, supa_output, rtol=0, atol=0)

    @pytest.mark.parametrize("shape, diagonal", params_triu)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_triu(self, shape, diagonal, dtype):
        cpu_input, supa_input = create_random_tensor(
            shape, dtype=dtype, requires_grad=False
        )

        cpu_output = torch.triu(cpu_input, diagonal=diagonal)
        supa_output = torch.triu(supa_input, diagonal=diagonal)

        assert_allclose(cpu_output, supa_output, rtol=0, atol=0)
