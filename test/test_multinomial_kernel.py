# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# noqa
import numpy as np
import pytest
import torch

br200_shape = [
    pytest.param(
        (1, 32),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.float32, torch.bfloat16, torch.float16]


def evalUniform(tensor_supa, uniform_min, uniform_max, threshold):
    supa_data = tensor_supa.cpu().detach().numpy()
    avg = supa_data.mean()
    avg_ref = (uniform_max - uniform_min) / 2 + uniform_min
    print(
        "\n avg: {} - avg_ref: {} - ae: {} \n".format(
            avg, avg_ref, np.abs(avg - avg_ref)
        )
    )
    assert (
        np.abs(avg - avg_ref) < threshold
    ), "Uniformly Distributed probability Not Equal 1/(to-from)"


@pytest.mark.parametrize("shape", br200_shape)
@pytest.mark.parametrize("dtype", dtypes)
def test_multinomial(shape, dtype):
    supa_in = torch.rand(shape, dtype=dtype, device="cpu").supa()
    supa_out = torch.multinomial(supa_in, 1)
    evalUniform(supa_out, 0, shape[1], shape[1])


@pytest.mark.parametrize("shape", br200_shape)
@pytest.mark.parametrize("dtype", dtypes)
def test_multinomial_out(shape, dtype):
    supa_in = torch.rand(shape, dtype=dtype, device="cpu").supa()
    supa_out = torch.rand((shape[0], 1)).to(torch.int64).supa()
    torch.multinomial(supa_in, 1, out=supa_out)
    evalUniform(supa_out, 0, shape[1], shape[1])
