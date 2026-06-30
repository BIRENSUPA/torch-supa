# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import numpy as np
import pytest
import torch

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor


br200_params = [
    pytest.param(
        (4, 3),
        0,
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 1024), 0, marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), 0, marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), 0, marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), 0, marks=[pytest.mark.gcuStress]),
]

data_type = [
    torch.float32,
    # torch.bfloat16,
    torch.float16,
    torch.int64,
    torch.int32,
]

RTOL = {
    torch.float32: 5e-5,
    torch.bfloat16: 5e-1,
    torch.float16: 2e-2,
    torch.int64: 0,
    torch.int32: 0,
}
ATOL = {
    torch.float32: 5e-5,
    torch.bfloat16: 5e-1,
    torch.float16: 1e-3,
    torch.int64: 0,
    torch.int32: 0,
}


@pytest.mark.parametrize("shape, dim", br200_params)
@pytest.mark.parametrize("dtype", data_type)
def test_cumsum(shape, dim, dtype):
    if dtype == torch.float32:
        requires_grad = True
    else:
        requires_grad = False

    cpu_in, supa_in = create_random_tensor(
        shape, dtype=dtype, requires_grad=requires_grad
    )
    cpu_grad, supa_grad = create_random_tensor(shape, dtype=dtype)

    cpu_out = torch.cumsum(cpu_in, dim)
    supa_out = torch.cumsum(supa_in, dim)

    # [NOTE: use numpy cumulative operation as golden data]
    #
    # In pytorch, the cpu implementation may choose `double` to do accumulation which has more
    # precision than `float`. In SUPA/CUDA, `float` is used as the accumulator type. Consequently,
    # the result of them might divergent significantly. Numpy operation should get a closer result
    # to SUPA.
    try:
        assert_allclose(cpu_out, supa_out, atol=ATOL[dtype], rtol=RTOL[dtype])
    except AssertionError:
        if dtype == torch.bfloat16:
            cpu_in_np = cpu_in.float().detach().numpy()
            cpu_out_np = cpu_in_np.cumsum(dim)
            assert_allclose(
                torch.from_numpy(cpu_out_np).to(torch.bfloat16),
                supa_out,
                atol=1e-2,
                rtol=0.5,
            )
        else:
            cpu_in_np = cpu_in.detach().numpy()
            cpu_out_np = cpu_in_np.cumsum(dim)
            assert_allclose(
                torch.from_numpy(cpu_out_np),
                supa_out,
                atol=ATOL[dtype],
                rtol=RTOL[dtype],
            )

    if requires_grad:
        cpu_out.backward(cpu_grad)
        supa_out.backward(supa_grad)

        try:
            assert_allclose(
                cpu_in.grad, supa_in.grad, atol=ATOL[dtype], rtol=RTOL[dtype]
            )
        except AssertionError:
            cpu_grad_np = cpu_grad.detach().numpy()
            cpu_in_grad_np = np.ascontiguousarray(
                np.flip(np.flip(cpu_grad_np, dim).cumsum(dim), dim)
            )
            assert_allclose(
                torch.from_numpy(cpu_in_grad_np),
                supa_in.grad,
                atol=ATOL[dtype],
                rtol=RTOL[dtype],
            )


@pytest.mark.parametrize("shape, dim", br200_params)
@pytest.mark.parametrize("dtype", data_type)
def test_cumprod(shape, dim, dtype):
    if dtype == torch.float32:
        requires_grad = True
    else:
        requires_grad = False

    cpu_in, supa_in = create_random_tensor(
        shape, dtype=dtype, requires_grad=requires_grad
    )
    cpu_grad, supa_grad = create_random_tensor(shape, dtype=dtype)

    cpu_out = torch.cumprod(cpu_in, dim)
    supa_out = torch.cumprod(supa_in, dim)

    # [NOTE: use numpy cumulative operation as golden data]
    #
    # In pytorch, the cpu implementation may choose `double` to do accumulation which has more
    # precision than `float`. In SUPA/CUDA, `float` is used as the accumulator type. Consequently,
    # the result of them might divergent significantly. Numpy operation should get a closer result
    # to SUPA.
    try:
        assert_allclose(cpu_out, supa_out, atol=ATOL[dtype], rtol=RTOL[dtype])
    except AssertionError:
        cpu_in_np = cpu_in.detach().numpy()
        cpu_out_np = cpu_in_np.cumprod(dim)
        assert_allclose(
            torch.from_numpy(cpu_out_np), supa_out, atol=ATOL[dtype], rtol=RTOL[dtype]
        )

    if requires_grad:
        cpu_out.backward(cpu_grad)
        supa_out.backward(supa_grad)

        try:
            assert_allclose(
                cpu_in.grad, supa_in.grad, atol=ATOL[dtype], rtol=RTOL[dtype]
            )
        except AssertionError:
            cpu_grad_np = cpu_grad.detach().numpy()
            cpu_in_grad_np = np.ascontiguousarray(
                np.flip(np.flip(cpu_grad_np, dim).cumsum(dim), dim)
            )
            assert_allclose(
                torch.from_numpy(cpu_in_grad_np),
                supa_in.grad,
                atol=ATOL[dtype],
                rtol=RTOL[dtype],
            )


@pytest.mark.parametrize("shape, dim", br200_params)
def test_logcumsumexp(shape, dim, dtype=torch.float32):
    if dtype == torch.float32:
        requires_grad = True
    else:
        requires_grad = False

    cpu_in, supa_in = create_random_tensor(
        shape, dtype=dtype, requires_grad=requires_grad
    )
    cpu_grad, supa_grad = create_random_tensor(shape, dtype=dtype)

    cpu_out = torch.logcumsumexp(cpu_in, dim)
    supa_out = torch.logcumsumexp(supa_in, dim)

    # [NOTE: use numpy cumulative operation as golden data]
    #
    # In pytorch, the cpu implementation may choose `double` to do accumulation which has more
    # precision than `float`. In SUPA/CUDA, `float` is used as the accumulator type. Consequently,
    # the result of them might divergent significantly. Numpy operation should get a closer result
    # to SUPA.
    try:
        assert_allclose(cpu_out, supa_out, atol=ATOL[dtype], rtol=RTOL[dtype])
    except AssertionError:
        cpu_in_np = cpu_in.detach().numpy()
        cpu_out_np = cpu_in_np.logcumsumexp(dim)
        assert_allclose(
            torch.from_numpy(cpu_out_np), supa_out, atol=ATOL[dtype], rtol=RTOL[dtype]
        )

    if requires_grad:
        cpu_out.backward(cpu_grad)
        supa_out.backward(supa_grad)

        assert_allclose(cpu_in.grad, supa_in.grad, atol=ATOL[dtype], rtol=RTOL[dtype])
