# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import copy
import math

# noqa
import pytest
import torch
from torch_supa.utils import torch_version_ge
from torch_supa.testing.common_utils import (
    assert_allclose,
    assert_equal,
    create_random_tensor,
    getDefaultRtolAndAtol,
)

ATOL = 8 * 1e-3
RTOL = 1 * 1e-5

supa_device = torch.device("supa")

# bias_shape, mat1_shape, mat2_shape
# shapes = [
#     [(1, 1000), (256, 2048), (2048, 1000)],
#     [(1, 1000), (128, 4096), (4096, 1000)],
#     [(1, 4096), (1, 255), (255, 4096)],
#     [(1, 103), (101, 97), (97, 103)]
# ]

shapes = [
    [(1, 5), (3, 4), (4, 5)],
]

betas = [
    1.0,
]

alphas = [
    1.0,
]

dtypes = [torch.float32, torch.bfloat16, torch.float16]

requires_grads = [False, True]

c_dtype_input_dtypes = [torch.float16, torch.bfloat16]

tf32_enables = [True, False]


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
@pytest.mark.parametrize("bias_shape, lshape, rshape", shapes)
@pytest.mark.parametrize("beta", betas)
@pytest.mark.parametrize("alpha", alphas)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("requires_grad", requires_grads)
def test_addmm(bias_shape, lshape, rshape, alpha, beta, dtype, requires_grad):
    bias_cpu, bias_supa = create_random_tensor(
        bias_shape, dtype=dtype, requires_grad=requires_grad
    )
    left_cpu, left_supa = create_random_tensor(
        lshape, dtype=dtype, requires_grad=requires_grad
    )
    right_cpu, right_supa = create_random_tensor(
        rshape, dtype=dtype, requires_grad=requires_grad
    )
    cpu_out = torch.addmm(bias_cpu, left_cpu, right_cpu, beta=beta, alpha=alpha)
    supa_out = torch.addmm(bias_supa, left_supa, right_supa, beta=beta, alpha=alpha)
    if not requires_grad:
        assert_allclose(cpu_out, supa_out, rtol=RTOL, atol=ATOL, equal_nan=False)
    else:
        cpu_grad, supa_grad = create_random_tensor(
            cpu_out.shape, dtype=dtype, requires_grad=False
        )
        cpu_out.backward(cpu_grad)
        bias_cpu_grad = bias_cpu.grad
        left_cpu_grad = left_cpu.grad
        right_cpu_grad = right_cpu.grad

        supa_out.backward(supa_grad)
        bias_supa_grad = bias_supa.grad
        left_supa_grad = left_supa.grad
        right_supa_grad = right_supa.grad
        assert_allclose(
            bias_cpu_grad, bias_supa_grad, rtol=RTOL, atol=ATOL, equal_nan=False
        )
        assert_allclose(
            left_cpu_grad, left_supa_grad, rtol=RTOL, atol=ATOL, equal_nan=False
        )
        assert_allclose(
            right_cpu_grad, right_supa_grad, rtol=RTOL, atol=ATOL, equal_nan=False
        )


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
@pytest.mark.parametrize("bias_shape, lshape, rshape", shapes)
@pytest.mark.parametrize("beta", betas)
@pytest.mark.parametrize("alpha", alphas)
@pytest.mark.parametrize("dtype", dtypes)
def test_addmm_activation(bias_shape, lshape, rshape, alpha, beta, dtype):
    bias_cpu, bias_supa = create_random_tensor(
        bias_shape, dtype=dtype, requires_grad=False
    )
    left_cpu, left_supa = create_random_tensor(lshape, dtype=dtype, requires_grad=False)
    right_cpu, right_supa = create_random_tensor(
        rshape, dtype=dtype, requires_grad=False
    )
    cpu_out = torch._addmm_activation(
        bias_cpu, left_cpu, right_cpu, beta=beta, alpha=alpha, use_gelu=True
    )
    supa_out = torch._addmm_activation(
        bias_supa, left_supa, right_supa, beta=beta, alpha=alpha, use_gelu=True
    )
    assert_allclose(cpu_out, supa_out, rtol=RTOL, atol=ATOL, equal_nan=False)


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
@pytest.mark.parametrize("bias_shape, lshape, rshape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("requires_grad", requires_grads)
def test_mm(bias_shape, lshape, rshape, dtype, requires_grad):
    left_cpu, left_supa = create_random_tensor(
        lshape, dtype=dtype, requires_grad=requires_grad
    )
    right_cpu, right_supa = create_random_tensor(
        rshape, dtype=dtype, requires_grad=requires_grad
    )
    cpu_out = torch.mm(left_cpu, right_cpu)
    supa_out = torch.mm(left_supa, right_supa)
    if not requires_grad:
        assert_allclose(cpu_out, supa_out, rtol=RTOL, atol=ATOL, equal_nan=False)
    else:
        cpu_grad, supa_grad = create_random_tensor(
            cpu_out.shape, dtype=dtype, requires_grad=False
        )
        cpu_out.backward(cpu_grad)
        left_cpu_grad = left_cpu.grad
        right_cpu_grad = right_cpu.grad

        supa_out.backward(supa_grad)
        left_supa_grad = left_supa.grad
        right_supa_grad = right_supa.grad
        assert_allclose(
            left_cpu_grad, left_supa_grad, rtol=RTOL, atol=ATOL, equal_nan=False
        )
        assert_allclose(
            right_cpu_grad, right_supa_grad, rtol=RTOL, atol=ATOL, equal_nan=False
        )


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
@pytest.mark.skipif(not torch_version_ge(2, 8, 0), reason="out_dtype matmul APIs require torch >= 2.8")
@pytest.mark.parametrize("bias_shape, lshape, rshape", shapes)
@pytest.mark.parametrize("dtype", c_dtype_input_dtypes)
def test_mm_out_dtype_float(bias_shape, lshape, rshape, dtype):
    left_cpu, left_supa = create_random_tensor(lshape, dtype=dtype, requires_grad=False)
    right_cpu, right_supa = create_random_tensor(rshape, dtype=dtype, requires_grad=False)
    cpu_out = torch.mm(left_cpu.float(), right_cpu.float())
    supa_out = torch.mm(left_supa, right_supa, out_dtype=torch.float32)
    assert cpu_out.dtype == torch.float32
    assert supa_out.dtype == torch.float32
    assert_allclose(cpu_out, supa_out, rtol=RTOL, atol=ATOL, equal_nan=False)


shapes_200 = [
    [(3, 4, 4), (3, 4, 5), (3, 5, 4)],
]


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
@pytest.mark.parametrize("input_shape, batch1_shape, batch2_shape", shapes_200)
@pytest.mark.parametrize("beta", betas)
@pytest.mark.parametrize("alpha", alphas)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("requires_grad", requires_grads)
def test_baddbmm(
    input_shape, batch1_shape, batch2_shape, alpha, beta, dtype, requires_grad
):
    input_cpu, input_supa = create_random_tensor(
        input_shape, dtype=dtype, requires_grad=requires_grad
    )
    batch1_cpu, batch1_supa = create_random_tensor(
        batch1_shape, dtype=dtype, requires_grad=requires_grad
    )
    batch2_cpu, batch2_supa = create_random_tensor(
        batch2_shape, dtype=dtype, requires_grad=requires_grad
    )
    cpu_out = torch.baddbmm(input_cpu, batch1_cpu, batch2_cpu, beta=beta, alpha=alpha)
    supa_out = torch.baddbmm(
        input_supa, batch1_supa, batch2_supa, beta=beta, alpha=alpha
    )
    if not requires_grad:
        assert_allclose(cpu_out, supa_out, rtol=RTOL, atol=ATOL, equal_nan=False)
    else:
        cpu_grad, supa_grad = create_random_tensor(
            cpu_out.shape, dtype=dtype, requires_grad=False
        )
        cpu_out.backward(cpu_grad)
        input_cpu_grad = input_cpu.grad
        batch1_cpu_grad = batch1_cpu.grad
        batch2_cpu_grad = batch2_cpu.grad

        supa_out.backward(supa_grad)
        input_supa_grad = input_supa.grad
        batch1_supa_grad = batch1_supa.grad
        batch2_supa_grad = batch2_supa.grad
        assert_allclose(
            input_cpu_grad, input_supa_grad, rtol=RTOL, atol=ATOL, equal_nan=False
        )
        assert_allclose(
            batch1_cpu_grad, batch1_supa_grad, rtol=RTOL, atol=ATOL, equal_nan=False
        )
        assert_allclose(
            batch2_cpu_grad, batch2_supa_grad, rtol=RTOL, atol=ATOL, equal_nan=False
        )


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
@pytest.mark.parametrize("input_shape, batch1_shape, batch2_shape", shapes_200)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("requires_grad", requires_grads)
def test_bmm(input_shape, batch1_shape, batch2_shape, dtype, requires_grad):
    batch1_cpu, batch1_supa = create_random_tensor(
        batch1_shape, dtype=dtype, requires_grad=requires_grad
    )
    batch2_cpu, batch2_supa = create_random_tensor(
        batch2_shape, dtype=dtype, requires_grad=requires_grad
    )
    cpu_out = torch.bmm(batch1_cpu, batch2_cpu)
    supa_out = torch.bmm(batch1_supa, batch2_supa)
    if not requires_grad:
        assert_allclose(cpu_out, supa_out, rtol=RTOL, atol=ATOL, equal_nan=False)
    else:
        cpu_grad, supa_grad = create_random_tensor(
            cpu_out.shape, dtype=dtype, requires_grad=False
        )
        cpu_out.backward(cpu_grad)
        batch1_cpu_grad = batch1_cpu.grad
        batch2_cpu_grad = batch2_cpu.grad

        supa_out.backward(supa_grad)
        batch1_supa_grad = batch1_supa.grad
        batch2_supa_grad = batch2_supa.grad
        assert_allclose(
            batch1_cpu_grad, batch1_supa_grad, rtol=RTOL, atol=ATOL, equal_nan=False
        )
        assert_allclose(
            batch2_cpu_grad, batch2_supa_grad, rtol=RTOL, atol=ATOL, equal_nan=False
        )


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
@pytest.mark.skipif(not torch_version_ge(2, 8, 0), reason="out_dtype matmul APIs require torch >= 2.8")
@pytest.mark.parametrize("input_shape, batch1_shape, batch2_shape", shapes_200)
@pytest.mark.parametrize("dtype", c_dtype_input_dtypes)
def test_bmm_out_dtype_float(input_shape, batch1_shape, batch2_shape, dtype):
    batch1_cpu, batch1_supa = create_random_tensor(
        batch1_shape, dtype=dtype, requires_grad=False
    )
    batch2_cpu, batch2_supa = create_random_tensor(
        batch2_shape, dtype=dtype, requires_grad=False
    )
    cpu_out = torch.bmm(batch1_cpu.float(), batch2_cpu.float())
    supa_out = torch.bmm(batch1_supa, batch2_supa, out_dtype=torch.float32)
    assert cpu_out.dtype == torch.float32
    assert supa_out.dtype == torch.float32
    assert_allclose(cpu_out, supa_out, rtol=RTOL, atol=ATOL, equal_nan=False)


dot_shapes = [
    [
        3,
    ],
]


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
@pytest.mark.parametrize("input_shape", dot_shapes)
@pytest.mark.parametrize("dtype", dtypes)
def test_dot(input_shape, dtype):
    cpu_input, supa_input = create_random_tensor(input_shape, dtype=dtype)
    output_cpu = torch.dot(cpu_input, cpu_input)
    output_supa = torch.dot(supa_input, supa_input)

    assert_allclose(output_cpu, output_supa, rtol=1e-5, atol=5e-5)

    cpu_out = torch.empty((), dtype=dtype)
    supa_out = cpu_out.supa()
    torch.dot(cpu_input, cpu_input, out=cpu_out)
    torch.dot(supa_input, supa_input, out=supa_out)
    assert_allclose(cpu_out, supa_out, rtol=1e-5, atol=5e-5)


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.regression
@pytest.mark.parametrize("input_shape", dot_shapes)
@pytest.mark.parametrize("dtype", dtypes)
def test_vdot(input_shape, dtype):
    cpu_input, supa_input = create_random_tensor(input_shape, dtype=dtype)
    output_cpu = torch.vdot(cpu_input, cpu_input)
    output_supa = torch.vdot(supa_input, supa_input)

    assert_allclose(output_cpu, output_supa, rtol=1e-5, atol=5e-5)

    cpu_out = torch.empty((), dtype=dtype)
    supa_out = cpu_out.supa()
    torch.vdot(cpu_input, cpu_input, out=cpu_out)
    torch.vdot(supa_input, supa_input, out=supa_out)
    assert_allclose(cpu_out, supa_out, rtol=1e-5, atol=5e-5)


mmv_shapes = [
    [(2,), (2, 3), (3,)],
]


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
@pytest.mark.parametrize("M, mat, vec", mmv_shapes)
@pytest.mark.parametrize("beta", betas)
@pytest.mark.parametrize("alpha", alphas)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("requires_grad", requires_grads)
def test_addmv(M, mat, vec, alpha, beta, dtype, requires_grad):
    bias_cpu, bias_supa = create_random_tensor(
        M, dtype=dtype, requires_grad=requires_grad
    )
    left_cpu, left_supa = create_random_tensor(
        mat, dtype=dtype, requires_grad=requires_grad
    )
    right_cpu, right_supa = create_random_tensor(
        vec, dtype=dtype, requires_grad=requires_grad
    )
    cpu_out = torch.addmv(bias_cpu, left_cpu, right_cpu, beta=beta, alpha=alpha)
    supa_out = torch.addmv(bias_supa, left_supa, right_supa, beta=beta, alpha=alpha)
    if not requires_grad:
        assert_allclose(cpu_out, supa_out, rtol=RTOL, atol=ATOL, equal_nan=False)
    else:
        cpu_grad, supa_grad = create_random_tensor(
            cpu_out.shape, dtype=dtype, requires_grad=False
        )
        cpu_out.backward(cpu_grad)
        bias_cpu_grad = bias_cpu.grad
        left_cpu_grad = left_cpu.grad
        right_cpu_grad = right_cpu.grad

        supa_out.backward(supa_grad)
        bias_supa_grad = bias_supa.grad
        left_supa_grad = left_supa.grad
        right_supa_grad = right_supa.grad
        assert_allclose(
            bias_cpu_grad, bias_supa_grad, rtol=RTOL, atol=ATOL, equal_nan=False
        )
        assert_allclose(
            left_cpu_grad, left_supa_grad, rtol=RTOL, atol=ATOL, equal_nan=False
        )
        assert_allclose(
            right_cpu_grad, right_supa_grad, rtol=RTOL, atol=ATOL, equal_nan=False
        )


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
@pytest.mark.parametrize("bias_shape, lshape, rshape", shapes)
@pytest.mark.parametrize("requires_grad", requires_grads)
def test_int_mm(bias_shape, lshape, rshape, requires_grad):
    cpu_w = torch.randint(-128, 127, (32, 32), dtype=torch.int32, device="cpu")
    cpu_x = torch.randint(-128, 127, (32, 32), dtype=torch.int32, device="cpu")
    supa_w = copy.deepcopy(cpu_w).to(torch.int8).to("supa")
    supa_x = copy.deepcopy(cpu_x).to(torch.int8).to("supa")
    cpu_out = torch.mm(cpu_w, cpu_x)
    supa_out = torch._int_mm(supa_w, supa_x)

    assert_allclose(cpu_out, supa_out, rtol=RTOL, atol=ATOL, equal_nan=False)


dtypesIfCUDA = [torch.float32, torch.bfloat16, torch.half]


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
@pytest.mark.parametrize("dtype", dtypesIfCUDA)
def test_blas_empty(dtype, device=supa_device):
    import math

    def fn(torchfn, *args, test_out=False, **kwargs):
        def call_torch_fn(*args, **kwargs):
            return torchfn(
                *tuple(
                    (
                        torch.randn(shape, device=device).to(dtype)
                        if isinstance(shape, tuple)
                        else shape
                    )
                    for shape in args
                ),
                **kwargs,
            )

        result = call_torch_fn(*args, **kwargs)
        if not test_out:
            return result
        else:
            out = torch.full_like(result, math.nan).to(dtype)
            out1 = call_torch_fn(*args, **kwargs, out=out)
            return out

    # mm, addmm
    assert (0, 0) == fn(torch.mm, (0, 0), (0, 0)).shape
    assert (0, 5) == fn(torch.mm, (0, 0), (0, 5)).shape
    assert (5, 0) == fn(torch.mm, (5, 0), (0, 0)).shape
    assert (3, 0) == fn(torch.mm, (3, 2), (2, 0)).shape
    assert_equal(
        torch.zeros((5, 6), device=device).cpu().to(torch.float),
        fn(torch.mm, (5, 0), (0, 6)).cpu().to(torch.float),
    )
    assert_equal(
        torch.zeros((5, 6), device=device).cpu().to(torch.float),
        fn(torch.mm, (5, 0), (0, 6), test_out=True).cpu().to(torch.float),
    )

    assert (0, 0) == fn(torch.addmm, (0, 0), (0, 0), (0, 0)).shape
    assert (0, 1) == fn(torch.addmm, (1,), (0, 17), (17, 1)).shape
    t = torch.randn((5, 6), device=device).to(dtype)
    assert_equal(
        t.cpu().to(torch.float),
        fn(torch.addmm, t, (5, 0), (0, 6)).cpu().to(torch.float),
    )
    assert_equal(
        t.cpu().to(torch.float),
        fn(torch.addmm, t, (5, 0), (0, 6), test_out=True).cpu().to(torch.float),
    )

    # mv, addmv
    assert (0,) == fn(torch.mv, (0, 0), (0,)).shape
    assert (0,) == fn(torch.mv, (0, 2), (2,)).shape
    assert_equal(
        torch.zeros((3,), device=device).to(dtype).cpu().to(torch.float),
        fn(torch.mv, (3, 0), (0,)).cpu().to(torch.float),
    )
    assert_equal(
        torch.zeros((3,), device=device).to(dtype).cpu().to(torch.float),
        fn(torch.mv, (3, 0), (0,), test_out=True).cpu().to(torch.float),
    )

    assert (0,) == fn(torch.addmv, (0,), (0, 0), (0,)).shape
    t = torch.randn((3,), device=device).to(dtype)
    assert_equal(
        t.to(dtype).cpu().to(torch.float),
        fn(torch.addmv, t, (3, 0), (0,)).cpu().to(torch.float),
    )
    assert_equal(
        t.to(dtype).cpu().to(torch.float),
        fn(torch.addmv, t, (3, 0), (0,), test_out=True).cpu().to(torch.float),
    )

    # bmm, baddbmm
    assert (0, 0, 0) == fn(torch.bmm, (0, 0, 0), (0, 0, 0)).shape
    assert (3, 0, 5) == fn(torch.bmm, (3, 0, 0), (3, 0, 5)).shape
    assert (0, 5, 6) == fn(torch.bmm, (0, 5, 0), (0, 0, 6)).shape
    assert_equal(
        torch.zeros((3, 5, 6), device=device).to(dtype).cpu().to(torch.float),
        fn(torch.bmm, (3, 5, 0), (3, 0, 6)).cpu().to(torch.float),
    )
    assert_equal(
        torch.zeros((3, 5, 6), device=device).to(dtype).cpu().to(torch.float),
        fn(torch.bmm, (3, 5, 0), (3, 0, 6), test_out=True).cpu().to(torch.float),
    )

    assert (0, 0, 0) == fn(torch.baddbmm, (0, 0, 0), (0, 0, 0), (0, 0, 0)).shape
    assert (3, 0, 5) == fn(torch.baddbmm, (3, 0, 5), (3, 0, 0), (3, 0, 5)).shape
    assert (0, 5, 6) == fn(torch.baddbmm, (0, 5, 6), (0, 5, 0), (0, 0, 6)).shape
    assert (3, 5, 6) == fn(torch.baddbmm, (3, 5, 6), (3, 5, 0), (3, 0, 6)).shape
    c = torch.arange(30, dtype=torch.float32, device=device).to(dtype).reshape(3, 2, 5)
    assert_equal(
        -2 * c.cpu().to(torch.float),
        fn(torch.baddbmm, c, (3, 2, 0), (3, 0, 5), beta=-2).cpu().to(torch.float),
    )  # Issue #33467
    assert_equal(
        -2 * c.cpu().to(torch.float),
        fn(torch.baddbmm, c, (3, 2, 0), (3, 0, 5), beta=-2, test_out=True)
        .cpu()
        .to(torch.float),
    )  # Issue #33467

    # addbmm
    assert (0, 0) == fn(torch.addbmm, (0, 0), (0, 0, 0), (0, 0, 0)).shape
    assert (0, 5) == fn(torch.addbmm, (0, 5), (3, 0, 0), (3, 0, 5)).shape
    t = torch.randn((5, 6), device=device).to(dtype)
    assert_equal(
        t.cpu().to(torch.float),
        fn(torch.addbmm, t, (0, 5, 0), (0, 0, 6)).cpu().to(torch.float),
    )
    assert_equal(
        t.cpu().to(torch.float),
        fn(torch.addbmm, t, (0, 5, 0), (0, 0, 6), test_out=True).cpu().to(torch.float),
    )

    # matmul
    # cublasSdot CUBLAS_STATUS_INVALID_VALUE
    # assert_equal(torch.tensor(0., device=device).cpu(), fn(torch.matmul, (0,), (0,)).cpu())
    # assert_equal(torch.tensor(0., device=device).cpu(), fn(torch.matmul, (0,), (0,), test_out=True).cpu())
    # cublasSdot CUBLAS_STATUS_INVALID_VALUE

    assert (0, 0) == fn(torch.matmul, (0, 0), (0, 0)).shape
    assert (0, 0, 0) == fn(torch.matmul, (0, 0, 0), (0, 0, 0)).shape
    assert (5, 0, 0) == fn(torch.matmul, (5, 0, 0), (5, 0, 0)).shape
    assert_equal(
        torch.zeros((5, 3, 4), device=device).to(dtype).cpu().to(torch.float),
        fn(torch.matmul, (5, 3, 0), (5, 0, 4)).cpu().to(torch.float),
    )
    assert_equal(
        torch.zeros((5, 3, 4), device=device).to(dtype).cpu().to(torch.float),
        fn(torch.matmul, (5, 3, 0), (5, 0, 4), test_out=True).cpu().to(torch.float),
    )

    # dot
    # cublasSdot cublasDotEx
    # cublasDotEx CUBLAS_STATUS_NOT_SUPPORTED
    # assert_equal(torch.tensor(0., device=device).to(dtype).cpu().to(torch.float), fn(torch.dot, (0,), (0,)).cpu().to(torch.float))
    # assert_equal(torch.tensor(0., device=device).to(dtype).cpu().to(torch.float), fn(torch.dot, (0,), (0,), test_out=True).cpu().to(torch.float))
    # assert(torch.tensor(0., device=device).shape == fn(torch.dot, (3,), (3,)).shape)
    # assert(torch.tensor(0., device=device).shape == fn(torch.dot, (3,), (3,), test_out=True).shape)

    # vdot
    # t = torch.randn((3,), device=device).to(dtype)
    # assert_equal(torch.tensor(0., device=device).to(dtype).cpu().to(torch.float), fn(torch.vdot, (0,), (0,)).cpu().to(torch.float))
    # assert_equal(torch.tensor(0., device=device).to(dtype).cpu().to(torch.float), fn(torch.vdot, (0,), (0,), test_out=True).cpu().to(torch.float))
    # assert(torch.tensor(0., device=device).shape == fn(torch.vdot, (3,), (3,)).shape)
    # assert(torch.tensor(0., device=device).shape == fn(torch.vdot, (3,), (3,), test_out=True).shape)
    # cublasDotEx CUBLAS_STATUS_NOT_SUPPORTED
    # cublasSdot CUBLAS_STATUS_INVALID_VALUE


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.skip(
    reason="bfloat16 CUBLAS_STATUS_INTERNAL_ERROR when calling `cublasGemmEx`,\
                          half CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasSgemmEx`"
)
@pytest.mark.regression
@pytest.mark.parametrize("dtype", dtypes)
def test_corner_cases_of_cublasltmatmul(dtype, device=supa_device):
    # common case
    M = torch.randn(128, device=device).to(dtype)
    m1 = torch.randn(2048, 2400, device=device).to(dtype)
    m2 = torch.randn(128, 2400, device=device).to(dtype)
    torch.nn.functional.linear(m1, m2, M)
    # Ntrans_B has ld >> rows
    m1 = torch.rand([128, 2400]).to(dtype).to(device).t()
    m2 = torch.rand([2048, 25272]).to(dtype).to(device).t()[21940:24340]
    M = torch.rand([128]).to(dtype).to(device)
    torch.addmm(M, m2.t(), m1)
    # trans_A has ld >> rows
    m1 = torch.rand([128, 25272]).to(dtype).to(device)[:, 21940:24340].t()
    m2 = torch.randn(2048, 2400, device=device).to(dtype)
    M = torch.rand([128]).to(dtype).to(device)
    torch.addmm(M, m2, m1)
    # large tensor dim > 65535
    M = torch.randn(16, device=device).to(dtype)
    m1 = torch.randn(32, 131071, device=device).to(dtype)
    m2 = torch.randn(16, 131071, device=device).to(dtype)
    torch.nn.functional.linear(m1, m2, M)


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
@pytest.mark.skip(reason="Hang Check in SQ issue instruction of 10000000 cycle.")
@pytest.mark.parametrize("dtype", dtypesIfCUDA)
def test_blas_alpha_beta_empty(dtype, device=supa_device):
    value = 11
    input = torch.full((2,), value, dtype=dtype, device=device)
    mat = torch.ones((2, 0), dtype=dtype, device=device)
    vec = torch.ones((0,), dtype=dtype, device=device)
    out = torch.empty((2,), dtype=dtype, device=device)
    if dtype.is_complex:
        alpha = 6 + 7j
        beta = 3 + 4j
    else:
        alpha = 6
        beta = 3
    rtol, atol = getDefaultRtolAndAtol(dtype)
    rtol, atol = max(0.0, rtol), max(0.0, atol)
    assert_allclose(
        torch.full((2,), beta * value, dtype=dtype, device=device).cpu(),
        torch.addmv(input=input, mat=mat, vec=vec, alpha=alpha, beta=beta).cpu(),
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        torch.full((2,), beta * value, dtype=dtype, device=device).cpu(),
        torch.addmv(
            input=input, mat=mat, vec=vec, alpha=alpha, beta=beta, out=out
        ).cpu(),
        rtol=rtol,
        atol=atol,
    )

    # torch.addmm
    input = torch.full((2, 3), value, dtype=dtype, device=device)
    mat2 = torch.ones((0, 3), dtype=dtype, device=device)
    out = torch.empty((2, 3), dtype=dtype, device=device)
    assert_allclose(
        torch.full((2, 3), beta * value, dtype=dtype, device=device).cpu(),
        torch.addmm(input=input, mat1=mat, mat2=mat2, alpha=alpha, beta=beta).cpu(),
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        torch.full((2, 3), beta * value, dtype=dtype, device=device).cpu(),
        torch.addmm(
            input=input, mat1=mat, mat2=mat2, alpha=alpha, beta=beta, out=out
        ).cpu(),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.skip(reason="Hang Check in SQ issue instruction.")
@pytest.mark.regression
def test_blas_nan_out(device=supa_device):
    # These functions should work correctly with NaN filled outputs,
    # but need special handling, see [NOTE: cpu_zero]
    b = 3
    n = 5
    m = 7
    p = 11

    # torch.mv
    nm = torch.randn((m, n), device=device).t()
    _m = torch.randn((), device=device).expand(m)
    _m_out = torch.full((m,), float("nan"), device=device)
    assert_equal(torch.mv(nm, _m).cpu(), torch.mv(nm, _m, out=_m_out).cpu())

    # torch.mm
    mp = torch.randn((p, m), device=device).t()
    np_out = torch.full((n, p), float("nan"), device=device)
    assert_equal(torch.mm(nm, mp).cpu(), torch.mm(nm, mp, out=np_out).cpu())

    # torch.bmm
    bnm = torch.randn((b, m, n), device=device).transpose(1, 2)
    bmp = torch.randn((b, p, m), device=device).transpose(1, 2)
    bnp_out = torch.full((b, n, p), float("nan"), device=device)
    assert_equal(torch.bmm(bnm, bmp).cpu(), torch.bmm(bnm, bmp, out=bnp_out).cpu())


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
@pytest.mark.parametrize("tf32_enable", tf32_enables)
def test_linear(tf32_enable, device=supa_device):
    class M(torch.nn.Module):
        def __init__(self, bias):
            super().__init__()
            self.linear = torch.nn.Linear(64, 64, bias, device=device)

        def forward(self, x):
            return self.linear(x)

    for bias in [True, False]:
        with torch.backends.cudnn.flags(allow_tf32=tf32_enable):
            torch.backends.cuda.matmul.allow_tf32 = tf32_enable
            mod = M(bias=bias)
            v = torch.randn(64, 64).to(device=device)
            out = mod(v)


def _test_addmm_addmv(
    f,
    t,
    m,
    v,
    *,
    alpha=None,
    beta=None,
    transpose_out=False,
    activation=None,
    precisionOverride={},
    equal=False,
):
    dtype = t.dtype
    numpy_dtype = dtype
    if dtype in {torch.bfloat16, torch.half}:
        numpy_dtype = torch.float
    if dtype.is_complex:
        alpha = 0.9 + 0.3j if alpha is None else alpha
        beta = 0.5 + 0.6j if beta is None else beta
    else:
        alpha = 1.2 if alpha is None else alpha
        beta = 0.8 if beta is None else beta
    if activation == "gelu":
        res1 = f(t, m, v, alpha=alpha, beta=beta, use_gelu=True)
    else:
        res1 = f(t, m, v, alpha=alpha, beta=beta)
    res2 = torch.full_like(res1, math.nan)
    if transpose_out:
        res2 = res2.t().clone(memory_format=torch.contiguous_format).t()
    if activation == "gelu":
        f(t, m, v, alpha=alpha, beta=beta, out=res2, use_gelu=True)
    else:
        f(t, m, v, alpha=alpha, beta=beta, out=res2)
    res3 = alpha * (m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy())
    if beta != 0:
        res3 += (beta * t).to(numpy_dtype).cpu().numpy()
    if activation == "relu":
        res3 = res3 * (res3 > 0)
    elif activation == "gelu":
        res3_t = torch.from_numpy(res3).to(dtype)
        approximate = "tanh" if t.is_cuda else "none"
        res3_t = torch.nn.functional.gelu(res3_t, approximate=approximate)
        res3 = res3_t.to(numpy_dtype).cpu().numpy()
    else:
        assert activation is None, f"unsupported activation {activation}"
    res3 = torch.from_numpy(res3).to(dtype)
    if not equal:
        rtol, atol = getDefaultRtolAndAtol(dtype)
        rtol, atol = max(0.0, rtol), max(precisionOverride.get(dtype, 0), atol)
        assert_allclose(res1.cpu(), res2.cpu(), atol=atol, rtol=rtol, equal_nan=True)
        assert_allclose(res1.cpu(), res3.cpu(), atol=atol, rtol=rtol, equal_nan=True)
    else:
        assert_equal(res1.cpu(), res2.cpu())
        assert_equal(res1.cpu(), res3.cpu())


@pytest.mark.skip(
    reason="float32 addmv mismath out is nan maybe a bug\
                          bfloat16 CUBLAS_STATUS_INTERNAL_ERROR when calling `cublasGemmEx`,\
                          half CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasSgemmEx`"
)
@pytest.mark.parametrize("dtype", dtypesIfCUDA)
def test_addmv_impl(dtype, device=supa_device):
    precisionOverride = {torch.bfloat16: 1e-0, torch.float: 1e-4}
    # have to use torch.randn(...).to(bfloat16) instead of
    # torch.randn(..., dtype=bfloat16). randn does not support
    # bfloat16 yet.
    # "*0.2" to reduce errors for low precision
    ts = [
        0.2 * torch.randn(50).to(dtype).to(device),
        0.2 * torch.randn(1).to(dtype).to(device).expand(50),
    ]
    vs = [
        0.2 * torch.randn(100).to(dtype).to(device),
        0.2
        * torch.ones(1)
        .to(dtype)
        .to(device)
        .expand(100),  # to reduce errors for low precision
    ]
    ms = [
        # 0d
        0.2
        * torch.ones(
            (),
        )
        .to(dtype)
        .to(device)
        .expand(50, 100),  # to reduce errors for low precision
        # 1d
        0.2 * torch.randn((1, 100)).to(dtype).to(device).expand(50, 100),
        # this initialization reduces errors for low precision for broadcasted matrices
        # by making sure that intermediate and result values are exactly representable
        # in low precision type
        0.2
        * torch.randint(3, (50, 1), dtype=torch.float, device=device)
        .to(dtype)
        .expand(50, 100),
        # 2d
        0.2 * torch.randn((50, 100)).to(dtype).to(device),
        0.2 * torch.randn((100, 50)).to(dtype).to(device).t(),
    ]
    for m, v, t in itertools.product(ms, vs, ts):
        _test_addmm_addmv(torch.addmv, t, m, v, precisionOverride=precisionOverride)
    # Test beta=0, t=nan
    t = torch.full((50,), math.nan, device=device).to(dtype)
    for m, v in itertools.product(ms, vs):
        _test_addmm_addmv(
            torch.addmv, t, m, v, beta=0, precisionOverride=precisionOverride
        )


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
@pytest.mark.parametrize("dtype", dtypes)
# @pytest.mark.skip(reason="bfloat16 CUBLAS_STATUS_INTERNAL_ERROR when calling `cublasGemmEx`,\
#                           half CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasSgemmEx`")
def test_addmv_rowmajor_colmajor_incx_incy_lda(dtype, device=supa_device):
    # tests (o, s)*(s).  o is output size, s is summed size.
    o = 5
    s = 3
    a_data = torch.arange(1, o * s + 1, device=device, dtype=dtype).view(o, s)
    x_data = torch.arange(1, s + 1, 1, device=device, dtype=dtype)
    y_data = torch.ones(o, device=device, dtype=dtype)
    control = torch.tensor([15.0, 33.0, 51.0, 69.0, 87.0], device=device, dtype=dtype)

    def _test(row_major, incx, incy, lda_tail):
        if row_major:
            a_storage = torch.full(
                (o, s + lda_tail), float("nan"), device=device, dtype=dtype
            )
        else:
            a_storage = torch.full(
                (s, o + lda_tail), float("nan"), device=device, dtype=dtype
            ).permute(1, 0)
        a = a_storage[:o, :s].copy_(a_data)

        x_storage = torch.full((s, incx), float("nan"), device=device, dtype=dtype)
        x = x_storage[:, 0].copy_(x_data)

        y_storage = torch.full((o, incy), float("nan"), device=device, dtype=dtype)
        y = y_storage[:, 0].copy_(y_data)

        _test_addmm_addmv(torch.addmv, y, a, x)

    for row_major, incx, incy, lda_tail in itertools.product(
        (False, True), (1, 2), (1, 2), (0, 1)
    ):
        _test(row_major, incx, incy, lda_tail)


def _test_addmm_impl(func, activation, device, dtype, precisionOverride):
    M = torch.randn(10, 25, device=device).to(dtype)
    m1 = torch.randn(10, 50, device=device).to(dtype)
    m2 = torch.randn(50, 25, device=device).to(dtype)
    _test_addmm_addmv(
        func, M, m1, m2, activation=activation, precisionOverride=precisionOverride
    )

    # vector-shaped bias and beta=1 result in epilogue fusion in CUDA
    V = torch.randn(25, device=device).to(dtype)
    _test_addmm_addmv(
        func,
        V,
        m1,
        m2,
        beta=1,
        activation=activation,
        precisionOverride=precisionOverride,
    )

    # Test 0-strided
    M = torch.randn(10, 1, device=device).to(dtype).expand(10, 25)
    m1 = torch.randn(10, 1, device=device).to(dtype).expand(10, 50)
    m2 = torch.randn(50, 25, device=device).to(dtype)
    _test_addmm_addmv(
        func, M, m1, m2, activation=activation, precisionOverride=precisionOverride
    )

    # Test beta=0, M=nan
    M = torch.full((10, 25), math.nan, device=device).to(dtype)
    m1 = torch.randn(10, 50, device=device).to(dtype)
    m2 = torch.randn(50, 25, device=device).to(dtype)
    _test_addmm_addmv(
        func,
        M,
        m1,
        m2,
        beta=0,
        activation=activation,
        precisionOverride=precisionOverride,
    )

    # Test transpose
    for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):

        def maybe_transpose(cond, m):
            if not cond:
                return m
            return m.t().clone(memory_format=torch.contiguous_format).t()

        M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
        m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
        m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
        _test_addmm_addmv(
            func,
            M,
            m1,
            m2,
            transpose_out=t4,
            activation=activation,
            precisionOverride=precisionOverride,
        )

        if t1:
            # use vector V instead of matrix M for epilogue fusion in CUDA (doesn't depend on t1)
            _test_addmm_addmv(
                func,
                V,
                m1,
                m2,
                beta=1,
                transpose_out=t4,
                activation=activation,
                precisionOverride=precisionOverride,
            )


# CUBLAS_STATUS_INTERNAL_ERROR when calling `cublasGemmEx`
# CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasSgemmEx`
# @pytest.mark.skip(reason="`cublasGemmEx` `cublasSgemmEx` NOT_SUPPORTED")
@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.regression
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.parametrize("dtype", dtypes)
def test_addmm_impl(dtype, device=supa_device, tf32_enable=False):
    precisionOverride = {torch.float: 1e-4, torch.bfloat16: 0.6, torch.half: 1e-1}
    if tf32_enable:
        precisionOverride = {torch.float: 0.05, torch.bfloat16: 0.6, torch.half: 1e-1}
    _test_addmm_impl(
        torch.addmm, None, device, dtype, precisionOverride=precisionOverride
    )


@pytest.mark.skip(
    reason="`cublasGemmEx` `cublasSgemmEx` NOT_SUPPORTED\
                  Bug maybe CUBLAS_STATUS_INVALID_VALUE when calling `::cublasLtMatmulDescSetAttribute(descriptor(), attr, &value, sizeof(T))`"
)
@pytest.mark.parametrize("dtype", dtypesIfCUDA)
def test_addmm_relu(dtype, device=supa_device, tf32_enable=False):
    precisionOverride = {torch.float: 1e-4, torch.bfloat16: 5e-2, torch.half: 5e-2}
    if tf32_enable:
        precisionOverride = {torch.float: 0.05, torch.bfloat16: 0.6, torch.half: 1e-1}
    _test_addmm_impl(
        torch._addmm_activation,
        "relu",
        device,
        dtype,
        precisionOverride=precisionOverride,
    )


@pytest.mark.skip(
    reason="`cublasGemmEx` `cublasSgemmEx` NOT_SUPPORTED\
                  Bug maybe CUBLAS_STATUS_INVALID_VALUE when calling `::cublasLtMatmulDescSetAttribute(descriptor(), attr, &value, sizeof(T))`"
)
@pytest.mark.parametrize("dtype", dtypesIfCUDA)
def test_addmm_gelu(dtype, device=supa_device, tf32_enable=False):
    precisionOverride = {torch.float: 1e-4, torch.bfloat16: 5e-2, torch.half: 5e-2}
    if tf32_enable:
        precisionOverride = {torch.float: 0.05, torch.bfloat16: 0.6, torch.half: 1e-1}

    _test_addmm_impl(
        torch._addmm_activation,
        "gelu",
        device,
        dtype,
        precisionOverride=precisionOverride,
    )


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
def test_addmm_sizes(device=supa_device, dtype=torch.float32, tf32_enable=False):
    precisionOverride = {torch.float: 1e-4}
    if tf32_enable:
        precisionOverride = {torch.float: 0.005}

    for m in [0, 1, 25]:
        for n in [0, 1, 10]:
            for k in [0, 1, 8]:
                M = torch.randn(n, m, device=device).to(dtype)
                m1 = torch.randn(n, k, device=device).to(dtype)
                m2 = torch.randn(k, m, device=device).to(dtype)
                _test_addmm_addmv(
                    torch.addmm, M, m1, m2, precisionOverride=precisionOverride
                )


import itertools


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.regression
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.parametrize("dtype", [torch.float32])
# @pytest.mark.skip(reason="CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasGemmStridedBatchedEx`\
#                           CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasSgemmEx`")
def test_baddbmm_nan_input_with_zero_beta(dtype, device=supa_device):
    for shape in [[3, 2, 2], [2, 20, 20]]:
        mat1, mat2 = (torch.randn(shape, dtype=dtype, device=device) for _ in range(2))
        inputs = [
            torch.randn(shape, dtype=dtype, device=device),
            torch.randn(shape, dtype=dtype, device=device).fill_(torch.nan),
        ]
        outs = [
            None,
            torch.randn(shape, dtype=dtype, device=device),
            torch.randn(shape, dtype=dtype, device=device).fill_(torch.nan),
        ]
        options = itertools.product(inputs, outs)
        for input, out in options:
            y_ref = torch.bmm(mat1, mat2)
            y = torch.baddbmm(input, mat1, mat2, beta=0.0, out=out)
            assert_equal(y_ref.cpu(), y.cpu())


@pytest.mark.sanity
@pytest.mark.gcuSmoke
@pytest.mark.gcuSanity
@pytest.mark.gcuStress
@pytest.mark.regression
def test_baddbmm_input_dtypes_compatibility(device=supa_device, dtype=torch.float32):
    batch1 = torch.rand((1, 2, 2), dtype=torch.float32, device=device)
    batch2 = torch.rand((1, 2, 2), dtype=torch.float32, device=device)
    input_tensor = torch.rand((1, 2, 2), device=device).to(dtype)
    out = torch.randn((1, 2, 2), dtype=dtype, device=device).fill_(torch.nan)
    y_ref = torch.bmm(batch1, batch2)
    y = torch.baddbmm(input_tensor, batch1, batch2, beta=0.0, out=out)
    assert_equal(out.cpu(), y_ref.cpu())


import numpy as np
from torch.testing import make_tensor


class TestEinsum:

    def _check_einsum(self, *args, np_args=None):
        if np_args is None:
            np_args = [
                arg.cpu().numpy() if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ]
        ref = np.einsum(*np_args)
        res = torch.einsum(*args)
        rtol, atol = 0.016, 1e-4
        np.testing.assert_allclose(ref, res.cpu().detach().numpy(), rtol, atol, True)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.regression
    @pytest.mark.skip(reason="mismatch")
    def test_einsum(self, dtype=torch.float32, device=torch.device("cuda:0")):

        # Test cases from https://gist.github.com/rockt/15ee013889d65342088e9260a377dc8f
        x = make_tensor((5,), dtype=dtype, device=device)
        y = make_tensor((7,), dtype=dtype, device=device)
        A = make_tensor((3, 5), dtype=dtype, device=device)
        B = make_tensor((2, 5), dtype=dtype, device=device)
        C = make_tensor((2, 3, 5), dtype=dtype, device=device)
        D = make_tensor((2, 5, 7), dtype=dtype, device=device)
        E = make_tensor((7, 9), dtype=dtype, device=device)
        F = make_tensor((2, 3, 3, 5), dtype=dtype, device=device)
        G = make_tensor((5, 4, 6), dtype=dtype, device=device)
        H = make_tensor((4, 4), dtype=dtype, device=device)
        I = make_tensor((2, 3, 2), dtype=dtype, device=device)

        # Vector operations
        self._check_einsum("i->", x)  # sum
        self._check_einsum("i,i->", x, x)  # dot
        self._check_einsum("i,i->i", x, x)  # vector element-wisem mul
        self._check_einsum("i,j->ij", x, y)  # outer

        # Matrix operations
        self._check_einsum("ij->ji", A)  # transpose
        self._check_einsum("ij->j", A)  # row sum
        self._check_einsum("ij->i", A)  # col sum
        self._check_einsum("ij,ij->ij", A, A)  # matrix element-wise mul
        self._check_einsum("ij,j->i", A, x)  # matrix vector multiplication
        self._check_einsum("ij,kj->ik", A, B)  # matmul
        self._check_einsum("ij,ab->ijab", A, E)  # matrix outer product

        # Tensor operations
        self._check_einsum("Aij,Ajk->Aik", C, D)  # batch matmul
        self._check_einsum("ijk,jk->i", C, A)  # tensor matrix contraction
        self._check_einsum("aij,jk->aik", D, E)  # tensor matrix contraction
        self._check_einsum("abCd,dFg->abCFg", F, G)  # tensor tensor contraction
        self._check_einsum(
            "ijk,jk->ik", C, A
        )  # tensor matrix contraction with double indices
        self._check_einsum(
            "ijk,jk->ij", C, A
        )  # tensor matrix contraction with double indices
        self._check_einsum("ijk,ik->j", C, B)  # non contiguous
        self._check_einsum("ijk,ik->jk", C, B)  # non contiguous with double indices

        # Test diagonals
        self._check_einsum("ii", H)  # trace
        self._check_einsum("ii->i", H)  # diagonal
        self._check_einsum("iji->j", I)  # non-contiguous trace
        self._check_einsum(
            "ngrg...->nrg...", make_tensor((2, 1, 3, 1, 4), dtype=dtype, device=device)
        )

        # Test ellipsis
        self._check_einsum("i...->...", H)
        self._check_einsum("ki,...k->i...", A.t(), B)
        self._check_einsum("k...,jk->...", A.t(), B)
        self._check_einsum("...ik, ...j -> ...ij", C, x)
        self._check_einsum(
            "Bik,k...j->i...j", C, make_tensor((5, 3), dtype=dtype, device=device)
        )
        self._check_einsum(
            "i...j, ij... -> ...ij",
            C,
            make_tensor((2, 5, 2, 3), dtype=dtype, device=device),
        )

        # torch.bilinear with noncontiguous tensors
        l = make_tensor((5, 10), dtype=dtype, device=device, noncontiguous=True)
        r = make_tensor((5, 20), dtype=dtype, device=device, noncontiguous=True)
        w = make_tensor((15, 10, 20), dtype=dtype, device=device)
        self._check_einsum("bn,anm,bm->ba", l, w, r)

        # with strided tensors
        self._check_einsum("bn,Anm,bm->bA", l[:, ::2], w[:, ::2, ::2], r[:, ::2])

        # test multiple inputs
        self._check_einsum("...,be,b...,beg,gi,bc...->bi...", A, B, C, D, E, F)
