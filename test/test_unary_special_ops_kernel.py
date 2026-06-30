# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
import scipy.signal as signal
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

shapes = [
    pytest.param(
        (1,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (32,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (4,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param(
        (3, 4, 5),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
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
    pytest.param(
        (15,),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((512, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1024, 1024), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((1023, 511), marks=[pytest.mark.gcuStress]),
    pytest.param((1025, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((1028, 2135), marks=[pytest.mark.gcuStress]),
]

# func_types1: without implementation for 'Half' and 'BFloat16'
func_types1 = [
    "special.i1",
    "special.i1e",
    "special.ndtri",
    "special.erfcx",
    "special.log_ndtr",
]
func_types2 = ["i0", "special.i0e", "special.entr", "special.logit"]

input_types = ["scalar", "tensor"]
dtypes = [torch.float32]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


class TestSpecialOps:

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("input_type", input_types)
    @pytest.mark.parametrize("func_type", func_types1)
    def test_single_ptwise_float(self, shape, input_type, func_type):
        dtype = torch.float32
        pt_func = eval("torch." + func_type)
        if input_type == "scalar":
            in_cpu = torch.tensor(2.0, dtype=dtype).to(cpu_device)
            in_supa = torch.tensor(2.0, dtype=dtype).to(supa_device)
        elif input_type == "tensor":
            in_cpu, in_supa = create_random_tensor(shape, dtype)

        res_cpu = pt_func(in_cpu)
        res_supa = pt_func(in_supa)
        assert_allclose(
            res_cpu, res_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("input_type", input_types)
    @pytest.mark.parametrize("func_type", func_types2)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_single_ptwise(self, shape, input_type, func_type, dtype):
        pt_func = eval("torch." + func_type)
        if input_type == "scalar":
            in_cpu = torch.tensor(2.0, dtype=dtype).to(cpu_device)
            in_supa = torch.tensor(2.0, dtype=dtype).to(supa_device)
        elif input_type == "tensor":
            in_cpu, in_supa = create_random_tensor(shape, dtype)

        res_cpu = pt_func(in_cpu)
        res_supa = pt_func(in_supa)
        assert_allclose(
            res_cpu, res_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )


class TestSigmoid:
    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_sigmoid(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        y_cpu = torch.sigmoid(x_cpu)
        y_supa = torch.sigmoid(x_supa)
        x_cpu.sigmoid_()
        x_supa.sigmoid_()

        assert_allclose(
            y_cpu, y_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
        assert_allclose(
            x_cpu, x_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
        assert_allclose(
            x_supa.cpu(), y_supa, atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_sigmoid_bwd(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)
        y_cpu = torch.sigmoid(x_cpu)
        y_supa = torch.sigmoid(x_supa)

        cpu_grad, supa_grad = create_random_tensor(shape, dtype=dtype)
        y_cpu.backward(cpu_grad)
        cpu_in_grad = x_cpu.grad

        y_supa.backward(supa_grad)
        supa_in_grad = x_supa.grad
        assert_allclose(
            cpu_in_grad, supa_in_grad.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype]
        )


class TestSinc:
    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_sinc(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype)
        y_cpu = torch.sinc(x_cpu)
        y_supa = torch.sinc(x_supa)
        x_cpu.sinc_()
        x_supa.sinc_()

        assert_allclose(y_cpu, y_supa, rtol=1e-0, atol=5e-0, equal_nan=True)
        assert_allclose(x_cpu, x_supa, rtol=1e-0, atol=5e-0, equal_nan=True)
        assert_allclose(x_supa.cpu(), y_supa, rtol=1e-0, atol=5e-0, equal_nan=True)


class TestLogit:
    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_logit(self, shape, dtype):
        x_cpu, x_supa = create_random_tensor(shape, dtype=dtype, requires_grad=True)
        y_cpu = torch.logit(x_cpu)
        y_supa = torch.logit(x_supa)

        cpu_grad_data, supa_grad_data = create_random_tensor(
            y_cpu.shape, min_value=0, max_value=1, dtype=dtype
        )
        y_cpu.backward(cpu_grad_data)
        y_supa.backward(supa_grad_data)
        cpu_grad = x_cpu.grad.clone()
        supa_grad = x_supa.grad
        assert_allclose(
            y_cpu, y_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype], equal_nan=True
        )
        assert_allclose(
            cpu_grad,
            supa_grad.cpu(),
            atol=ATOL[dtype],
            rtol=RTOL[dtype],
            equal_nan=True,
        )


class TestErf:
    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_erf_ptwise(self, shape, dtype):
        in_cpu, in_supa = create_random_tensor(shape, dtype)

        out_cpu = torch.erf(in_cpu)
        out_supa = torch.erf(in_supa)
        in_cpu.erf_()
        in_supa.erf_()

        assert_allclose(in_cpu, in_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(in_supa.cpu(), out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_erfc_ptwise(self, shape, dtype):
        in_cpu, in_supa = create_random_tensor(shape, dtype)

        out_cpu = torch.erfc(in_cpu)
        out_supa = torch.erfc(in_supa)
        in_cpu.erfc_()
        in_supa.erfc_()

        assert_allclose(in_cpu, in_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(in_supa.cpu(), out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_erfinv_ptwise(self, shape, dtype):
        if dtype == torch.bfloat16:
            return
        in_cpu, in_supa = create_random_tensor(shape, dtype)

        out_cpu = torch.erfinv(in_cpu)
        out_supa = torch.erfinv(in_supa)
        in_cpu.erfinv_()
        in_supa.erfinv_()

        assert_allclose(in_cpu, in_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(out_cpu, out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)
        assert_allclose(in_supa.cpu(), out_supa, rtol=1e-5, atol=5e-5, equal_nan=True)


params = [
    [0, 1, 2, 5, 32, 50, 100, 1024],
]


class TestKaiser:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.parametrize("sizes", params)
    def test_signal_window_functions(self, sizes):
        import random

        def _test_signal_window_functions(name, dtype, device, sizes, **kwargs):
            torch_method = getattr(torch, name + "_window")
            for size in sizes:
                for periodic in [True, False]:
                    res = torch_method(
                        size, periodic=periodic, **kwargs, device=device, dtype=dtype
                    )
                    # NB: scipy always returns a float64 result
                    ref = torch.from_numpy(
                        signal.get_window(
                            (name, *(kwargs.values())), size, fftbins=periodic
                        )
                    )
                    assert_allclose(ref, res, atol=5e-5, rtol=1e-5)

        _test_signal_window_functions(
            "kaiser",
            torch.float,
            torch.device("supa"),
            sizes,
            beta=random.random() * 30,
        )
