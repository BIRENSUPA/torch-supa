# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.


# noqa
import pytest
import torch
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")


br200_params = [
    pytest.param(
        (2, 2, 6),
        marks=[
            pytest.mark.sanity,
            pytest.mark.gcuSmoke,
            pytest.mark.regression,
            pytest.mark.gcuSanity,
            pytest.mark.gcuStress,
        ],
    ),
    pytest.param((4, 128, 256), marks=[pytest.mark.gcuSanity, pytest.mark.gcuStress]),
    pytest.param((4, 511, 257), marks=[pytest.mark.gcuStress]),
    pytest.param((15, 127, 513), marks=[pytest.mark.gcuStress]),
    pytest.param((15, 400, 512), marks=[pytest.mark.gcuStress]),
]

dtypes = [torch.float32, torch.bfloat16, torch.float16]
RTOL = {torch.float32: 1.3e-6, torch.bfloat16: 1.6e-2, torch.float16: 1e-3}
ATOL = {torch.float32: 1e-5, torch.bfloat16: 1e-5, torch.float16: 1e-5}


class TestLayerNormTrainingDyn:
    @pytest.mark.parametrize("shape", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_layernorm_training_dyn(self, shape, dtype):

        N, H, W = shape[0], shape[1], shape[2]
        x_cpu = torch.randn(N, H, W)
        ln = torch.nn.LayerNorm(W)
        ln.train()
        y_cpu = ln(x_cpu)

        x_supa = x_cpu.to(supa_device)
        ln_supa = ln.to(supa_device)
        y_supa = ln_supa(x_supa)

        assert_allclose(y_cpu, y_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])

    @pytest.mark.parametrize("shape", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_layernorm_eval_dyn(self, shape, dtype):

        N, H, W = shape[0], shape[1], shape[2]
        x_cpu = torch.randn(N, H, W)
        ln = torch.nn.LayerNorm(W)
        ln.eval()
        y_cpu = ln(x_cpu)

        x_supa = x_cpu.to(supa_device)
        ln_supa = ln.to(supa_device)
        y_supa = ln_supa(x_supa)

        assert_allclose(y_cpu, y_supa.cpu(), atol=ATOL[dtype], rtol=RTOL[dtype])

    @pytest.mark.parametrize("shape", br200_params)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_layernorm_backward_dyn(self, shape, dtype):
        N, H, W = shape[0], shape[1], shape[2]
        x_cpu, x_supa = create_random_tensor([N, H, W], dtype=dtype, requires_grad=True)
        grad_cpu, grad_supa = create_random_tensor([N, H, W], dtype=dtype)
        weight_cpu, weight_supa = create_random_tensor(
            [W], dtype=dtype, requires_grad=True
        )
        bias_cpu, bias_supa = create_random_tensor([W], dtype=dtype, requires_grad=True)

        ln_cpu = torch.nn.LayerNorm(W).to(dtype)

        y_cpu = ln_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        x_cpu_grad = x_cpu.grad
        weight_cpu_grad = ln_cpu.weight.grad
        bias_cpu_grad = ln_cpu.bias.grad

        ln_supa = torch.nn.LayerNorm(W).to(dtype).to(supa_device)
        y_supa = ln_supa(x_supa)
        y_supa.backward(grad_supa)
        x_supa_grad = x_supa.grad
        weight_supa_grad = ln_supa.weight.grad
        bias_supa_grad = ln_supa.bias.grad

        if dtype == torch.bfloat16 or dtype == torch.float16:
            atol = 1e-1
            rtol = 0
            atol_w = 5e-1
            rtol_w = 1e-3
        else:
            atol = ATOL[dtype]
            rtol = RTOL[dtype]
            atol_w = 1e-4
            rtol_w = 1e-5

        assert_allclose(y_cpu, y_supa.cpu(), atol=atol, rtol=rtol)
        assert_allclose(x_cpu, x_supa.cpu(), atol=atol, rtol=rtol)
        assert_allclose(x_cpu_grad, x_supa_grad.cpu(), atol=atol, rtol=rtol)
        if dtype != torch.bfloat16:
            assert_allclose(
                weight_cpu_grad, weight_supa_grad.cpu(), atol=atol_w, rtol=rtol_w
            )
            assert_allclose(
                bias_cpu_grad, bias_supa_grad.cpu(), atol=atol_w, rtol=rtol_w
            )
