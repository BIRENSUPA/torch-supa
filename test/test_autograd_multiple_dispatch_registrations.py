# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch

import torch_supa  # noqa: F401  enables transfer_to_supa patching
import torch_supa.contrib.transfer_to_supa  # noqa: F401


@pytest.mark.regression
def test_autograd_multiple_dispatch_registrations():
    """Regression test for AutogradCUDA-to-AutogradPrivateUse1 dispatch.

    Migrated from pytorch test_autograd.py::TestAutogradMultipleDispatch::
    test_autograd_multiple_dispatch_registrations (cuda device path).

    Fix: generated AutogradCUDA-to-AutogradPrivateUse1 registrations in the
    codegen so operators with AutogradCUDA derivative formulas (e.g.
    _test_autograd_multiple_dispatch.fullcoverage) dispatch to the SUPA
    AutogradPrivateUse1 backend instead of the bogus Default grad+1 fallback.
    On the cuda/supa device the gradient should be grad * 2.
    """
    device = "supa"

    t = torch.randn(3, 3, device=device, requires_grad=True)
    out = torch._test_autograd_multiple_dispatch(t)
    grad = torch.randn(3, 3, device=device)
    out.backward(grad)

    # bogus gradient registered for AutogradCUDA is grad * 2; after the codegen
    # fix this path must reach the SUPA AutogradPrivateUse1 registration.
    assert torch.equal(t.grad, grad * 2)

    # test registered AutogradNestedTensor formula
    a = (
        torch.arange(6, dtype=torch.float, device=device)
        .reshape(2, 3)
        .requires_grad_(True)
    )
    b = (
        torch.arange(8, dtype=torch.float, device=device)
        .reshape(2, 4)
        .requires_grad_(True)
    )
    nt = torch.nested.as_nested_tensor([a, b], dtype=torch.float, device=device)

    nt_out = torch._test_autograd_multiple_dispatch(nt)
    c = torch.randn(2, 3, device=device)
    d = torch.randn(2, 4, device=device)
    nt_grad = torch.nested.nested_tensor([c, d], dtype=torch.float, device=device)
    nt_out.backward(nt_grad)

    # bogus gradient for AutogradNestedTensor is grad * grad
    assert torch.equal(a.grad, c * c)
    assert torch.equal(b.grad, d * d)
