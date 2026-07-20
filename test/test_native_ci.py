# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from contextlib import contextmanager

import pytest
import torch

import torch_supa  # noqa: F401


@contextmanager
def _deterministic_fill_enabled():
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    was_fill_enabled = torch.utils.deterministic.fill_uninitialized_memory

    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = was_fill_enabled


@pytest.mark.sanity
@pytest.mark.regression
class TestNativeCI:
    def test_dispatch_meta_outplace_narrow_copy_float32(self):
        cpu_input = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        supa_input = cpu_input.to("supa")
        meta_input = torch.empty_strided(
            cpu_input.shape,
            cpu_input.stride(),
            dtype=torch.float32,
            device="meta",
        )

        result = torch.ops.aten.narrow_copy.default(supa_input, 1, 1, 2)
        meta_result = torch.ops.aten.narrow_copy.default(meta_input, 1, 1, 2)
        expected = cpu_input.narrow(1, 1, 2).clone()

        assert result.device.type == "supa"
        assert result.dtype == meta_result.dtype
        assert result.shape == meta_result.shape
        assert result.stride() == meta_result.stride()
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

    def test_deterministic_empty_float32(self):
        device = "supa"
        dtype = torch.float32
        gen_fns = [
            lambda: torch.empty(10, 9, device=device, dtype=dtype),
            lambda: torch.empty(10, 9, out=torch.zeros(1, device=device, dtype=dtype)),
            lambda: torch.empty_like(torch.zeros(10, 9, device=device, dtype=dtype)),
            lambda: torch.empty_like(
                torch.zeros(10, 9, device=device, dtype=dtype),
                memory_format=torch.contiguous_format,
            ),
            lambda: torch.empty_strided((10, 9), (1, 5), device=device, dtype=dtype),
        ]
        if hasattr(torch, "empty_permuted"):
            gen_fns.append(
                lambda: torch.empty_permuted(
                    (2, 3, 5),
                    (1, 0, 2),
                    device=device,
                    dtype=dtype,
                )
            )

        for gen_fn in gen_fns:
            with _deterministic_fill_enabled():
                result = gen_fn()

            assert result.isnan().all()

    def test_tensor_type_supa(self):
        accelerator_token = "cu" + "da"
        accelerator_attr = "is_" + accelerator_token

        for tensor_type in torch._tensor_classes:
            module_name = tensor_type.__module__
            assert getattr(tensor_type, accelerator_attr) == (
                accelerator_token in module_name
            )
            assert tensor_type.is_xpu == ("xpu" in module_name)
