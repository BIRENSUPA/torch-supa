# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import os
import torch
import pytest
from torch._inductor.test_case import TestCase
from torch._inductor.utils import (
    run_and_get_code
)
from torch._dynamo.utils import ifdynstaticdefault
from torch_supa.utils import torch_version_ge


def _run_and_assert_no_indirect_indexing(
    test_case, func, *args, has_wrapping=None, has_assert=False, **kwargs
):
    result, source_codes = run_and_get_code(func, *args, **kwargs)

    for code in source_codes:
        for line in code.split("\n"):
            stmt = None
            # Find indexing expressions
            if ".load(" in line:
                stmt = line.split(".load")[-1]
            elif "tl.store" in line:
                stmt = line.split(".store")[-1]
                stmt = ",".join(stmt.split(",")[:-2])  # Remove store value and mask
            elif ".store" in line:
                stmt = line.split(".store")[-1]
            elif "[" in line:
                stmt = line.split("[")[-1].split("]")[0]
            if "tl.make_block_ptr(" in line:
                continue

            if stmt is None:
                continue

            # indirect indexing involves a `tmp` variable
            test_case.assertTrue(
                "tmp" not in stmt,
                msg=f"Found indirect indexing in statement '{stmt}' from code:\n{code}",
            )
        if has_wrapping is not None:
            test_case.assertTrue(
                ("where" in code or ") ? (" in code) is has_wrapping,
                msg=f"Wanted {has_wrapping=} but got\n{code}",
            )
    test_case.assertTrue(
        any(
            ("device_assert" in code or "TORCH_CHECK" in code) is has_assert
            for code in source_codes
        )
    )
    return result


@pytest.mark.sanity
@pytest.mark.regression
class InductorTest(TestCase):
    def test_index_propagation(self):
        def copy(x):
            i = torch.arange(x.size(0), device=x.device)
            return x[i]

        x = torch.randn(8, device="cuda")
        copy_opt = torch.compile(copy, backend="inductor")

        expect = copy(x)
        actual = _run_and_assert_no_indirect_indexing(self, copy_opt, x)
        self.assertEqual(expect, actual)


    def test_index_propagation_flip(self):
        def flip(x):
            i = torch.arange(x.size(0) - 1, -1, -1, device=x.device)
            return x[i]

        x = torch.randn(8, device="cuda")
        flip_opt = torch.compile(flip, backend="inductor")

        expect = flip(x)
        actual = _run_and_assert_no_indirect_indexing(self, flip_opt, x)
        self.assertEqual(expect, actual)


    @pytest.mark.skip(reason="cmodel not support")
    def test_index_propagation_floordiv(self):
        def repeat_interleave(x, n):
            # e.g. x=[1, 2, 3], n=2 => returns [1, 1, 2, 2, 3, 3]
            i = torch.arange(x.shape[0] * n, device=x.device)
            return x[i // n]

        x = torch.randn(8, 16, device="cuda")
        repeat_interleave_opt = torch.compile(repeat_interleave, backend="inductor")
        # With static shapes we can prove the bound, our dynamic shapes reasoning is not good enough
        has_assert = ifdynstaticdefault(False, True)
        # this should be collapsed to direct indexing
        actual = _run_and_assert_no_indirect_indexing(
            self, repeat_interleave_opt, x, 3, has_assert=has_assert
        )
        expect = torch.repeat_interleave(x, 3, dim=0)
        self.assertEqual(expect, actual)
        self.assertEqual(actual, repeat_interleave(x, 3))


    @pytest.mark.skip(reason="cmodel not support")
    def test_index_propagation_remainder(self):
        def repeat(x, n):
            # e.g. x=[1, 2, 3], n=2 => returns [1, 2, 3, 1, 2, 3]
            i = torch.arange(x.shape[0] * n, device=x.device)
            return x[i % x.shape[0]]

        x = torch.randn(8, 16, device="cuda")
        repeat_opt = torch.compile(repeat, backend="inductor")

        # With static shapes we can prove the bound, our dynamic shapes reasoning is not good enough
        has_assert = ifdynstaticdefault(False, True)
        # this should be collapsed to direct indexing
        actual = _run_and_assert_no_indirect_indexing(
            self, repeat_opt, x, 3, has_wrapping=False, has_assert=has_assert
        )
        expect = x.repeat(3, 1)
        self.assertEqual(expect, actual)
        self.assertEqual(actual, repeat(x, 3))


    def test_index_propagation_abs(self):
        def reflection_pad_left(x, n):
            # e.g. x=[1, 2, 3], n=2 => returns [3, 2, 1, 2, 3]
            i = torch.arange(x.shape[0] + n, device=x.device)
            return x[(i - n).abs()]

        x = torch.randn(8, device="cuda")
        opt_fn = torch.compile(reflection_pad_left, backend="inductor")

        # With static shapes we can prove the bound, our dynamic shapes reasoning is not good enough
        has_assert = ifdynstaticdefault(False, True)
        # this should be collapsed to direct indexing
        actual = _run_and_assert_no_indirect_indexing(
            self, opt_fn, x, 3, has_wrapping=False, has_assert=has_assert
        )
        expect = reflection_pad_left(x, 3)
        self.assertEqual(expect, actual)


    def test_computed_buffer_inlining(self):
        def flip(x):
            idx = torch.arange(x.size(0) - 1, -1, -1, device=x.device)
            return x[idx], idx

        flip_opt = torch.compile(flip, backend="inductor")
        x = torch.randn(8, device="cuda")

        expect = flip(x)
        actual = _run_and_assert_no_indirect_indexing(self, flip_opt, x)
        self.assertEqual(expect, actual)


    def test_compile_matmul(self):
        @torch.compile
        def matmul(x, y):
            return torch.mm(x, y)

        x = torch.randn(8, 16, device="cuda")
        y = torch.randn(16, 8, device="cuda")

        expect = matmul(x, y)
        actual = torch.mm(x, y)
        self.assertEqual(expect, actual)

    def test_compile_matmul_with_setting(self):
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "TRITON"
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "1"

        @torch.compile
        def matmul(x, y):
            return torch.mm(x, y)

        x = torch.randn(8, 16, device="cuda")
        y = torch.randn(16, 8, device="cuda")

        expect = matmul(x, y)
        actual = torch.mm(x, y)
        self.assertEqual(expect, actual)
        os.environ.pop("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS")
        os.environ.pop("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM")


    @pytest.mark.skipif(not torch_version_ge(2, 10, 0), reason="out_dtype matmul APIs require torch >= 2.10")
    def test_compile_matmul_out_dtype(self):
        @torch.compile
        def matmul(x, y):
            return torch.mm(x, y, out_dtype=torch.float32)

        x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        y = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)

        expect = torch.mm(x.float(), y.float())
        actual = matmul(x, y)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(expect, actual)


    def test_compile_with_device_type_wrap_enabled(self):
        from torch_supa.contrib.transfer_to_supa import device_type_context

        @torch.compile
        def creator(size, stride):
            a = torch.empty_strided(size, stride, device="cuda")
            b = torch.empty(size, device="cuda")
            return a, b

        with device_type_context():
            a, b = creator((8, 8), (8, 1))
            self.assertEqual(a.device.type, "cuda")
            self.assertEqual(b.device.type, "cuda")
