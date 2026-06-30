# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import os
import collections
import torch
from torch.testing._internal.common_utils import TestCase


class SupaAutocastTestLists:

    def __init__(self, dev):
        super().__init__()
        n = 8
        # Utility arguments, created as one-element tuples
        pointwise0_bf16 = (torch.randn(n, dtype=torch.bfloat16, device=dev),)
        pointwise1_bf16 = (torch.randn(n, dtype=torch.bfloat16, device=dev),)
        pointwise2_bf16 = (torch.randn(n, dtype=torch.bfloat16, device=dev),)
        mat0_bf16 = (torch.randn((n, n), dtype=torch.bfloat16, device=dev),)
        mat1_bf16 = (torch.randn((n, n), dtype=torch.bfloat16, device=dev),)
        mat2_bf16 = (torch.randn((n, n), dtype=torch.bfloat16, device=dev),)

        dimsets = ((n, n, n), (n, n, n, n), (n, n, n, n, n))
        conv_args_fp32 = [(torch.randn(dimset, dtype=torch.float32, device=dev),
                           torch.randn(dimset, dtype=torch.float32, device=dev))
                          for dimset in dimsets]
        bias_fp32 = (torch.randn((n,), dtype=torch.float32, device=dev),)
        element0_fp32 = (torch.randn(1, dtype=torch.float32, device=dev),)
        pointwise0_fp32 = (torch.randn(n, dtype=torch.float32, device=dev),)
        pointwise1_fp32 = (torch.randn(n, dtype=torch.float32, device=dev),)
        mat0_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
        mat1_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
        mat2_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
        mat3_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)

        # The remaining lists organize ops that autocast treats explicitly.
        self.torch_bf16 = [
            ("prelu", pointwise0_fp32 + element0_fp32),
            ("addmm", mat1_fp32 + mat2_fp32 + mat3_fp32),
            ("addmv", pointwise0_fp32 + mat2_fp32 + pointwise1_fp32),
            ("addr", mat0_fp32 + pointwise0_fp32 + pointwise1_fp32),
        ]
        self.torch_fp32 = [
            ("acos", (pointwise0_bf16[0].clamp(-.9, 0.9),)),
            ("asin", (pointwise0_bf16[0].clamp(-.9, 0.9),)),
            ("cosh", pointwise0_bf16),
            ("erfinv", (pointwise0_bf16[0].clamp(-.9, .9),)),
            ("exp", pointwise0_bf16),
            ("expm1", pointwise0_bf16),
            ("log", (pointwise0_bf16[0].clamp(0.1, 100.0),)),
            ("log10", (pointwise0_bf16[0].clamp(0.1, 100.0),)),
            ("log2", (pointwise0_bf16[0].clamp(0.1, 100.0),)),
            ("log1p", (pointwise0_bf16[0].clamp(-0.9, 100.0),)),
            ("reciprocal", pointwise0_bf16),
            ("rsqrt", (pointwise0_bf16[0].clamp(0.0, 100.0),)),
            ("sinh", pointwise0_bf16),
            ("tan", (pointwise0_bf16[0].clamp(-3.1 / 2, 3.1 / 2),)),
            ("pow", ((pointwise0_bf16[0] + 1.).clamp(0.0, 100.0),) + pointwise1_bf16),
            ("pow", ((pointwise0_bf16[0] + 1.).clamp(0.0, 100.0),) + (1.7,)),
            ("softmax", pointwise0_bf16 + (0,)),
            ("log_softmax", pointwise0_bf16 + (0,)),
            ("layer_norm", pointwise0_bf16 + ((pointwise0_bf16[0].numel(),),)),
            ("group_norm", mat0_bf16 + (1,)),
            ("norm", pointwise0_bf16),
            ("norm", pointwise0_bf16, {"dim": 0}),

            ("norm", pointwise0_bf16, {"p": 1}),
            ("norm", pointwise0_bf16, {"p": 1, "dim": 0}),
            ("cosine_similarity", mat0_bf16 + mat1_bf16),
            ("poisson_nll_loss", mat0_bf16 + mat1_bf16 + (True, False, 1.e-8, torch.nn._reduction.get_enum('mean'))),
            ("cosine_embedding_loss", (torch.tensor([[1, 2, 3]], device=dev, dtype=torch.bfloat16),
                                       torch.tensor([[1, 3, 4]], device=dev, dtype=torch.bfloat16),
                                       torch.tensor([1], device=dev, dtype=torch.int))),
            ("hinge_embedding_loss", mat0_bf16 + (torch.ones(n, device=dev, dtype=torch.int),)),
            ("kl_div", mat0_bf16 + (torch.rand((n, n), device=dev, dtype=torch.bfloat16),)),
            ("margin_ranking_loss", mat0_bf16 + mat1_bf16 + (torch.ones((n,), device=dev, dtype=torch.bfloat16),)),
            ("triplet_margin_loss", mat0_bf16 + mat1_bf16 + mat2_bf16),
            ("binary_cross_entropy_with_logits", mat0_bf16 + (torch.rand((n, n), device=dev, dtype=torch.bfloat16),)),
            ("cumprod", pointwise0_bf16 + (0,)),
            ("cumsum", pointwise0_bf16 + (0,)),
            ("dist", pointwise0_bf16 + pointwise1_bf16),
            # ("pdist", mat0_bf16),
            # ("cdist", mat0_bf16 + mat1_bf16),
            ("prod", pointwise0_bf16),
            ("prod", pointwise0_bf16 + (0,)),
            ("renorm", mat0_bf16 + (2, 0, 1.0)),
            ("sum", pointwise0_bf16),
            ("sum", mat0_bf16 + (1,)),
            ("logsumexp", mat0_bf16 + (1,)),
        ]
        self.nn_bf16 = [
            ("linear", mat0_fp32 + mat1_fp32 + mat2_fp32),
        ]
        self.banned = [
            ("binary_cross_entropy", (torch.rand((n, n), device=dev, dtype=torch.float32),
                                      torch.rand((n, n), device=dev, dtype=torch.float32)), torch._C._nn),
        ]


class TestSupaAutocast(TestCase):
    def setUp(self):
        super().setUp()
        self.autocast_lists = SupaAutocastTestLists(torch.device("supa:0"))

    def tearDown(self):
        del self.autocast_lists
        super().tearDown()

    def args_maybe_kwargs(self, op_with_args):
        if len(op_with_args) == 2:
            return op_with_args[0], op_with_args[1], {}
        else:
            return op_with_args[0], op_with_args[1], op_with_args[2]

    def _run_autocast_outofplace(
        self,
        op,
        args,
        run_as_type,
        device,
        out_type=None,
        module=torch,
        add_kwargs=None,
        amp_dtype=torch.bfloat16,
    ):
        # helper to cast args
        def cast(val, to_type):
            if isinstance(val, torch.Tensor):
                return val.to(to_type) if val.is_floating_point() else val
            elif isinstance(val, collections.abc.Iterable):
                return type(val)(cast(v, to_type) for v in val)
            else:
                return val

        if add_kwargs is None:
            add_kwargs = {}

        self.assertFalse(torch.supa.is_autocast_enabled())
        with torch.amp.autocast(device_type=device, dtype=amp_dtype):
            self.assertTrue(torch.supa.is_autocast_enabled())

            out_type = out_type if out_type is not None else run_as_type
            output = output_method = None

            # Try module.* variant, if requested:
            if module is not None and hasattr(module, op):
                output = getattr(module, op)(*args, **add_kwargs)
                if isinstance(output, torch.Tensor):
                    self.assertTrue(
                        out_type == output.dtype,
                        f"autocast for torch.{op} produced {output.dtype}, should produce {out_type}",
                    )
            # Try Tensor.* variant:
            if hasattr(torch.Tensor, op):
                output_method = getattr(args[0], op)(*args[1:], **add_kwargs)
                if isinstance(output_method, torch.Tensor):
                    self.assertTrue(
                        out_type == output_method.dtype,
                        f"autocast for torch.{op} produced {output_method.dtype}, should produce torch.{out_type}",
                    )

            self.assertTrue(
                (output is not None) or (output_method is not None),
                f"{op} not found as an attribute on either Tensor or the requested module {module}",
            )

            # Accounts for ops that return Tensors, iterables, and other non-Tensors.
            # For example, lstm_cell returns a tuple and equal returns bool.
            def compare(first, second):
                if isinstance(first, torch.Tensor):
                    return torch.equal(first, second)
                elif isinstance(first, collections.abc.Iterable):
                    return all(compare(f, s) for f, s in zip(first, second))
                else:
                    return first == second

            # If both torch.* and Tensor.* variants were found, check outputs are identical
            if (output is not None) and (output_method is not None):
                self.assertTrue(type(output) == type(output_method))
                comparison = compare(output, output_method)
                self.assertTrue(
                    comparison, f"torch.{op} result did not match Tensor.{op} result"
                )

            # Compare numerics to Python-side "autocasting" that (we expect) does the same thing
            # as the C++-side autocasting, and should be bitwise accurate.
            output_to_compare = output if output is not None else output_method
            with torch.amp.autocast(device_type=device, enabled=False):
                self.assertFalse(
                    torch.supa.is_autocast_enabled()
                )

                if module is not None and hasattr(module, op):
                    control = getattr(module, op)(
                        *cast(args, run_as_type), **add_kwargs
                    )
                else:
                    control = getattr(args[0].to(run_as_type), op)(
                        *cast(args[1:], run_as_type), **add_kwargs
                    )
                self.assertTrue(type(output_to_compare) == type(control))
                comparison = compare(output_to_compare, control)
                self.assertTrue(comparison, f"torch.{op} result did not match control")
            self.assertTrue(torch.supa.is_autocast_enabled())
        self.assertFalse(torch.supa.is_autocast_enabled())

    def test_autocast_torch_bf16(self):
        # with torch.backends.cudnn.flags(enabled=True, deterministic=True):
        for op_with_args in self.autocast_lists.torch_bf16:
            skip_test = False
            op, args = op_with_args[0], op_with_args[1]
            if len(op_with_args) == 3:
                skip_test = op_with_args[2]  # TEST_WITH_ROCM
            should_error_from_cudnn = "cudnn" in op and (
                "TORCH_CUDNN_V8_API_DISABLED" in os.environ
                and int(os.environ["TORCH_CUDNN_V8_API_DISABLED"])
                or torch.supa.get_device_capability() < (8, 0)
            )
            should_error_from_not_implemented = should_error_from_cudnn
            if not skip_test:
                if should_error_from_not_implemented:
                    with self.assertRaises(
                        RuntimeError,
                        msg=str(op) + " should not be supported for bfloat16!",
                    ):
                        self._run_autocast_outofplace(
                            op, args, torch.bfloat16, device="supa"
                        )
                else:
                    if True:  # torch.supa.is_bf16_supported():
                        self._run_autocast_outofplace(
                            op, args, torch.bfloat16, device="supa"
                        )
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError, "Device does not support bfloat16"
                        ):
                            self._run_autocast_outofplace(
                                op, args, torch.bfloat16, device="supa"
                            )

    def test_autocast_torch_fp32(self):
        for op_with_args in self.autocast_lists.torch_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="supa",
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.bfloat16,
            )

    def test_autocast_nn_bf16(self):
        for op, args in self.autocast_lists.nn_bf16:
            if True:  # torch.supa.is_bf16_supported():
                self._run_autocast_outofplace(
                    op, args, torch.bfloat16, device="supa", module=torch._C._nn
                )
            else:
                with self.assertRaisesRegex(
                    RuntimeError, "Device does not support bfloat16"
                ):
                    self._run_autocast_outofplace(
                        op, args, torch.bfloat16, device="supa", module=torch._C._nn
                    )

    def test_autocast_banned(self):
        with torch.autocast("supa", dtype=torch.bfloat16):
            for op, args, module in self.autocast_lists.banned:
                with self.assertRaises(RuntimeError):
                    getattr(module, op)(*args)

    def test_autocast_device_type(self):
        x = torch.zeros((5, 5), device="cuda", dtype=torch.float16)
        op = torch.nn.Linear(5, 5, bias=False, device="cuda", dtype=torch.float16)
        with torch.autocast("cuda", dtype=torch.float32):
            y = op(x)
        self.assertEqual(y.dtype, torch.float32)
