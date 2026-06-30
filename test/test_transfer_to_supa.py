# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import os

import pytest

import torch
import torch.distributed as c10d
from torch.testing._internal.common_utils import TestCase
from torch_supa.contrib import transfer_to_supa  # noqa


@pytest.mark.sanity
@pytest.mark.regression
class TestTransferToSupa(TestCase):
    def test_amp_function(self):
        self.assertEqual(torch.cuda.amp.autocast_mode, torch.supa.amp.autocast_mode)
        self.assertEqual(torch.cuda.amp.common, torch.supa.amp.common)
        self.assertEqual(torch.cuda.amp.grad_scaler, torch.supa.amp.grad_scaler)

    def test_amp_enabled_api(self):
        device = "cuda"
        self.assertFalse(torch.is_autocast_enabled(device))
        with torch.amp.autocast(device_type=device, dtype=torch.float):
            self.assertTrue(torch.is_autocast_enabled())


    def test_wrap_device(self):
        device = torch.device(f"cuda:{0}")
        torch.cuda.set_device(device)
        a = torch.randint(1, 5, (2, 3), device=device)
        self.assertEqual(a.device.type, "supa")

        self.assertEqual(torch.randint(1, 5, (2, 3), device="cuda").device.type, "supa")
        self.assertEqual(torch.Generator("cuda").device.type, "supa")
        b = torch.randint(1, 5, (2, 3))
        self.assertEqual(b.new_empty((2, 3), device="cuda").device.type, "supa")
        self.assertEqual(torch.amp.autocast_mode.autocast("cuda").device, "supa")

    @pytest.mark.skipif(torch.supa.device_count() < 2, reason="requires two device")
    def test_wrap_device_type(self):
        device = torch.device(f"cuda:{1}")
        torch.cuda.set_device(device)
        a = torch.randint(1, 5, (2, 3), device=device)
        with transfer_to_supa.device_type_context():
            self.assertTrue("supa" in a.device.__repr__())
            self.assertEqual(a.device.type, "cuda")
            self.assertEqual(device.type, "cuda")
            self.assertTrue(a.is_cuda)

        self.assertTrue(a.is_cuda)

    def test_wrap_cuda_version(self):
        version = torch.version.cuda
        self.assertIsNotNone(version)

    def test_wrap_device_int_type(self):
        a = torch.rand(1)
        device_id = torch.cuda.current_device()
        b = a.to(device=device_id)
        c = a.to(device_id)
        d = torch.tensor(1, device=device_id)

    def test_wrap_isinstance(self):
        # check builtins isinstance grammar
        self.assertTrue(isinstance(1, int))
        self.assertTrue(isinstance(1, (int, str)))
        self.assertFalse(isinstance(1, str))
        with self.assertRaises(TypeError):
            isinstance(1, [str, int])

        # check torch.device
        self.assertFalse(isinstance(1, torch.device))

        # check torch.cuda.device
        device = -1
        torch.cuda.device(device)
        self.assertTrue(
            isinstance(
                torch.tensor([1.0], dtype=torch.bfloat16).cuda(),
                torch.cuda.BFloat16Tensor,
            )
        )
        self.assertTrue(
            isinstance(
                torch.tensor([1.0], dtype=torch.float).cuda(), torch.cuda.FloatTensor
            )
        )

    def test_device_context(self):
        with torch.device("cuda"):
            a = torch.ones(3, 3)
        self.assertEqual(a.device.type, "supa")

        with torch.device(type="cuda"):
            b = torch.ones(3, 3)
            with torch.device(type="cpu"):
                d = torch.ones(3, 3)
                e = torch.ones(3, 3, device="cuda")

        self.assertEqual(b.device.type, "supa")
        self.assertEqual(d.device.type, "cpu")
        self.assertEqual(e.device.type, "supa")

    def test_device_inside_inductor(self):
        def my_custom_backend(gm, example_inputs):
            # gm.graph.print_tabular()
            return gm.forward

        @torch.compile(backend=my_custom_backend)
        def fn2():
            input_supa = torch.empty((2, 2), device=torch.device("cuda"))
            return input_supa

        output = fn2()
        self.assertEqual(output.device.type, "supa")

    def test_bccl(self):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "32768"
        c10d.init_process_group("nccl", world_size=1, rank=0)
        pg = c10d.distributed_c10d._get_default_group()
        self.assertEqual(pg._get_backend_name(), "custom")
        pg2 = torch.distributed.new_group(backend="nccl")
        self.assertEqual(pg2._get_backend_name(), "custom")
        c10d.destroy_process_group()

    def test_brtx(self):
        self.assertTrue(hasattr(torch.supa, "brtx"))
        self.assertEqual(torch.cuda.nvtx.range_push, torch.supa.brtx.range_push)
        self.assertEqual(torch.cuda.nvtx.mark, torch.supa.brtx.mark)
        self.assertEqual(torch.cuda.nvtx.range_pop, torch.supa.brtx.range_pop)
        self.assertEqual(torch.cuda.nvtx.range_start, torch.supa.brtx.range_start)
        self.assertEqual(torch.cuda.nvtx.range_end, torch.supa.brtx.range_end)

        # Just making sure we can see the symbols
        torch.cuda.nvtx.range_push("foo")
        torch.cuda.nvtx.mark("bar")
        torch.cuda.nvtx.range_pop()
        range_handle = torch.cuda.nvtx.range_start("range_start")
        torch.cuda.nvtx.range_end(range_handle)

    def test_default_generators(self):
        torch.cuda.init()
        self.assertEqual(
            len(torch.cuda.default_generators), len(torch.supa.default_generators)
        )

    def test_version_info(self):
        compute_capability = torch.cuda.get_device_capability()
        self.assertTrue(compute_capability >= (9, 0))
        self.assertTrue(torch._C._cuda_getCompiledVersion() >= 12090)

    def test_conv_benchmark_empty_cache(self):
        orig_enable_conv_benchmark = torch._C._cuda_get_conv_benchmark_empty_cache()
        self.assertTrue(orig_enable_conv_benchmark)
        torch._C._cudnn_set_conv_benchmark_empty_cache(False)
        self.assertFalse(torch._C._cuda_get_conv_benchmark_empty_cache())
        torch._C._cudnn_set_conv_benchmark_empty_cache(orig_enable_conv_benchmark)

    def test_trans_sleep(self):
        self.assertEqual(torch.cuda._sleep, torch.supa._sleep)
