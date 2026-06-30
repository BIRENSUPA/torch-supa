# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import gc
import os
import threading

import torch
import torch_supa
from torch_supa.utils import torch_version_ge
from torch.testing._internal.common_utils import TestCase

import pytest

FLOAT_RTOL, FLOAT_ATOL = 1e-5, 5e-5

@pytest.mark.sanity
@pytest.mark.regression
class TestSupa(TestCase):
    _do_supa_memory_leak_check = True
    _do_supa_non_default_stream = True
    FIFTY_MIL_CYCLES = 50000000

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_pinned_memory_with_suparegister(self):
        try:
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
            )
            t = torch.ones(20)
            self.assertFalse(t.is_pinned())
            try:
                pinned_t = torch.ones(1 << 21).pin_memory()
                self.assertTrue(pinned_t.is_pinned())
                pinned_t = torch.ones(1 << 24).pin_memory()
                self.assertTrue(pinned_t.is_pinned())
            except RuntimeError as e:
                # Some GPUs don't support same address space on host and device side
                pass
        finally:
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_cuda_host_register:False"
            )


    def test_pinned_memory_with_suparegister_multithread(self):
        num_threads = 4
        threads = [
            threading.Thread(target=self.test_pinned_memory_with_suparegister)
            for t in range(num_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()


    def test_pinned_memory_empty_cache(self):
        try:
            for alloc_settings in (True, False):
                torch.cuda.memory._set_allocator_settings(
                    f"pinned_use_cuda_host_register:{alloc_settings}"
                )
                try:
                    t = torch.ones(1024 * 1024, pin_memory=True)
                    self.assertTrue(t.is_pinned())
                    del t
                    torch_supa._C._host_emptyCache()
                except RuntimeError as e:
                    # Some GPUs don't support same address space on host and device side
                    pass
        finally:
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_cuda_host_register:False"
            )


    def test_memory_allocation(self):
        gc.collect()
        torch.cuda.empty_cache()
        mem = None
        size = 1
        prev = 0
        try:
            prev = torch.cuda.memory_allocated()
            mem = torch.cuda.caching_allocator_alloc(size)
            self.assertGreater(torch.cuda.memory_allocated(), prev)
        finally:
            if mem is not None:
                torch.cuda.caching_allocator_delete(mem)
                self.assertEqual(torch.cuda.memory_allocated(), prev)


    def test_supa_get_device_name(self):
        # Testing the behaviour with None as an argument
        current_device = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device)
        device_name_None = torch.cuda.get_device_name(None)
        self.assertEqual(current_device_name, device_name_None)

        # Testing the behaviour for No argument
        device_name_no_argument = torch.cuda.get_device_name()
        self.assertEqual(current_device_name, device_name_no_argument)


    def test_supa_get_device_properties_uuid(self):
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        uuid = props.uuid

        self.assertEqual(type(uuid).__name__, "_SUuuid")
        self.assertTrue(hasattr(uuid, "bytes"))
        self.assertEqual(len(uuid.bytes), 16)
        for byte in uuid.bytes:
            self.assertIsInstance(byte, int)
            self.assertGreaterEqual(byte, 0)
            self.assertLessEqual(byte, 255)

        uuid_str = str(uuid)
        self.assertEqual(len(uuid_str), 36)
        self.assertEqual(uuid_str[8], "-")
        self.assertEqual(uuid_str[13], "-")
        self.assertEqual(uuid_str[18], "-")
        self.assertEqual(uuid_str[23], "-")


    def test_out_of_memory(self):
        tensor = torch.zeros(1024, device="cuda")
        oom_regex = (f"SUPA out of memory. Tried to allocate 800000000.00 GiB. GPU ")  # noqa: F541
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            torch.empty(1024 * 1024 * 1024 * 800000000, dtype=torch.int8, device="cuda")

        with self.assertRaisesRegex(
            RuntimeError, "Tried to allocate more than 1EB memory"
        ):
            torch.empty(
                1024 * 1024 * 1024 * 8000000000, dtype=torch.int8, device="cuda"
            )

        # ensure out of memory error doesn't disturb subsequent kernel
        tensor.fill_(1)
        self.assertTrue((tensor == 1).all())


    def test_out_of_memory_retry(self):
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        oom_regex = (
            "Tried to allocate"
        )
        size = int(total_memory * 0.5)
        a = torch.empty(size, dtype=torch.int8, device="cuda")
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            b = torch.empty(size, dtype=torch.int8, device="cuda")
        del a
        b = torch.empty(size, dtype=torch.int8, device="cuda")
        del b
        # We used a lot of memory here, clean up so we don't affect other tests too much
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


    def test_set_per_process_memory_fraction(self):
        try:
            # test invalid fraction value.
            with self.assertRaisesRegex(TypeError, "Invalid type"):
                torch.cuda.set_per_process_memory_fraction(1)
            with self.assertRaisesRegex(ValueError, "Invalid fraction value"):
                torch.cuda.set_per_process_memory_fraction(-0.1)
            with self.assertRaisesRegex(ValueError, "Invalid fraction value"):
                torch.cuda.set_per_process_memory_fraction(2.0)

            tensor = torch.zeros(1024, device="cuda")
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(0.5, 0)

            # test 0.499 allocation is ok.
            application = int(total_memory * 0.499) - torch.cuda.max_memory_reserved()
            tmp_tensor = torch.empty(application, dtype=torch.int8, device="cuda")
            del tmp_tensor
            torch.cuda.empty_cache()

            application = int(total_memory * 0.5)
            # it will get OOM when try to allocate more than half memory.
            oom_regex = (
                "out of memory"
            )
            with self.assertRaisesRegex(RuntimeError, oom_regex):
                torch.empty(application, dtype=torch.int8, device="cuda")

            # ensure out of memory error doesn't disturb subsequent kernel
            tensor.fill_(1)
            self.assertTrue((tensor == 1).all())
        finally:
            torch.cuda.set_per_process_memory_fraction(1.0, 0)


    def test_get_per_process_memory_fraction(self):
        # get the initial memory fraction
        init_fraction = torch.cuda.get_per_process_memory_fraction()

        # set and get the limiting cases
        torch.cuda.set_per_process_memory_fraction(1.0)
        self.assertEqual(torch.cuda.get_per_process_memory_fraction(), 1.0)
        torch.cuda.set_per_process_memory_fraction(0.0)
        self.assertEqual(torch.cuda.get_per_process_memory_fraction(), 0.0)

        # test a few random cases
        for val in torch.rand(3):
            torch.cuda.set_per_process_memory_fraction(float(val))
            self.assertEqual(torch.cuda.get_per_process_memory_fraction(), float(val))

        # restore the initial memory fraction
        torch.cuda.set_per_process_memory_fraction(init_fraction)


    def test_get_device_index(self):
        from torch.cuda import _get_device_index

        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            _get_device_index("supa0", optional=True)

        with self.assertRaisesRegex(ValueError, "Expected a cuda device"):
            cpu_device = torch.device("cpu")
            _get_device_index(cpu_device, optional=True)


    def test_streams(self):
        default_stream = torch.cuda.current_stream()
        user_stream = torch.cuda.Stream()
        self.assertEqual(torch.cuda.current_stream(), default_stream)
        self.assertNotEqual(default_stream, user_stream)
        self.assertEqual(default_stream.cuda_stream, 0)
        self.assertNotEqual(user_stream.cuda_stream, 0)
        with torch.cuda.stream(user_stream):
            self.assertEqual(torch.cuda.current_stream(), user_stream)
        self.assertTrue(user_stream.query())
        tensor1 = torch.ByteTensor(5).pin_memory()
        tensor2 = tensor1.cuda(non_blocking=True) + 1
        default_stream.synchronize()
        self.assertTrue(default_stream.query())


    def test_generic_stream_event(self):
        stream = torch.Stream("cuda")
        self.assertEqual(stream.device_index, torch.cuda.current_device())
        cuda_stream = torch.cuda.Stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=stream.device_type,
        )
        self.assertIsInstance(cuda_stream, torch.Stream)
        self.assertTrue(issubclass(type(cuda_stream), torch.Stream))
        self.assertTrue(torch.Stream in type(cuda_stream).mro())
        self.assertEqual(stream.stream_id, cuda_stream.stream_id)
        self.assertNotEqual(stream.stream_id, torch.cuda.current_stream().stream_id)

        event1 = torch.Event("cuda", enable_timing=True)
        event2 = torch.Event("cuda", enable_timing=True)
        self.assertEqual(event1.event_id, 0)
        a = torch.randn(1000)
        b = torch.randn(1000)
        with torch.cuda.stream(cuda_stream):
            a_cuda = a.to("cuda", non_blocking=True)
            b_cuda = b.to("cuda", non_blocking=True)
            self.assertEqual(stream.stream_id, torch.cuda.current_stream().stream_id)
        event1.record(stream)
        event1.synchronize()
        self.assertTrue(event1.query())
        c_cuda = a_cuda + b_cuda
        event2.record()
        event2.synchronize()
        self.assertTrue(event2.query())
        self.assertNotEqual(event1.event_id, event2.event_id)
        self.assertEqual(c_cuda.cpu(), a + b)
        self.assertTrue(event1.elapsed_time(event2) > 0)
        cuda_event = torch.cuda.Event()
        self.assertIsInstance(cuda_event, torch.Event)
        self.assertTrue(issubclass(type(cuda_event), torch.Event))
        self.assertTrue(torch.Event in type(cuda_event).mro())


    def test_stream_event_repr(self):
        s = torch.cuda.current_stream()
        self.assertTrue("torch_supa.supa.Stream" in s.__repr__())
        e = torch.cuda.Event()
        self.assertTrue("torch_supa.supa.Event" in e.__repr__())
        s.record_event(e)
        self.assertTrue("torch_supa.supa.Event" in e.__repr__())


    def test_cublas_allow_tf32_get_set(self):
        skip_tf32_cublas = "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in os.environ and int(
            os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]
        )
        if skip_tf32_cublas:
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            return

        orig = torch.backends.cuda.matmul.allow_tf32
        self.assertEqual(torch._C._get_cublas_allow_tf32(), orig)
        torch.backends.cuda.matmul.allow_tf32 = not orig
        self.assertEqual(torch._C._get_cublas_allow_tf32(), not orig)
        torch.backends.cuda.matmul.allow_tf32 = orig


    def test_float32_matmul_precision_get_set(self):
        orig = torch.get_float32_matmul_precision()
        skip_tf32_cublas = "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in os.environ and int(
            os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]
        )
        # this is really just checking that the environment variable is respected during testing
        # and not overwritten by another function that doesn't revert it to the intitial value
        if not skip_tf32_cublas:
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        else:
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        for p in ("medium", "high"):
            torch.set_float32_matmul_precision(p)
            self.assertEqual(torch.get_float32_matmul_precision(), p)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        torch.set_float32_matmul_precision("highest")
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        torch.set_float32_matmul_precision(orig)


    def test_cublas_allow_fp16_reduced_precision_reduction_get_set(self):
        if torch_version_ge(2, 10, 0):
            orig = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            orig_splitk = (
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction_split_k
            )
            self.assertEqual(
                torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
                (orig, orig_splitk),
            )
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = not orig
            self.assertEqual(
                torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
                (not orig, True),
            )
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
                False,
                False,
            )
            self.assertEqual(
                torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
                (False, False),
            )
        else:
            orig = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            self.assertEqual(
                torch._C._get_cublas_allow_fp16_reduced_precision_reduction(), orig
            )
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = not orig
            self.assertEqual(
                torch._C._get_cublas_allow_fp16_reduced_precision_reduction(), not orig
            )

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = orig


    def test_cublas_allow_bf16_reduced_precision_reduction_get_set(self):
        if torch_version_ge(2, 10, 0):
            orig = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            orig_splitk = (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k
            )
            self.assertEqual(
                torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
                (orig, orig_splitk),
            )
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not orig
            self.assertEqual(
                torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
                (not orig, True),
            )
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                False,
                False,
            )
            self.assertEqual(
                torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
                (False, False),
            )
        else:
            orig = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            self.assertEqual(
                torch._C._get_cublas_allow_bf16_reduced_precision_reduction(), orig
            )
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not orig
            self.assertEqual(
                torch._C._get_cublas_allow_bf16_reduced_precision_reduction(), not orig
            )

        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = orig


    def test_cudnn_allow_tf32_get_set(self):
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=False
        ):
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=True
        ):
            self.assertTrue(torch.backends.cudnn.allow_tf32)


    def test_pinned_memory_copy(self):
        expected = [6023, 15588, 15588]
        a_dev = torch.tensor(expected, dtype=torch.int64, pin_memory=True).to(
            "supa", non_blocking=True
        )
        torch.tensor([3], dtype=torch.int64, pin_memory=True).to(
            "supa", non_blocking=True
        )

        torch.get_device_module("supa").synchronize()
        assert a_dev.cpu().tolist() == expected

    @pytest.mark.skipif(
        not torch_version_ge(2, 8, 0), reason="MemPool/use_mem_pool not available for torch 2.6"
    )
    def test_empty_cache_during_use_mem_pool(self):
        """empty_cache() inside use_mem_pool() with allocations should not crash."""
        x = torch.randn(1024, 1024, device="cuda")
        del x

        pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool):
            y = torch.randn(512, 512, device="cuda")
            torch.cuda.empty_cache()
            del y
