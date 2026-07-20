# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import ctypes
import gc
import json
import os
import pickle
import threading

import torch
import torch_supa
from torch_supa.utils import torch_version_ge
from torch.testing._internal.common_utils import TestCase, get_cycles_per_ms
from torch_supa.testing.common_utils import freeze_rng_state

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


    def test_supa_get_device_capability(self):
        # Testing the behaviour with None as an argument
        current_device = torch.cuda.current_device()
        current_device_capability = torch.cuda.get_device_capability(current_device)
        device_capability_None = torch.cuda.get_device_capability(None)
        self.assertEqual(current_device_capability, device_capability_None)

        # Testing the behaviour for No argument
        device_capability_no_argument = torch.cuda.get_device_capability()
        self.assertEqual(current_device_capability, device_capability_no_argument)


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


    def test_copy_non_blocking(self):
        def _test_copy_non_blocking(a, b):
            event = torch.cuda.Event()
            a.copy_(b, non_blocking=True)
            event.record()
            event.synchronize()
            self.assertEqual(a, b)

        # 1kb copies
        x = torch.ones(1000, dtype=torch.uint8).cuda()
        y = torch.zeros(1000, dtype=torch.uint8).pin_memory()
        _test_copy_non_blocking(x, y)

        x = torch.zeros(1000, dtype=torch.uint8).pin_memory()
        y = torch.ones(1000, dtype=torch.uint8).cuda()
        _test_copy_non_blocking(x, y)

        # Test the case where the pinned data_ptr is not equal to the storage data_ptr.
        x_base = torch.zeros(1000, dtype=torch.uint8).pin_memory()
        x = x_base[1:]
        self.assertTrue(x.is_pinned())
        self.assertTrue(x_base.is_pinned())
        self.assertNotEqual(x_base.data_ptr(), x.data_ptr())
        self.assertEqual(x_base.storage().data_ptr(), x.storage().data_ptr())
        y = torch.ones(1000 - 1, dtype=torch.uint8).cuda()
        _test_copy_non_blocking(x, y)


    def test_copy_non_blocking_type_conversion(self):
        a = torch.ones(1, device="cuda")
        b = torch.zeros(1, device="cpu", pin_memory=True)
        c = torch.empty(1, device="cuda", dtype=torch.long)
        torch.cuda._sleep(int(5 * get_cycles_per_ms()))
        b.copy_(a, non_blocking=True)
        c.copy_(b, non_blocking=True)
        self.assertEqual(a, c, exact_dtype=False)


    def test_to_cpu_blocking_by_default(self):
        src = torch.randn(1000, device="cuda")
        torch.cuda.synchronize()
        torch.cuda._sleep(int(5 * get_cycles_per_ms()))
        dst = src.to(device="cpu")
        self.assertEqual(torch.cuda.current_stream().query(), True)
        self.assertEqual(src, dst)
        self.assertFalse(dst.is_pinned())


    def test_torch_manual_seed_seeds_cuda_devices(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().cuda()
            torch.manual_seed(2)
            self.assertEqual(torch.cuda.initial_seed(), 2)
            x.uniform_()
            torch.manual_seed(2)
            y = x.clone().uniform_()
            self.assertEqual(x, y)
            self.assertEqual(torch.cuda.initial_seed(), 2)


    def test_manual_seed(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().cuda()
            torch.cuda.manual_seed(2)
            self.assertEqual(torch.cuda.initial_seed(), 2)
            x.uniform_()
            a = torch.bernoulli(torch.full_like(x, 0.5))
            torch.cuda.manual_seed(2)
            y = x.clone().uniform_()
            b = torch.bernoulli(torch.full_like(x, 0.5))
            self.assertEqual(x, y)
            self.assertEqual(a, b)
            self.assertEqual(torch.cuda.initial_seed(), 2)


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


    def test_events(self):
        stream = torch.cuda.current_stream()
        event = torch.cuda.Event(enable_timing=True)
        self.assertTrue(event.query())
        start_event = torch.cuda.Event(enable_timing=True)
        stream.record_event(start_event)
        torch.cuda._sleep(int(5 * get_cycles_per_ms()))
        stream.record_event(event)
        self.assertFalse(event.query())
        event.synchronize()
        self.assertTrue(event.query())
        self.assertGreater(start_event.elapsed_time(event), 0)

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
        x = torch.randn(1024, device="cuda")
        del x

        pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool):
            y = torch.randn(512, device="cuda")
            torch.cuda.empty_cache()
            del y


class TestSupaMallocAsync(TestCase):
    @pytest.mark.regression
    def test_max_split_expandable(self):
        try:
            torch.cuda.memory.empty_cache()
            mb = 1024 * 1024
            _, all_memory = torch.cuda.memory.mem_get_info()
            pre_reserved = torch.cuda.memory_reserved()
            total_allowed = 120 * mb + pre_reserved
            fraction_allowed = total_allowed / all_memory
            self.assertEqual(int(fraction_allowed * all_memory), total_allowed)
            torch.cuda.memory.set_per_process_memory_fraction(fraction_allowed)

            def alloc(n):
                return torch.ones(n * mb, dtype=torch.int8, device="cuda")

            torch.cuda.memory._set_allocator_settings(
                "expandable_segments:False,max_split_size_mb:40"
            )
            a = alloc(40)
            torch.cuda.memory._set_allocator_settings(
                "expandable_segments:True,max_split_size_mb:40"
            )
            b = alloc(40)
            torch.cuda.memory._set_allocator_settings(
                "expandable_segments:False,max_split_size_mb:40"
            )
            c = alloc(40)
            with self.assertRaises(torch.OutOfMemoryError):
                alloc(40)
            del a, b, c
            # force release_cached_blocks to run with some expandable segments in the free list
            alloc(120)
        finally:
            torch.cuda.memory.set_per_process_memory_fraction(1.0)

    @pytest.mark.regression
    def test_garbage_collect_expandable(self):
        try:
            torch.cuda.memory.empty_cache()
            mb = 1024 * 1024
            _, all_memory = torch.cuda.memory.mem_get_info()
            pre_reserved = torch.cuda.memory_reserved()
            total_allowed = 120 * mb + pre_reserved
            fraction_allowed = total_allowed / all_memory
            self.assertEqual((fraction_allowed * all_memory), total_allowed)
            torch.cuda.memory.set_per_process_memory_fraction(fraction_allowed)

            def alloc(n):
                return torch.ones(n * mb, dtype=torch.int8, device="cuda")

            torch.cuda.memory._set_allocator_settings(
                "expandable_segments:False,garbage_collection_threshold:0.5"
            )
            a = alloc(40)
            torch.cuda.memory._set_allocator_settings(
                "expandable_segments:True,garbage_collection_threshold:0.5"
            )
            b = alloc(40)
            del a, b
            # causes GC to run. The expandable segment block will be split
            # so GC would not attempt to free it anyway, but this at least makes sure
            # expandable_segment blocks can be in the free list when this is called.
            alloc(80)
        finally:
            torch.cuda.memory.set_per_process_memory_fraction(1.0)


    @pytest.mark.sanity
    @pytest.mark.regression
    def test_cpp_memory_snapshot_pickle(self):
        from torch_supa.utils.cpp_extension import load_inline
        torch_supa_home = os.path.dirname(os.path.realpath(torch_supa.__file__))

        source = """
        #include <torch_supa/csrc/supa/memory_snapshot.h>
        py::object do_snapshot() {
            std::string data = torch_supa::supa::_memory_snapshot_pickled();
            return py::bytes(data);
        }
        void record(bool e, bool ctx) {
            torch_supa::supa::_record_memory_history(e, ctx, 10, ctx, ctx);
        }
        """
        m = load_inline(
            name="snapshot", 
            cpp_sources=[source], 
            functions=["do_snapshot", "record"],
            with_supa=True,
            extra_ldflags=[f"-L{torch_supa_home}/lib", "-ltorch_supa"],
        )
        for ctx in (False, True):
            try:
                m.record(True, ctx)

                @torch.jit.script
                def the_script_fn():
                    return torch.rand(31, 41, device="cuda")

                def run():
                    t = the_script_fn()
                    return pickle.loads(m.do_snapshot())

                mem = run()
                found = False
                for s in mem["segments"]:
                    for b in s["blocks"]:
                        if b["state"] == "active_allocated":
                            if b["requested_size"] == 31 * 41 * 4:
                                if ctx:
                                    frame_text = str(b["frames"])
                                    # C++ frame
                                    self.assertTrue("::rand" in frame_text)
                                    # script frame
                                    self.assertTrue("the_script_fn" in frame_text)
                                    # python frame
                                    self.assertTrue("case.py" in frame_text)
                                found = True
                last_action = mem["device_traces"][0][-1]
                self.assertEqual(last_action["action"], "alloc")
                self.assertEqual(last_action["size"], 31 * 41 * 4)
                self.assertTrue(found)
            finally:
                m.record(False, False)


    @pytest.mark.sanity
    @pytest.mark.regression
    def test_memory_plots_free_stack(self):
        for context in ["alloc", "all", "state"]:
            try:
                torch.cuda.memory.empty_cache()
                torch.cuda.memory._record_memory_history(context=context)
                x = None

                def thealloc():
                    nonlocal x
                    x = torch.rand(3, 4, device="cuda")

                def thefree():
                    nonlocal x
                    del x

                thealloc()
                thefree()
                ss = json.dumps(torch.cuda.memory._snapshot())
                self.assertEqual(("thefree" in ss), (context == "all"))
                self.assertEqual(("thealloc" in ss), (context != "state"))
            finally:
                torch.cuda.memory._record_memory_history(None)


    @pytest.mark.sanity
    @pytest.mark.regression
    def test_memory_plots_free_segment_stack(self):
        for context in ["alloc", "all", "state"]:
            try:
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.memory.empty_cache()
                torch.cuda.memory._record_memory_history(context=context)
                x = torch.rand(3, 4, device="cuda")
                del x
                torch.cuda.memory.empty_cache()

                ss = json.dumps(torch.cuda.memory._snapshot())
                self.assertEqual(("empty_cache" in ss), (context == "all"))
            finally:
                torch.cuda.memory._record_memory_history(None)


def cudagraphify(fn, inputs, pool=None):
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(*inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, pool=pool):
        static_outputs = fn(*inputs)

    return graph, static_outputs


@pytest.mark.sanity
@pytest.mark.regression
class TestBlockStateAbsorption(TestCase):
    def test_check_pool_live_allocations(self):
        def foo():
            return torch.ones([4], device="cuda")

        pool = torch.cuda.graph_pool_handle()
        graph, outputs = cudagraphify(foo, [], pool=pool)

        index = outputs[0].device.index

        def check(live_dps):
            return torch._C._cuda_checkPoolLiveAllocations(index, pool, live_dps)

        self.assertTrue(check({outputs[0].data_ptr()}))

        self.assertFalse(check({outputs[0].data_ptr(), 0}))
        self.assertFalse(check(set()))

        del outputs
        self.assertTrue(check(set()))


class TestMemPool(TestCase):
    def get_dummy_allocator(self):
        from torch_supa.utils.cpp_extension import load_inline
        torch_supa_home = os.path.dirname(os.path.realpath(torch_supa.__file__))

        dummy_allocator_source = """
        #include <torch/extension.h>
        #include <torch_supa/csrc/core/supa/SUPAException.h>
        #include <supa_runtime.h>

        extern "C" {
          C10_EXPORT int called_dummy_alloc = 0;
          C10_EXPORT int called_dummy_free = 0;

          C10_EXPORT void* dummy_alloc(size_t size, int device, void* stream) {
            called_dummy_alloc = 123;
            void* ptr;
            C10_SUPA_CHECK(supaMallocManaged(&ptr, size));
            return ptr;
          }

          C10_EXPORT void dummy_free(void* ptr, size_t size, int device, void* stream) {
            called_dummy_free = 321;
            C10_SUPA_CHECK(supaFree(ptr));
          }
        }
        """
        dummy_allocator_libname = "dummy_allocator"
        dummy_allocator = load_inline(
            name=dummy_allocator_libname,
            cpp_sources=dummy_allocator_source,
            is_python_module=False,
            keep_intermediates=False,
            verbose=True,
            with_supa=True,
            extra_ldflags=[f"-L{torch_supa_home}/lib", "-ltorch_supa"],
        )
        allocator = torch.cuda.memory.CUDAPluggableAllocator(
            dummy_allocator,
            "dummy_alloc",
            "dummy_free",
        )
        return allocator, dummy_allocator


    @pytest.mark.sanity
    @pytest.mark.regression
    def test_mempool_id(self):
        pool1 = torch.cuda.graph_pool_handle()
        pool2 = torch.cuda.MemPool().id

        # first value of id in a user created pool is always zero
        self.assertEqual(pool1[0] == 0, pool2[0] == 0)

        # each call to torch.cuda.graph_pool_handle() or torch.cuda.MemPool()
        # increments the id
        self.assertTrue(abs(pool2[1] - pool1[1]) > 0)


    @pytest.mark.sanity
    @pytest.mark.regression
    def test_mempool_context(self):
        active_pool = torch.cuda.MemPoolContext.active_pool()

        # there is no active pool if none was made active
        self.assertEqual(active_pool, None)

        pool = torch.cuda.MemPool()
        ctx = torch.cuda.MemPoolContext(pool)
        active_pool = torch.cuda.MemPoolContext.active_pool()

        # pool was made active
        self.assertEqual(active_pool, pool)

        del ctx
        active_pool = torch.cuda.MemPoolContext.active_pool()

        # ctx was deleted, so active pool is the previous one
        self.assertEqual(active_pool, None)


    @pytest.mark.regression
    def test_mempool_with_allocator(self):
        pool = torch.cuda.MemPool()

        # MemPool doesn't have an allocator by default
        self.assertEqual(pool.allocator, None)
        allocator, dummy_allocator = self.get_dummy_allocator()

        pool = torch.cuda.MemPool(allocator.allocator())

        # pool should point to the same allocator as the one passed into it
        self.assertEqual(allocator.allocator(), pool.allocator)

        # pool's use count should be 1 at this point as MemPool object
        # holds a reference
        self.assertEqual(pool.use_count(), 1)

        # no allocations happened yet, so called_dummy_alloc and
        # called_dummy_free should be 0
        alloc_lib = ctypes.CDLL(dummy_allocator)
        called_dummy_alloc = ctypes.c_int.in_dll(alloc_lib, "called_dummy_alloc")
        called_dummy_free = ctypes.c_int.in_dll(alloc_lib, "called_dummy_free")
        self.assertEqual(called_dummy_alloc.value, 0)
        self.assertEqual(called_dummy_free.value, 0)

        nelem_1mb = 1024 * 1024 // 4

        with torch.cuda.use_mem_pool(pool):
            out_0 = torch.randn(nelem_1mb, device="cuda")

            # pool's use count should be 2 at this point as use_mem_pool
            # holds a reference
            self.assertEqual(pool.use_count(), 2)

        # pool's use count should be back to 1 at this point as use_mem_pool
        # released its reference
        self.assertEqual(pool.use_count(), 1)

        # called_dummy_alloc should be 123 if dummy_alloc was used to allocate
        # out tensor
        self.assertEqual(called_dummy_alloc.value, 123)

        with torch.cuda.use_mem_pool(pool):
            # pool should have 1 segment since we made a small allocation (1 MB)
            # above and so the SUPACachingAllocator packed it into a 2 MB buffer
            self.assertEqual(len(pool.snapshot()), 1)

            out_1 = torch.randn(nelem_1mb, device="cuda")

            # pool should still have 1 segment since we made another small allocation
            # (1 MB) that got packed into the existing 2 MB buffer
            self.assertEqual(len(pool.snapshot()), 1)

            out_2 = torch.randn(nelem_1mb, device="cuda")

            # pool now should have 2 segments since the SUPACachingAllocator had
            # to make a new 2 MB buffer to accomodate out_2
            self.assertEqual(len(pool.snapshot()), 2)

        del out_0, out_1, out_2

        # pool's destructor calls emptyCache()
        del pool

        # called_dummy_free should be 321 if dummy_free was used to deallocate
        # out tensor
        self.assertEqual(called_dummy_free.value, 321)
