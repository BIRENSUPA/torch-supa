# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import gc
import pytest

import torch

import torch_supa._C._transfer as _transfer
from torch_supa.testing.common_utils import assert_allclose
from torch_supa.utils import torch_version_le
FLOAT_RTOL, FLOAT_ATOL = 1e-5, 5e-5

@pytest.mark.sanity
class TestAccelerator:

    @pytest.fixture(autouse=True, params=[True, False])
    def transfer_device_type(self, request):
        pre_status = _transfer.device_type_status()
        _transfer.device_type(request.param)
        yield request.param
        _transfer.device_type(pre_status)

    @pytest.fixture(autouse=True)
    def restore_accelerator_state(self):
        device_count = torch.accelerator.device_count()
        saved_device = torch.accelerator.current_device_index()
        saved_streams = [
            torch.accelerator.current_stream(d) for d in range(device_count)
        ]
        yield
        for stream in saved_streams:
            torch.accelerator.set_stream(stream)
        torch.accelerator.set_device_index(saved_device)

    def test_set_device_index(self, transfer_device_type):
        device_type = "cuda" if transfer_device_type else "supa"
        devices = [
            f"{device_type}:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
        ]
        for device in devices:
            torch.accelerator.set_device_index(device)
            assert torch.accelerator.current_accelerator().type == device_type
            assert torch.accelerator.current_accelerator().index is None
            assert torch.accelerator.current_device_index() == int(device.split(":")[1])

    def test_current_accelerator(self, transfer_device_type):
        assert torch.accelerator.is_available()
        accelerators = ["cuda"] if transfer_device_type else ["supa"]
        for accelerator in accelerators:
            if torch.get_device_module(accelerator).is_available():
                assert torch.accelerator.current_accelerator().type == accelerator
                assert torch.accelerator.current_accelerator().index is None
                with pytest.raises(ValueError, match="doesn't match the current accelerator"):
                    torch.accelerator.set_device_index("cpu")

    @pytest.mark.skipif(
        torch.accelerator.device_count() <= 1, reason="only one accelerator detected"
    )
    def test_generic_multi_device_behavior(self):
        orig_device = torch.accelerator.current_device_index()
        target_device = (orig_device + 1) % torch.accelerator.device_count()

        torch.accelerator.set_device_index(target_device)
        assert target_device == torch.accelerator.current_device_index()
        torch.accelerator.set_device_index(orig_device)
        assert orig_device == torch.accelerator.current_device_index()

        s1 = torch.Stream(target_device)
        torch.accelerator.set_stream(s1)
        assert target_device == torch.accelerator.current_device_index()
        torch.accelerator.synchronize(orig_device)
        assert target_device == torch.accelerator.current_device_index()

    def test_generic_stream_behavior(self):

        s1 = torch.Stream()
        s2 = torch.Stream()
        torch.accelerator.set_stream(s1)
        assert torch.accelerator.current_stream() == s1
        event = torch.Event()
        a = torch.randn(1000)
        b = torch.randn(1000)
        c = a + b
        torch.accelerator.set_stream(s2)
        assert torch.accelerator.current_stream() == s2
        a_acc = a.to(torch.accelerator.current_accelerator(), non_blocking=True)
        b_acc = b.to(torch.accelerator.current_accelerator(), non_blocking=True)
        torch.accelerator.set_stream(s1)
        assert torch.accelerator.current_stream() == s1
        event.record(s2)
        event.synchronize()
        c_acc = a_acc + b_acc
        event.record(s2)
        torch.accelerator.synchronize()
        assert event.query()
        assert_allclose(c_acc.cpu(), c, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)

    def test_current_stream_query(self):
        s = torch.accelerator.current_stream()
        assert torch.accelerator.current_stream(s.device) == s
        assert torch.accelerator.current_stream(s.device.index) == s
        assert torch.accelerator.current_stream(str(s.device)) == s
        other_device = torch.device("cpu")
        with pytest.raises(ValueError, match="doesn't match the current accelerator"):
            torch.accelerator.current_stream(other_device)

    @pytest.mark.skipif(torch_version_le(2, 6, 0), reason="requires torch version higher than 2.6.0")
    def test_device_context_manager(self):
        prev_device = torch.accelerator.current_device_index()
        with torch.accelerator.device_index(None):
            assert torch.accelerator.current_device_index() == prev_device
        assert torch.accelerator.current_device_index() == prev_device
        with torch.accelerator.device_index(0):
            assert torch.accelerator.current_device_index() == 0
        assert torch.accelerator.current_device_index() == prev_device

    @pytest.mark.skipif(
        torch.accelerator.device_count() <= 1, reason="only one accelerator detected"
    )
    @pytest.mark.skipif(torch_version_le(2, 6, 0), reason="requires torch version higher than 2.6.0")
    def test_multi_device_context_manager(self):
        src_device = 0
        dst_device = 1
        torch.accelerator.set_device_index(src_device)
        with torch.accelerator.device_index(dst_device):
            assert torch.accelerator.current_device_index() == dst_device
        assert torch.accelerator.current_device_index() == src_device

    @pytest.mark.skipif(torch_version_le(2, 6, 0), reason="requires torch version higher than 2.6.0")
    def test_stream_context_manager(self):
        prev_stream = torch.accelerator.current_stream()
        with torch.Stream() as s:
            assert torch.accelerator.current_stream() == s
        assert torch.accelerator.current_stream() == prev_stream

    @pytest.mark.skipif(
        torch.accelerator.device_count() <= 1, reason="only one accelerator detected"
    )
    @pytest.mark.skipif(torch_version_le(2, 6, 0), reason="requires torch version higher than 2.6.0")
    def test_multi_device_stream_context_manager(self):
        src_device = 0
        dst_device = 1
        torch.accelerator.set_device_index(src_device)
        src_prev_stream = torch.accelerator.current_stream()
        dst_prev_stream = torch.accelerator.current_stream(dst_device)
        with torch.Stream(dst_device) as dst_stream:
            assert torch.accelerator.current_device_index() == dst_device
            assert torch.accelerator.current_stream() == dst_stream
            assert torch.accelerator.current_stream(src_device) == src_prev_stream
        assert torch.accelerator.current_device_index() == src_device
        assert torch.accelerator.current_stream() == src_prev_stream
        assert torch.accelerator.current_stream(dst_device) == dst_prev_stream

    @pytest.mark.skipif(
        hasattr(torch, "TEST_MPS") and torch.TEST_MPS, reason="MPS doesn't support pin memory!"
    )
    def test_pin_memory_on_non_blocking_copy(self):
        t_acc = torch.randn(100).to(torch.accelerator.current_accelerator())
        t_host = t_acc.to("cpu", non_blocking=True)
        torch.accelerator.synchronize()
        assert t_host.is_pinned()
        assert_allclose(t_acc.cpu(), t_host, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)

    @pytest.mark.skipif(torch_version_le(2, 6, 0), reason="requires torch version higher than 2.6.0")
    def test_generic_event_behavior(self):
        event1 = torch.Event(enable_timing=False)
        event2 = torch.Event(enable_timing=False)
        with pytest.raises(
            ValueError,
            match="Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)

        event1 = torch.Event(enable_timing=True)
        event2 = torch.Event(enable_timing=True)
        with pytest.raises(
            ValueError,
            match="Both events must be recorded before calculating elapsed time",
        ):
            event1.elapsed_time(event2)

        # check default value of enable_timing: False
        event1 = torch.Event()
        event2 = torch.Event()
        with pytest.raises(
            ValueError,
            match="Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)

    @pytest.mark.skipif(
        hasattr(torch, "TEST_MPS") and torch.TEST_MPS,
        reason="MPS doesn't support torch.accelerator memory API!",
    )
    @pytest.mark.skipif(torch_version_le(2, 6, 0), reason="requires torch version higher than 2.6.0")
    def test_memory_stats(self):
        # Ensure that device allocator is initialized
        acc = torch.accelerator.current_accelerator()
        tmp = torch.randn(100, device=acc)
        del tmp
        gc.collect()
        assert torch._C._accelerator_isAllocatorInitialized()
        torch.accelerator.empty_cache()

        pool_type = ["all", "small_pool", "large_pool"]
        metric_type = ["peak", "current", "allocated", "freed"]
        stats_type = [
            "allocated_bytes",
            "reserved_bytes",
            "active_bytes",
            "requested_bytes",
        ]
        mem_stats = torch.accelerator.memory_stats()
        expected_stats = [
            f"{st}.{pt}.{mt}" for st in stats_type for pt in pool_type for mt in metric_type
        ]
        missing_stats = [stat for stat in expected_stats if stat not in mem_stats]
        assert len(missing_stats) == 0, f"Missing expected memory statistics: {missing_stats}"

        prev_allocated = torch.accelerator.memory_allocated()
        prev_reserved = torch.accelerator.memory_reserved()
        prev_max_allocated = torch.accelerator.max_memory_allocated()
        prev_max_reserved = torch.accelerator.max_memory_reserved()
        assert prev_allocated >= 0
        assert prev_reserved >= 0
        assert prev_max_allocated > 0
        assert prev_max_reserved > 0
        tmp = torch.ones(256, device=acc)
        assert torch.accelerator.memory_allocated() > prev_allocated
        assert torch.accelerator.memory_reserved() >= prev_reserved
        del tmp
        gc.collect()
        torch.accelerator.empty_cache()
        torch.accelerator.reset_peak_memory_stats()
        assert torch.accelerator.memory_allocated() == prev_allocated
        assert torch.accelerator.memory_reserved() == prev_reserved
        torch.accelerator.reset_accumulated_memory_stats()
        prev_max_allocated = torch.accelerator.max_memory_allocated()
        prev_max_reserved = torch.accelerator.max_memory_reserved()
        # Activate 1kB memory
        prev_active_current = torch.accelerator.memory_stats()["active_bytes.all.current"]
        tmp = torch.randn(256, dtype=torch.float32, device=acc)
        # Detect if the current active memory is 1kB
        # (1024 + 32 + 512 - 1) / 512 * 512 = 1536 
        assert (
            torch.accelerator.memory_stats()["active_bytes.all.current"]
            == 1536 + prev_active_current
        )
        assert torch.accelerator.memory_stats()["active_bytes.all.freed"] == 0
        del tmp
        gc.collect()
        torch.accelerator.empty_cache()
        assert torch.accelerator.memory_stats()["active_bytes.all.current"] == prev_active_current
        assert torch.accelerator.memory_stats()["active_bytes.all.freed"] == 1536
        torch.accelerator.reset_peak_memory_stats()
        assert torch.accelerator.max_memory_allocated() == prev_max_allocated
        assert torch.accelerator.max_memory_reserved() == prev_max_reserved
