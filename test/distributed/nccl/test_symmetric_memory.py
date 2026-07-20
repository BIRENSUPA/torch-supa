# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# Owner(s): ["module: c10d"]

import os
from contextlib import nullcontext
from unittest import skip

from torch_supa.utils import torch_version_ge 

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch._C._autograd import DeviceType
from torch._C._distributed_c10d import _SymmetricMemory
if torch_version_ge(2, 8, 0):
    from torch._inductor.utils import fresh_cache, run_and_get_triton_code
else:
    from torch._inductor.utils import fresh_inductor_cache as fresh_cache, run_and_get_triton_code
from torch.distributed._functional_collectives import all_gather_tensor
from torch.distributed._symmetric_memory import (
    _test_mode,
    enable_symm_mem_for_group,
)
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_multicast_support,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
    TestCase,
)

import pytest

os.environ["NCCL_TOPO_FILE"] = os.getenv("BIREN_HOME", "/usr/local/birensupa/all/latest") + "/bccl/xml/topo_2c.xml"

test_contexts = [nullcontext, _test_mode]

# So that tests are written in device-agnostic way
device_type = "cuda"
device_module = torch.get_device_module(device_type)


def requires_cuda_p2p_access():
    cuda_p2p_access_available = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (8, 0)
        and torch.cuda.device_count() >= 2
    )
    num_devices = torch.cuda.device_count()
    for i in range(num_devices - 1):
        for j in range(i + 1, num_devices):
            if not torch.cuda.can_device_access_peer(i, j):
                cuda_p2p_access_available = False
                break
        if not cuda_p2p_access_available:
            break

    return skip_but_pass_in_sandcastle_if(
        not cuda_p2p_access_available,
        "cuda p2p access is not available",
    )


@instantiate_parametrized_tests
@requires_cuda_p2p_access()
@pytest.mark.sanity
@pytest.mark.regression
class SymmetricMemoryTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self, set_device: bool = True):
        if set_device:
            torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(42 + self.rank)

    def test_has_multicast_support(self) -> None:
        # validate that has_multicast_support() returns "false" instead of throwing
        self.assertFalse(_SymmetricMemory.has_multicast_support(DeviceType.CPU, 0))
        # NOTE: DeviceType.CUDA is implicitly tested through @requires_multicast_support

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_blink_connectivity_detection(self) -> None:
        from torch._C._distributed_c10d import _detect_dma_connectivity

        connectivity = _detect_dma_connectivity(DeviceType.PrivateUse1, "blink")
        self.assertEqual(connectivity.device_type, DeviceType.PrivateUse1)
        self.assertEqual(connectivity.connection_type, "blink")
        self.assertEqual(len(connectivity.matrix), torch.cuda.device_count())
        for row in connectivity.matrix:
            self.assertEqual(len(row), torch.cuda.device_count())

    def _get_test_alloc_args(self):
        shape = (64, 64)
        stride = (64, 1)
        dtype = torch.float32
        device = self.device
        group_name = "0"
        return (shape, stride, dtype, device, group_name)

    def _verify_symmetric_memory(self, symm_mem_hdl):
        self.assertEqual(symm_mem_hdl.world_size, self.world_size)

        buf = symm_mem_hdl.get_buffer(
            0, (symm_mem_hdl.buffer_size // 4,), torch.float32
        )
        self.assertEqual(buf.storage_offset(), 0)
        self.assertEqual(buf.untyped_storage().size(), symm_mem_hdl.buffer_size)

        if symm_mem_hdl.rank == 0:
            symm_mem_hdl.wait_signal(src_rank=1)
            self.assertTrue(buf.eq(42).all())
        else:
            buf.fill_(42)
            symm_mem_hdl.put_signal(dst_rank=0)

        symm_mem_hdl.barrier()

        if symm_mem_hdl.rank == 0:
            symm_mem_hdl.barrier()
            self.assertTrue(buf.eq(43).all())
        else:
            buf.fill_(43)
            symm_mem_hdl.barrier()

        symm_mem_hdl.barrier()

    @skip_if_lt_x_gpu(2)
    @parametrize("set_device", [True, False])
    def test_empty_strided_p2p(self, set_device: bool) -> None:
        self._init_process(set_device)
        enable_symm_mem_for_group(dist.group.WORLD.group_name)

        alloc_args = self._get_test_alloc_args()

        t = torch.empty((64, 64), device=self.device)
        self.assertIsNone(_SymmetricMemory.rendezvous(t))

        t = _SymmetricMemory.empty_strided_p2p(*alloc_args)
        symm_mem_hdl = _SymmetricMemory.rendezvous(t)

        del t
        self._verify_symmetric_memory(symm_mem_hdl)

    @skipIfRocm  # started failing during ROCm 6.4 CI upgrade
    @skip_if_lt_x_gpu(2)
    @parametrize("set_device", [True, False])
    def test_empty_strided_p2p_persistent(self, set_device: bool) -> None:
        self._init_process(set_device)
        enable_symm_mem_for_group(dist.group.WORLD.group_name)

        alloc_args = self._get_test_alloc_args()

        t = _SymmetricMemory.empty_strided_p2p(*alloc_args, alloc_id=42)
        data_ptr = t.data_ptr()

        # Verify that persistent allocation would fail if there's an active
        # allocation with the same alloc_id.
        with self.assertRaises(RuntimeError):
            _SymmetricMemory.empty_strided_p2p(*alloc_args, alloc_id=42)

        # Verify that persistent allocation would succeed in lieu of activate
        # allocations with the same alloc_id, and the returned tensor would
        # have the same data pointer.
        del t
        t = _SymmetricMemory.empty_strided_p2p(*alloc_args, alloc_id=42)
        self.assertEqual(t.data_ptr(), data_ptr)

        symm_mem_hdl = _SymmetricMemory.rendezvous(t)
        self._verify_symmetric_memory(symm_mem_hdl)

    @skip_if_lt_x_gpu(2)
    def test_get_signal_pad(self) -> None:
        self._init_process()

        t = symm_mem.empty(1, device="cuda")
        symm_mem_hdl = symm_mem.rendezvous(t, group=dist.group.WORLD)
        peer_rank = (self.rank + 1) % self.world_size

        signal_pad = symm_mem_hdl.get_signal_pad(self.rank)
        self.assertEqual(
            signal_pad.data_ptr(), symm_mem_hdl.signal_pad_ptrs[symm_mem_hdl.rank]
        )

        signal_pad = symm_mem_hdl.get_signal_pad(peer_rank)
        self.assertEqual(signal_pad.dtype, torch.uint32)
        self.assertEqual(signal_pad.numel(), symm_mem_hdl.signal_pad_size // 4)

        # Only specify sizes
        signal_pad = symm_mem_hdl.get_signal_pad(peer_rank, (8, 8))
        self.assertEqual(signal_pad.dtype, torch.uint32)
        self.assertEqual(signal_pad.numel(), 64)

        # Only specify dtype
        signal_pad = symm_mem_hdl.get_signal_pad(peer_rank, dtype=torch.uint64)
        self.assertEqual(signal_pad.dtype, torch.uint64)
        self.assertEqual(signal_pad.numel(), symm_mem_hdl.signal_pad_size // 8)

        # Specify both sizes and dtype
        signal_pad = symm_mem_hdl.get_signal_pad(peer_rank, (8, 8), dtype=torch.uint64)
        self.assertEqual(signal_pad.dtype, torch.uint64)
        self.assertEqual(signal_pad.numel(), 64)

        # Sanity check that writes to buffer doesn't corrupt signal_pad
        t = symm_mem.empty(1, device="cuda")
        symm_mem_hdl = symm_mem.rendezvous(t, group=dist.group.WORLD)
        signal_pad = symm_mem_hdl.get_signal_pad(self.rank)
        signal_pad.fill_(42)
        t.fill_(0)
        self.assertTrue(signal_pad.eq(42).all())

    @requires_cuda
    def test_allow_overlapping_devices(self) -> None:
        os.environ["TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES"] = "1"
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        t = symm_mem.empty(64, device="cuda:0")
        symm_mem_hdl = symm_mem.rendezvous(t, group=dist.group.WORLD)

        self.assertEqual(symm_mem_hdl.rank, self.rank)
        self.assertEqual(symm_mem_hdl.world_size, self.world_size)

        for rank in range(self.world_size):
            buf = symm_mem_hdl.get_buffer(rank, (64,), torch.float32)
            if rank == self.rank:
                self.assertEqual(buf.data_ptr(), t.data_ptr())
            else:
                self.assertEqual(buf.device, t.device)

        os.environ["TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES"] = "0"

    @skip_if_lt_x_gpu(4)
    def test_reduce_scatter_corner_cases(self) -> None:
        self._init_process()
        dtype = torch.bfloat16
        group_name = dist.group.WORLD.group_name
        t = symm_mem.empty(16384, dtype=dtype, device=self.device).fill_(0)
        symm_mem.rendezvous(t, group=group_name)
        res = t[:0]
        out_size = res.shape[0] // self.world_size
        out = torch.empty(out_size, dtype=dtype, device=self.device)
        torch.ops.symm_mem.reduce_scatter_out(res, group_name, False, out)
        res = t[:48]
        out_size = res.shape[0] // self.world_size
        out = torch.empty(out_size, dtype=dtype, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "divisible"):
            torch.ops.symm_mem.reduce_scatter_out(res, group_name, False, out)
        res = t[: 2 * 48].view(2, 48)
        out = torch.empty(2, 48 // self.world_size, dtype=dtype, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "divisible"):
            torch.ops.symm_mem.reduce_scatter_out(res, group_name, True, out)

    def _verify_reduce_scatter_result(self, inp, res):
        gathered_res = all_gather_tensor(res, 0, "0").view(self.world_size, *res.shape)
        gathered_inps = all_gather_tensor(inp, 0, "0").view(self.world_size, *inp.shape)
        sum_inps = gathered_inps.sum(0)
        slice_width = sum_inps.shape[-1] // self.world_size
        for i in range(self.world_size):
            torch.testing.assert_close(
                gathered_res[i],
                sum_inps[..., i * slice_width : (i + 1) * slice_width],
                rtol=1e-01,
                atol=1.1e-01,
            )

    @skip_if_lt_x_gpu(4)
    @requires_multicast_support()
    @parametrize("align_bytes", [4, 8, 16])
    def test_multimem_all_gather(self, align_bytes: int) -> None:
        self._init_process()
        group_name = dist.group.WORLD.group_name

        input_numel = 32
        shift = align_bytes // 4
        input = torch.zeros(shift + input_numel, device=self.device)[shift:].fill_(
            self.rank
        )

        out = symm_mem.empty(
            shift + input_numel * self.world_size, device=self.device
        ).zero_()[shift:]
        symm_mem.rendezvous(out, group=group_name)

        torch.ops.symm_mem.multimem_all_gather_out(input, group_name, out)
        ref = torch.ops._c10d_functional.all_gather_into_tensor(
            input, self.world_size, group_name
        )
        ref = torch.ops._c10d_functional.wait_tensor(ref)

        self.assertTrue(out.eq(ref).all())


@instantiate_parametrized_tests
@requires_cuda_p2p_access()
@pytest.mark.sanity
@pytest.mark.regression
class LoweringTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        enable_symm_mem_for_group(dist.group.WORLD.group_name)
        torch.manual_seed(42 + self.rank)

        torch._inductor.config._collective.auto_select = True

    @skip("Fails with 'one_shot_all_reduce' not found in AOT graph, TODO: fix")
    @skipIfRocm  # requires registered-buffer support
    @skip_if_lt_x_gpu(2)
    @fresh_cache()
    def test_lowering_one_shot_all_reduce(self):
        self._init_process()
        arg = torch.rand(4, 4, device=self.device)

        def func_0(x):
            x = x + 1
            x = torch.ops._c10d_functional.all_reduce(x, "sum", "0")
            return torch.ops._c10d_functional.wait_tensor(x)

        compiled_0 = torch.compile(func_0, fullgraph=True)
        code_0 = run_and_get_triton_code(compiled_0, arg)

        self.assertIn("one_shot_all_reduce", code_0)
        self.assertNotIn("return (buf0", code_0)

        # All-reduce on a slice view
        def func_1(x):
            x = x + 1
            x = x[2:]
            x = torch.ops._c10d_functional.all_reduce(x, "sum", "0")
            return torch.ops._c10d_functional.wait_tensor(x)

        compiled_1 = torch.compile(func_1, fullgraph=True)
        code_1 = run_and_get_triton_code(compiled_1, arg)

        self.assertIn("one_shot_all_reduce", code_1)
        self.assertNotIn("return (buf0", code_1)

        # All-reduce on input
        def func_2(x):
            x = torch.ops._c10d_functional.all_reduce(x, "sum", "0")
            return torch.ops._c10d_functional.wait_tensor(x)

        compiled_2 = torch.compile(func_2, fullgraph=True)
        code_2 = run_and_get_triton_code(compiled_2, arg)

        self.assertNotIn("one_shot_all_reduce", code_2)

        # All-reduce on matmul output
        def func_3(x):
            x = x @ x
            x = torch.ops._c10d_functional.all_reduce(x, "sum", "0")
            return torch.ops._c10d_functional.wait_tensor(x)

        compiled_3 = torch.compile(func_3, fullgraph=True)
        code_3 = run_and_get_triton_code(compiled_3, arg)

        self.assertIn("one_shot_all_reduce", code_3)
        self.assertNotIn("return (buf0", code_3)


@pytest.mark.sanity
@pytest.mark.regression
class SymmMemSingleProcTest(TestCase):
    @requires_cuda
    def test_stream_write_value32(self):
        tensor = torch.zeros(4, dtype=torch.uint32, device="cuda")
        expect = torch.tril(torch.ones(4, 4, device="cuda")).to(torch.uint32)

        for i in range(4):
            _SymmetricMemory.stream_write_value32(tensor, i, 1)
            torch.testing.assert_close(tensor, expect[i])

        with self.assertRaises(RuntimeError):
            _SymmetricMemory.stream_write_value32(tensor, offset=-1, val=1)

        with self.assertRaises(RuntimeError):
            _SymmetricMemory.stream_write_value32(tensor, offset=0, val=4294967296)

    @requires_cuda
    def test_memset32(self):
        t = _SymmetricMemory.empty_strided_p2p(
            (64,),
            (1,),
            dtype=torch.uint32,
            device=torch.device("cuda:0"),
            group_name="0",
        ).fill_(0)

        _SymmetricMemory.memset32(t, offset=32, val=1, count=16)
        self.assertTrue(t[:32].eq(0).all())
        self.assertTrue(t[32:48].eq(1).all())
        self.assertTrue(t[48:].eq(0).all())

        with self.assertRaisesRegex(
            RuntimeError, "input must be a flat, contiguous uint32 tensor"
        ):
            _SymmetricMemory.memset32(t.view(8, 8), offset=0, val=1, count=1)

        with self.assertRaisesRegex(
            RuntimeError, "input must be a flat, contiguous uint32 tensor"
        ):
            _SymmetricMemory.memset32(t.view(torch.float32), offset=0, val=1, count=1)

        with self.assertRaisesRegex(
            RuntimeError, "offset must be greater than or equal to 0"
        ):
            _SymmetricMemory.memset32(t, offset=-1, val=1, count=1)

        with self.assertRaisesRegex(
            RuntimeError, r"val must be in the range of.*\(uint32_t\)"
        ):
            _SymmetricMemory.memset32(t, offset=0, val=4294967296, count=1)

        with self.assertRaisesRegex(RuntimeError, "count must be a positive integer"):
            _SymmetricMemory.memset32(t, offset=0, val=1, count=-1)

        with self.assertRaisesRegex(RuntimeError, "count must be a positive integer"):
            _SymmetricMemory.memset32(t, offset=0, val=1, count=0)

        with self.assertRaisesRegex(
            RuntimeError, r"offset \+ count.*exceeded the numel of the input"
        ):
            _SymmetricMemory.memset32(t, offset=64, val=1, count=1)

        with self.assertRaisesRegex(
            RuntimeError, r"offset \+ count.*exceeded the numel of the input"
        ):
            _SymmetricMemory.memset32(t, offset=0, val=1, count=65)

        _SymmetricMemory.memset32(t, offset=0, val=1, count=64)
        _SymmetricMemory.memset32(t, offset=63, val=1, count=1)


if __name__ == "__main__":
    run_tests()
