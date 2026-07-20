# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import os
import time
import json
import pytest

import torch
import torch.distributed as c10d
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import skip_but_pass_in_sandcastle_if

import torch_supa  # noqa
from torch_supa.distributed.distributed_c10d import dump_bccl_trace_json


os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "36666"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1024"

os.environ["NCCL_TOPO_FILE"] = os.getenv("BIREN_HOME", "/usr/local/birensupa/all/latest") + "/bccl/xml/topo_2c.xml"


def requires_bccl():
    return skip_but_pass_in_sandcastle_if(
        not torch_supa.distributed.distributed_c10d._BCCL_AVAILABLE,
        "c10d was not compiled with the bccl backend",
    )

def requires_cards(count):
    return skip_but_pass_in_sandcastle_if(
        torch_supa.supa.device_count() < count,
        f"needs {count} cards at least, but found {torch_supa.supa.device_count()} cards. check env configuration."
    )

@requires_bccl()
@requires_cards(2)
# @pytest.mark.multicard
@pytest.mark.pt20600
@pytest.mark.sanity
@pytest.mark.regression
class TestC10d(MultiProcessTestCase):

    def setUp(self):
        super(TestC10d, self).setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    @staticmethod
    def setup_c10d(func):
        # can't use setup_method/teardown_method since test_function is called directly by MultiProcessTestCase..
        # so use decorator here.
        def wrapper(self, *args, **kwargs):
            self.device = torch.device(f"supa:{self.rank}")
            torch.supa.set_device(self.rank)
            c10d.init_process_group("bccl", world_size=self.world_size, rank=self.rank)
            self.pg = c10d.distributed_c10d._get_default_group()

            result = func(self, *args, **kwargs)

            c10d.destroy_process_group()
            return result
        return wrapper

    @setup_c10d
    def test_bccl_collective(self) -> None:
        shape = (1024, 1024)

        def for_each_dtype(func):
            def inner():
                for dtype in (torch.int32, torch.float32):
                    print(f"[rank {self.rank}] {func.__name__} with {dtype}..")
                    func(dtype)
                    print(f"[rank {self.rank}] {func.__name__} with {dtype}... \x1b[32mPASSED\x1b[0m")

            return inner

        @for_each_dtype
        def broadcast(dtype):
            if self.rank == 0:
                a = torch.ones(shape, dtype=dtype, device=self.device)
            else:
                a = torch.zeros(shape, dtype=dtype, device=self.device)
            c10d.broadcast(a, 0)

            if self.rank == 1:
                self.assertEqual(a.cpu(), torch.ones(shape, dtype=dtype, device="cpu"), msg="broadcast failed")
            c10d.barrier()

        @for_each_dtype
        def reduce(dtype):
            if self.rank == 0:
                a = torch.ones(shape, dtype=dtype, device=self.device)
            else:
                a = torch.ones(shape, dtype=dtype, device=self.device)
            c10d.reduce(a, 1)

            if self.rank == 1:
                self.assertEqual(a.cpu(), torch.ones(shape, dtype=dtype, device="cpu") * 2, msg="reduce failed")

        @for_each_dtype
        def all_reduce(dtype):
            if self.rank == 0:
                a = torch.ones(shape, dtype=dtype, device=self.device)
            else:
                a = torch.ones(shape, dtype=dtype, device=self.device)
            c10d.all_reduce(a)

            self.assertEqual(a.cpu(), torch.ones(shape, dtype=dtype, device="cpu") * 2, msg="all_reduce failed")

        @for_each_dtype
        def gather(dtype):
            local_tensor = torch.ones(16, device=self.device, dtype=dtype) * self.rank

            gathered_tensors = None
            if self.rank == 0:
                gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]

            c10d.gather(local_tensor, gathered_tensors, dst=0)
            if self.rank == 0:
                self.assertEqual(gathered_tensors[1].cpu(), torch.ones(16, dtype=dtype) * 1)

        @for_each_dtype
        def all_gather(dtype):
            local_tensor = torch.ones(16, device=self.device, dtype=dtype) * self.rank
            gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]

            c10d.all_gather(gathered_tensors, local_tensor)
            for i in range(self.world_size):
                self.assertEqual(gathered_tensors[i].cpu(), torch.ones(16, dtype=dtype) * i)

        @for_each_dtype
        def all_to_all(dtype):
            send_tensors = [torch.ones(shape, device=self.device, dtype=dtype) * self.rank,
                            torch.ones(shape, device=self.device, dtype=dtype) * self.rank]
            recv_tensors = [torch.zeros(shape, device=self.device, dtype=dtype),
                            torch.zeros(shape, device=self.device, dtype=dtype)]
            c10d.all_to_all(recv_tensors, send_tensors)
            self.assertEqual(recv_tensors[0].cpu(), torch.zeros(shape, dtype=dtype))
            self.assertEqual(recv_tensors[1].cpu(), torch.ones(shape, dtype=dtype))

        broadcast()
        reduce()
        all_reduce()
        gather()
        all_gather()
        all_to_all()

    @setup_c10d
    def test_tracing(self):
        shape = (1024, 1024)
        """ must set BCCL_TRACE_BUFFER_SIZE in order to enable tracing.   """
        self.pg._enable_collectives_timing()

        a = torch.full(shape, float(self.rank), device=self.device)
        for i in range(2):
            f = self.pg.allreduce(a)
        f.wait()
        torch_supa.supa.synchronize(device=self.device)
        # gah ok so now the duration_ms is populated best-effort since it can only happen outside "dump()" api
        time.sleep(1)

        t = json.loads(dump_bccl_trace_json(includeCollectives=True))
        self.assertEqual(t["version"], "2.4")
        self.assertEqual(len(t["pg_config"]), 1)
        default_pg_info = t["pg_config"]["0"]
        self.assertIn("name", default_pg_info)
        self.assertIn("desc", default_pg_info)
        self.assertIn("ranks", default_pg_info)
        self.assertEqual(len(t["entries"]), 2)
        last = t["entries"][-1]
        self.assertEqual(last["record_id"], 1)
        self.assertEqual(last["input_sizes"], (shape,))
        self.assertEqual(last["input_dtypes"], ["Float"])
        self.assertEqual(last["output_sizes"], (shape,))
        self.assertEqual(last["output_dtypes"], ["Float"])
        self.assertEqual(last["collective_seq_id"], 2)
        self.assertEqual(last["timeout_ms"], 600000)


if __name__ == "__main__":
    pytest.main()
