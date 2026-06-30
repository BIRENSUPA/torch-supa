# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import os
import sys

import torch
import torch.distributed as dist
import torch.testing._internal.distributed._tensor.common_dtensor as common_dtensor
from torch.testing._internal.common_distributed import MultiProcessTestCase, TEST_SKIPS
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase
import torch_supa.utils.utils as supa_utils


def patch_common_dtensor():
    patch_DTensorTestBase_setUp()
    patch_DTensorTestBase_init_pg()
    patch_DTensorTestBase_backend()
    patch_DTensorTestBase_device_type()


def patch_DTensorTestBase_setUp():
    def setUp(self) -> None:
        MultiProcessTestCase.setUp(self)

        os.environ["DISTRIBUTED_TESTS_DEFAULT_TIMEOUT"] = "3600000"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "36666"
        os.environ["BCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["BCCL_TRACE_BUFFER_SIZE"] = "1024"

        os.environ["NCCL_TOPO_FILE"] = os.getenv("BIREN_HOME", "/usr/local/birensupa/all/latest") + "/bccl/xml/topo_2c.xml"

        self._spawn_processes()

    DTensorTestBase.setUp = setUp


def patch_DTensorTestBase_backend():
    def backend(self) -> str:
        return "bccl" if self.device_type == "supa" else "gloo"

    DTensorTestBase.backend = property(backend)


def patch_DTensorTestBase_device_type():
    common_dtensor.DEVICE_TYPE = "supa"
    DTensorTestBase.device_type = "supa"


def patch_DTensorTestBase_init_pg():
    def init_pg(self, eager_init, *args, **kwargs) -> None:
        backend = self.backend
        if supa_utils.torch_version_ge(2, 9, 0):
            backend = kwargs.pop("backend", None)
            if args:
                backend = args[0]
            if backend is None:
                backend = self.backend

        if ("nccl" in backend or "bccl" in backend) and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        if backend not in [
            "nccl",
            "gloo",
            "mpi",
            "cpu:gloo,cuda:nccl",
            "hccl",
            "xccl",
            "bccl",
            "fake",
        ]:
            raise RuntimeError(f"Backend {backend} not supported!")

        device_id = None
        if "nccl" in backend or "xccl" in backend or "bccl" in backend:
            # set device for nccl pg for collectives
            torch.accelerator.set_device_index(self.rank)
            # we only need to set device_id for nccl backend with eager init
            device_id = (
                torch.device(f"{self.device_type}:{self.rank}") if eager_init else None
            )
        # For nccl backend, bind the device to the process if device_id is not None
        # so the nccl communicator is immediately formed and we can use `ncclCommSplit`
        # for form subgroup to avoid unnecesssary overhead.
        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,  # pyre-ignore[16]
            init_method=f"file://{self.file_name}",  # pyre-ignore[16]
            device_id=device_id,
        )

    DTensorTestBase.init_pg = init_pg
