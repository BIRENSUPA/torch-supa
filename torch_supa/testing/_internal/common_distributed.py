# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch
import torch.nn as nn
from torch.testing._internal.common_distributed import DistributedTestBase, SaveForwardInputsModule


def patch_common_distributed():
    patch_DistributedTestBase_backend()
    patch_DistributedTestBase_create_pg()
    patch_SaveForwardInputsModule___init__()


def patch_DistributedTestBase_backend():
    def backend(self, device) -> str:
        if "supa" in device:
            return "bccl"
        elif "cuda" in device:
            return "nccl"
        elif "hpu" in device:  # intel gaudi
            return "hccl"
        elif "xpu" in device:
            return "xccl"
        else:
            return "gloo"

    DistributedTestBase.backend = backend


def patch_DistributedTestBase_create_pg():
    def DistributedTestBase_create_pg(self, device, world_size=None):
        if world_size is None:
            world_size = self.world_size
        num_visible_devices = torch.get_device_module(device).device_count()
        store = torch.distributed.FileStore(self.file_name, num_visible_devices)
        torch.distributed.init_process_group(
            backend=self.backend(device),
            world_size=world_size,
            rank=self.rank,
            store=store,
        )
        if "nccl" in self.backend(device) or "xccl" in self.backend(device) or "bccl" in self.backend(device):
            torch.accelerator.set_device_index(self.rank)
        return torch.distributed.distributed_c10d._get_default_group()

    DistributedTestBase.create_pg = DistributedTestBase_create_pg


def patch_SaveForwardInputsModule___init__():
    def SaveForwardInputsModule___init__(
        self,
        forward_inputs: dict[nn.Module, torch.Tensor],
        cast_forward_inputs: bool,
    ) -> None:
        nn.Module.__init__(self)
        self.l = nn.Linear(4, 4)
        self.forward_inputs = forward_inputs
        self.cast_forward_inputs = cast_forward_inputs

    SaveForwardInputsModule.__init__ = SaveForwardInputsModule___init__
