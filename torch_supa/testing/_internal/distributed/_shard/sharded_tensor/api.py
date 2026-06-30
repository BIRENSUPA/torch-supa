# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch
import torch.distributed as dist
from torch._utils import _get_device_module
from torch.distributed._shard.sharded_tensor import ShardedTensor

from torch_supa.utils import torch_version_le


def patch_api():
    if torch_version_le(2, 6, 0):
        patch_ShardedTensor__get_preferred_device()


def patch_ShardedTensor__get_preferred_device():
    def _get_preferred_device(self) -> torch.device:
        """
        Return the preferred device to be used when creating tensors for collectives.
        This method takes into account the associated process group
        """
        backend = dist.get_backend(self._process_group)
        if backend == dist.Backend.NCCL:
            return torch.device(torch.cuda.current_device())
        elif backend == dist.Backend.GLOO:
            return torch.device("cpu")
        else:
            backend_config = dist.BackendConfig(backend)
            for device, backend_str in backend_config.get_device_backend_map().items():
                if backend_str == backend and device != "cpu":
                    return torch.device(
                        device, _get_device_module(device).current_device()
                    )
        return torch.device("cpu")

    ShardedTensor._get_preferred_device = _get_preferred_device
