# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch


def patch_backend_registry():
    patch__tensorpipe_validate_devices()


def patch__tensorpipe_validate_devices():
    def _tensorpipe_validate_devices(devices, device_count):
        privateuse1_backend_name = torch._C._get_privateuse1_backend_name()
        return all(
            d.type == "cpu"
            or (d.type in ("cuda", privateuse1_backend_name) and 0 <= d.index < device_count)
            for d in devices
        )

    torch.distributed.rpc.backend_registry._tensorpipe_validate_devices = _tensorpipe_validate_devices
