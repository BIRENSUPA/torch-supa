# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from torch._inductor.codegen.common import DeviceOpOverrides, register_device_op_overrides, get_device_op_overrides


class SUPADeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        return f"from torch_supa._C import _supa_getCurrentRawStream as {name}"

    def set_device(self, device_idx):
        return f"torch.supa.set_device({device_idx})"

    def synchronize(self):
        return "torch.supa.synchronize()"

    def device_guard(self, device_idx):
        return f"torch.supa._DeviceGuard({device_idx})"


def _inductor_register_device_op_overrides():
    from torch._inductor.codegen import cpu_device_op_overrides  # noqa F401
    register_device_op_overrides("supa", SUPADeviceOpOverrides())
    get_device_op_overrides("supa")  # after torch v2.12, ensure _device_op_overrides_initialized is true.
    register_device_op_overrides("cuda", SUPADeviceOpOverrides())
