# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch
from torch._dynamo.device_interface import DeviceInterface, register_interface_for_device, \
    caching_worker_current_devices, caching_worker_device_properties, get_interface_for_device
from torch._dynamo.variables.torch import TorchInGraphFunctionVariable
from torch_supa._C import _supa_getCurrentRawStream as get_supa_stream
from ..supa.streams import Event, Stream
from ..supa import current_device, set_device, device_count, stream, current_stream, \
    set_stream, set_stream_by_id, synchronize, get_device_capability
from ..supa import get_device_properties as get_device_properties_supa

class SupaInterface(DeviceInterface):
    device = torch.device
    Event = Event
    Stream = Stream

    class Worker:
        @staticmethod
        def set_device(device: int):
            caching_worker_current_devices["supa"] = device

        @staticmethod
        def current_device() -> int:
            if "supa" in caching_worker_current_devices:
                return caching_worker_current_devices["supa"]
            return current_device()

        @staticmethod
        def get_device_properties(device=None):
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    if device.type != torch.device("supa").type:
                        raise AssertionError('device.type should be equal to supa.')
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = SupaInterface.Worker.current_device()

            if "supa" not in caching_worker_device_properties:
                device_prop = [
                    get_device_properties_supa(i)
                    for i in range(device_count())
                ]
                caching_worker_device_properties["supa"] = device_prop

            return caching_worker_device_properties["supa"][device]

    current_device = staticmethod(current_device)
    set_device = staticmethod(set_device)
    device_count = staticmethod(device_count)
    stream = staticmethod(stream)
    current_stream = staticmethod(current_stream)
    set_stream = staticmethod(set_stream)
    synchronize = staticmethod(synchronize)
    get_device_properties = staticmethod(get_device_properties_supa)
    get_raw_stream = staticmethod(get_supa_stream)
    _set_stream_by_id = staticmethod(set_stream_by_id)

    @staticmethod
    def is_available() -> bool:
        return device_count() > 0

    @staticmethod
    def get_compute_capability(device=None):
        major, minor = get_device_capability(device)
        return major * 10 + minor

    @staticmethod
    def exchange_device(device: int) -> int:
        curr_device = current_device()
        set_device(device)
        return curr_device

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        return device

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False):
        return True


def _dynamo_register_interface_for_device():
    def _register_device_interface(device):
        register_interface_for_device(device, SupaInterface)
        for i in range(torch.supa.device_count()):
            register_interface_for_device(f"{device}:{i}", SupaInterface)

    _register_device_interface("supa")
    get_interface_for_device("cuda")  # ensure device_interface._device_initialized is true.
    _register_device_interface("cuda")  # update device interface to supa for 'cuda'

    # SUPA is registered after Dynamo's stream handler table may be cached.
    TorchInGraphFunctionVariable._get_handlers.cache_clear()
