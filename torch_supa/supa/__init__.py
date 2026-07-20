# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.

__all__ = [
    # Typed storage and Tensor
    "BFloat16Storage",
    "BFloat16Tensor",
    "BoolStorage",
    "BoolTensor",
    "ByteStorage",
    "ByteTensor",
    "CharStorage",
    "CharTensor",
    "ComplexDoubleStorage",
    "ComplexFloatStorage",
    "DoubleStorage",
    "DoubleTensor",
    "FloatStorage",
    "FloatTensor",
    "HalfStorage",
    "HalfTensor",
    "IntStorage",
    "IntTensor",
    "LongStorage",
    "LongTensor",
    "ShortStorage",
    "ShortTensor",
    # keep the order
    "Event",
    "ExternalStream",
    "MemPool",
    "MemPoolContext",
    "SUPAGraph",
    "Stream",
    "StreamContext",
    "_is_compiled",
    "_is_in_bad_fork",
    "_lazy_call",
    "_lazy_init",
    "_sleep",
    "amp",
    "bccl",
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "caching_allocator_enable",
    "can_device_access_peer",
    "check_error",
    "current_blas_handle",
    "current_device",
    "current_stream",
    "SUPAPluggableAllocator",
    "change_current_allocator",
    "default_generators",
    "default_stream",
    "device",
    "device_count",
    "empty_cache",
    "ipc_collect",
    "get_allocator_backend",
    "get_amp_supported_dtype",
    "get_autocast_dtype",
    "get_device_name",
    "get_device_properties",
    "get_per_process_memory_fraction",
    "get_rng_state",
    "get_rng_state_all",
    "graph",
    "graph_pool_handle",
    "init",
    "initial_seed",
    "is_autocast_enabled",
    "is_available",
    "is_bf16_supported",
    "is_current_stream_capturing",
    "is_initialized",
    "make_graphed_callables",
    "manual_seed",
    "manual_seed_all",
    "max_memory_allocated",
    "max_memory_cached",
    "max_memory_reserved",
    "mem_get_info",
    "memory_allocated",
    "memory_cached",
    "memory_reserved",
    "memory_snapshot",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "memory_summary",
    "profiler",
    "random",
    "reset_accumulated_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "reset_peak_memory_stats",
    "seed",
    "seed_all",
    "set_autocast_dtype",
    "set_autocast_enabled",
    "set_device",
    "set_per_process_memory_fraction",
    "set_rng_state",
    "set_rng_state_all",
    "set_stream",
    "stream",
    "supart",
    "synchronize",
    "use_mem_pool",
    "brtx",
]

from typing import Any, Tuple, Union
from .random import *  # noqa: F403
import torch
from torch import device as _device
from torch._utils import classproperty

import torch_supa
from torch.storage import _LegacyStorage, _warn_typed_storage_removal

default_generators: Tuple[torch._C.Generator] = ()

from .streams import Event, ExternalStream, Stream
from .utils import (
    _is_in_bad_fork,
    _lazy_call,
    _lazy_init,
    can_device_access_peer,
    current_blas_handle,
    current_device,
    current_stream,
    default_stream,
    device,
    device_count,
    get_device_name,
    get_device_properties,
    init,
    is_initialized,
    is_bf16_supported,
    set_device,
    set_stream,
    set_stream_by_id,
    stream,
    StreamContext,
    synchronize,
    _get_device_index,
    get_device_capability,
    _sleep,
    supart,
    check_error
)
from .graphs import (
    SUPAGraph,
    graph,
    graph_pool_handle,
    is_current_stream_capturing,
    make_graphed_callables,
)

from . import profiler, brtx, bccl, _gpu_trace  # noqa

from .streams import Stream, Event, ExternalStream # noqa

def _is_compiled() -> bool:
    r"""Return true if compile with SUPA support."""
    return hasattr(torch_supa._C, "_supa_getDeviceCount")

def is_available():
    if not hasattr(torch_supa._C, "_supa_setDevice"):
        return False
    return device_count() > 0


def __getattr__(name: str) -> Any:
    if name == "_initialized":
        return is_initialized()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from .amp import *  # noqa: F403
from .memory import *  # noqa: F403

if hasattr(torch_supa._C, "_supa_exchangeDevice"):
    _exchange_device = torch_supa._C._supa_exchangeDevice
else:

    def _exchange_device(device: int) -> int:
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without SUPA support")


if hasattr(torch_supa._C, "_supa_maybeExchangeDevice"):
    _maybe_exchange_device = torch_supa._C._supa_maybeExchangeDevice  # noqa: F811
else:

    def _maybe_exchange_device(device: int) -> int:
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without SUPA support")


################################################################################
# Define Storage and Tensor classes
################################################################################

@staticmethod  # type: ignore[misc]
def _lazy_new(cls, *args, **kwargs):
    _lazy_init()
    # We may need to call lazy init again if we are a forked child
    # del _SupaBase.__new__
    return super(_SupaBase, cls).__new__(cls, *args, **kwargs)


class _DeviceGuard:
    def __init__(self, index: int):
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch_supa.supa._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = torch_supa.supa._maybe_exchange_device(self.prev_idx)
        return False


class _SupaBase:
    is_supa = True
    is_sparse = False

    def type(self, *args, **kwargs):
        # We could use a Protocol here to tell mypy that self has `get_device` method
        # but it is only available in the typing module on Python >= 3.8
        # or on typing_extensions module on Python >= 3.6
        with device(self.get_device()):  # type: ignore[attr-defined]
            return super().type(*args, **kwargs)  # type: ignore[misc]

    __new__ = _lazy_new

class _SupaLegacyStorage(_LegacyStorage):
    @classmethod
    def from_buffer(cls, *args, **kwargs):
        _warn_typed_storage_removal()
        raise RuntimeError("from_buffer: Not available for SUPA storage")

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs):
        raise RuntimeError("_new_with_weak_ptr: Not available for SUPA storage")

    @classmethod
    def _new_shared_filename(cls, manager, obj, size, *, device=None, dtype=None):
        raise RuntimeError("_new_shared_filename: Not available for SUPA storage")

class ByteStorage(_SupaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.uint8


class DoubleStorage(_SupaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.double


class FloatStorage(_SupaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.float


class HalfStorage(_SupaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.half


class LongStorage(_SupaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.long


class IntStorage(_SupaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int


class ShortStorage(_SupaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.short


class CharStorage(_SupaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int8


class BoolStorage(_SupaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bool


class BFloat16Storage(_SupaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bfloat16


class ComplexDoubleStorage(_SupaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cdouble


class ComplexFloatStorage(_SupaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cfloat

del _LegacyStorage
del _SupaLegacyStorage

torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(BFloat16Storage)
torch._storage_classes.add(ComplexDoubleStorage)
torch._storage_classes.add(ComplexFloatStorage)
