# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from typing import Callable

from torch._utils import CallbackRegistry

__all__ = [
    'register_callback_for_event_creation',
    'register_callback_for_event_deletion',
    'register_callback_for_event_record',
    'register_callback_for_event_wait',
    'register_callback_for_memory_allocation',
    'register_callback_for_memory_deallocation',
    'register_callback_for_stream_creation',
    'register_callback_for_device_synchronization',
    'register_callback_for_stream_synchronization',
    'register_callback_for_event_synchronization'
]

EventCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "SUPA event creation"
)
EventDeletionCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "SUPA event deletion"
)
EventRecordCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry(
    "SUPA event record"
)
EventWaitCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry("SUPA event wait")
MemoryAllocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "SUPA memory allocation"
)
MemoryDeallocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "SUPA memory deallocation"
)
StreamCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "SUPA stream creation"
)
DeviceSynchronizationCallbacks: "CallbackRegistry[[]]" = CallbackRegistry(
    "SUPA device synchronization"
)
StreamSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "SUPA stream synchronization"
)
EventSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "SUPA event synchronization"
)


def register_callback_for_event_creation(cb: Callable[[int], None]) -> None:
    EventCreationCallbacks.add_callback(cb)


def register_callback_for_event_deletion(cb: Callable[[int], None]) -> None:
    EventDeletionCallbacks.add_callback(cb)


def register_callback_for_event_record(cb: Callable[[int, int], None]) -> None:
    EventRecordCallbacks.add_callback(cb)


def register_callback_for_event_wait(cb: Callable[[int, int], None]) -> None:
    EventWaitCallbacks.add_callback(cb)


def register_callback_for_memory_allocation(cb: Callable[[int], None]) -> None:
    MemoryAllocationCallbacks.add_callback(cb)


def register_callback_for_memory_deallocation(cb: Callable[[int], None]) -> None:
    MemoryDeallocationCallbacks.add_callback(cb)


def register_callback_for_stream_creation(cb: Callable[[int], None]) -> None:
    StreamCreationCallbacks.add_callback(cb)


def register_callback_for_device_synchronization(cb: Callable[[], None]) -> None:
    DeviceSynchronizationCallbacks.add_callback(cb)


def register_callback_for_stream_synchronization(cb: Callable[[int], None]) -> None:
    StreamSynchronizationCallbacks.add_callback(cb)


def register_callback_for_event_synchronization(cb: Callable[[int], None]) -> None:
    EventSynchronizationCallbacks.add_callback(cb)
