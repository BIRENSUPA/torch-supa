# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.

import os
import traceback
import threading
from functools import lru_cache
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
from torch import device as _device
from torch._utils import _get_device_index as _torch_get_device_index

import torch_supa
import torch_supa._C


def _dummy_type(name: str):
    def init_err(self):
        class_name = self.__class__.__name__
        raise RuntimeError(f"Tried to instantiate dummy base class {class_name}")

    return type(name, (object,), {"__init__": init_err})


if not hasattr(torch_supa._C, '_SUPAStreamBase'):
    torch_supa._C.__dict__['_SUPAStreamBase'] = _dummy_type('SUPAStreamBase')
    torch_supa._C.__dict__['_SUPAEventBase'] = _dummy_type('SUPAEventBase')

if TYPE_CHECKING:
    from torch.types import Device

try:
    from torch_supa._C import _supart  # type: ignore[attr-defined]
except ImportError:
    _supart = None


_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []  # don't invoke these until initialization occurs
_is_internal_in_bad_fork = False  # this global is also used in torch.manual_seed
_original_pid = False
_device_t = Union[_device, str, int, None]

def _is_in_bad_fork():
    return _is_internal_in_bad_fork


def is_initialized():
    r"""Returns whether PyTorch's SUPA state has been initialized."""
    return _initialized and not _is_internal_in_bad_fork

def is_bf16_supported(including_emulation: bool = True):
    r"""Return a bool indicating if the current SUPA device supports dtype bfloat16."""
    # If SUPA is not available, than it does not support bf16 either
    if not is_available():
        return False

    device = torch_supa.supa.current_device()

    if not including_emulation:
        return False

    # Finally try to create a bfloat16 device.
    return _check_bf16_tensor_supported(device)

@lru_cache(maxsize=16)
def _check_bf16_tensor_supported(device: _device_t):
    try:
        torch.tensor([1.0], dtype=torch.bfloat16, device=device)
        return True
    except Exception:
        return False

def _lazy_call(callable):
    if _initialized:
        callable()
    else:
        # Don't store the actual traceback to avoid memory cycle
        _queued_calls.append((callable, traceback.format_stack()))


class DeferredSupaCallError(Exception):
    pass


def init():
    r"""Initialize PyTorch's SUPA state.  You may need to call
    this explicitly if you are interacting with PyTorch via
    its C API, as Python bindings for SUPA functionality will not
    be until this initialization takes place.  Ordinary users
    should not need this, as all of PyTorch's SUPA methods
    automatically initialize SUPA state on-demand.

    Does nothing if the SUPA state is already initialized.
    """
    torch_supa.supa._lazy_init()


def _lazy_init():
    def _queue_call(queued_calls):
        for queued_call, orig_traceback in queued_calls:
            try:
                queued_call()
            except Exception as e:
                msg = (
                    f"SUPA call failed lazily at initialization with error: {str(e)}\n\n"
                    f"SUPA call was originally invoked at:\n\n{orig_traceback}"
                )
                raise DeferredSupaCallError(msg) from e

    global _initialized, _original_pid
    if _initialized or hasattr(_tls, "is_initializing"):
        return
    with _initialization_lock:
        # We be double-checked locking, boys!  This is OK because
        # the above test was GIL protected anyway.  The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if _initialized:
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if _is_internal_in_bad_fork:
            from sys import version_info

            if version_info < (3, 4):
                msg = "To use SUPA with multiprocessing, you must use Python " "3.4+ and the 'spawn' start method"
            else:
                msg = "To use SUPA with multiprocessing, you must use the " "'spawn' start method"
            raise RuntimeError("Cannot re-initialize SUPA in forked subprocess. " + msg)

        if _supart is None:
            raise AssertionError(
                "libsupart functions unavailable. It looks like you have a broken build?"
            )

        torch_supa._C._supa_init()

        _original_pid = os.getpid()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True
        try:
            _queue_call(_queued_calls)
        finally:
            delattr(_tls, "is_initializing")
        _initialized = True


def supart():
    r"""Retrieves the SUPA runtime API module.


    This function initializes the SUPA runtime environment if it is not already
    initialized and returns the SUPA runtime API module (_supart). The SUPA
    runtime API module provides access to various SUPA runtime functions.

    Args:
        ``None``

    Returns:
        module: The SUPA runtime API module (_supart).

    Raises:
        RuntimeError: If SUPA cannot be re-initialized in a forked subprocess.
        AssertionError: If PyTorch is not compiled with SUPA support or if libsupart functions are unavailable.

    Example of SUPA operations with profiling:
        >>> import torch
        >>> from torch.supa import supart, check_error
        >>> import os
        >>>
        >>> os.environ['SUPA_PROFILE'] = '1'
        >>>
        >>> def perform_supa_operations_with_streams():
        >>>     stream = torch.supa.Stream()
        >>>     with torch.supa.stream(stream):
        >>>         x = torch.randn(100, 100, device='supa')
        >>>         y = torch.randn(100, 100, device='supa')
        >>>         z = torch.mul(x, y)
        >>>     return z
        >>>
        >>> torch.supa.synchronize()
        >>> print("====== Start nsys profiling ======")
        >>> check_error(supart().supaProfilerStart())
        >>> with torch.autograd.profiler.emit_nvtx():
        >>>     result = perform_supa_operations_with_streams()
        >>>     print("SUPA operations completed.")
        >>> check_error(torch.supa.supart().supaProfilerStop())
        >>> print("====== End nsys profiling ======")

    To run this example and save the profiling information, execute:
        >>> $ suprof -O ./result supart_test.sh

    This command profiles the SUPA operations in the provided script and saves
    the profiling information to a file named `trace_name.prof`.
    The `--profile-from-start off` option ensures that profiling starts only
    after the `supaProfilerStart` call in the script.
    The `--csv` and `--print-summary` options format the profiling output as a
    CSV file and print a summary, respectively.
    The `-o` option specifies the output file name, and the `-f` option forces the
    overwrite of the output file if it already exists.
    """
    _lazy_init()
    return _supart

class SupaError(RuntimeError):
    def __init__(self, code: int) -> None:
        # pyrefly: ignore  # missing-attribute
        msg = _supart.supaGetErrorString(_supart.supaError(code))
        super().__init__(f"{msg} ({code})")


def check_error(res: int) -> None:
    r"""Raise an error if the result of a SUPA runtime API call is not success."""
    # pyrefly: ignore  # missing-attribute
    if res != _supart.supaError.success:
        raise SupaError(res)

def synchronize(device=None):
    r"""Waits for all kernels in all streams on a SUPA device to complete.

    Arguments:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch_supa.supa.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    torch_supa.supa._lazy_init()
    with torch_supa.supa.device(device):
        return torch_supa._C._supa_synchronize()


@lru_cache(maxsize=1)
def device_count():
    return torch_supa._C._supa_getDeviceCount()


def set_device(device):
    device_id = _get_device_index(device, optional=True)
    if device_id >= 0:
        torch_supa._C._supa_setDevice(device_id)


def current_device():
    torch_supa.supa._lazy_init()
    return torch_supa._C._supa_getDevice()


def get_device_name(device=None):
    return get_device_properties(device).name


def get_device_properties(device=None):
    device_id = _get_device_index(device, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id")
    torch_supa.supa._lazy_init()
    return torch_supa._C._supa_getDeviceProperties(device_id)


def can_device_access_peer(device: "Device", peer_device: "Device") -> bool:
    r"""Check if peer access between two devices is possible."""
    torch_supa.supa._lazy_init()
    device = _get_device_index(device, optional=True)
    peer_device = _get_device_index(peer_device)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    if peer_device < 0 or peer_device >= device_count():
        raise AssertionError("Invalid peer device id")
    return torch_supa._C._supa_canDeviceAccessPeer(device, peer_device)


def _get_device_index(device, optional=False, allow_cpu: bool = False):
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a SUPA device. Note that for a SUPA device without a specified index,
    i.e., ``torch.device('supa')``, this will return the current default SUPA
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default SUPA
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ["supa", "cpu"]:
                raise ValueError("Expected a supa or cpu device, but got: {}".format(device))
        elif device.type != torch.device("supa").type:
            raise ValueError("Expected a supa device, but got: {}".format(device))
    if not torch.jit.is_scripting():
        if isinstance(device, torch_supa.supa.device):
            return device.idx
    return _torch_get_device_index(device, optional, allow_cpu)


def get_device_capability(device=None):
    r"""Query the minor and major data of device. Cann does not
    have a corresponding concept and is not supported. By default, it returns None
    """
    prop = get_device_properties(device)
    return prop.major, prop.minor


def _sleep(cycles):
    torch_supa._C._supa_sleep(cycles)


class device(object):
    r"""Context-manager that changes the selected device.

    Arguments:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx == -1:
            return
        self.prev_idx = torch_supa._C._supa_getDevice()
        if self.prev_idx != self.idx:
            torch_supa._C._supa_setDevice(self.idx)
        torch_supa.supa._lazy_init()

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            torch_supa._C._supa_setDevice(self.prev_idx)
        return False

class StreamContext:
    r"""Context-manager that selects a given stream.

    All SUPA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """
    cur_stream: Optional["torch_supa.supa.Stream"]

    def __new__(cls, stream_ctx: Optional["torch_supa.supa.Stream"]):
        """
        Dynamo may inspect StreamContext before __init__ completes.
        """
        obj = super().__new__(cls)
        obj.stream = stream_ctx
        return obj

    def __init__(self, stream_ctx: Optional["torch_supa.supa.Stream"]):
        self.stream = stream_ctx
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                self.idx = -1

        self.src_prev_stream = (
            None if not torch.jit.is_scripting() else torch_supa.supa.default_stream()
        )
        self.dst_prev_stream = (
            None if not torch.jit.is_scripting() else torch_supa.supa.default_stream()
        )

    def __enter__(self):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or SUPA device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch_supa.supa.current_stream()

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = torch_supa.supa.current_stream(cur_stream.device)
        torch_supa.supa.set_stream(cur_stream)

    def __exit__(self, exec_type: Any, exec_value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no supa device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            torch_supa.supa.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        torch_supa.supa.set_stream(self.src_prev_stream)  # type: ignore[arg-type]


def stream(stream):
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.

    ..Note:: In eager mode stream is of type Stream class while in JIT it is
      an object of the custom class ``torch.classes.supa.Stream``.
    """
    return StreamContext(stream)

def set_stream(stream):
    r"""Sets the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.
    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    torch_supa._C._supa_setStream(stream_id=stream.stream_id,
                                  device_index=stream.device_index,
                                  device_type=stream.device_type)


def set_stream_by_id(stream_id, device_index, device_type):
    r"""set stream specified by the stream id, device index and
        device type

    Args: stream_id (int): stream id in stream pool
          device_index (int): device index in topo
          device_type (int): enum device type
    """
    if stream is None:
        return
    torch_supa._C._supa_setStream(stream_id=stream_id, device_index=device_index, device_type=device_type)


def current_stream(device=None):
    r"""Returns the currently selected :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch_supa.supa.current_device`, if :attr:`device` is ``None``
            (default).
    """
    torch_supa.supa._lazy_init()
    streamdata = torch_supa._C._supa_getCurrentStream(
        _get_device_index(device, optional=True))
    return torch_supa.supa.Stream(stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2])


def default_stream(device=None):
    r"""Returns the default :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch_supa.supa.current_device`, if :attr:`device` is ``None``
            (default).
    """
    torch_supa.supa._lazy_init()
    streamdata = torch_supa._C._supa_getDefaultStream(
        _get_device_index(device, optional=True))
    return torch_supa.supa.Stream(stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2])


def current_blas_handle():
    r"""Return cublasHandle_t pointer to current cuBLAS handle"""
    torch_supa.supa._lazy_init()
    return torch_supa._C._supa_getCurrentBlasHandle()


def is_available():
    if not hasattr(torch_supa._C, "_supa_setDevice"):
        return False
    return device_count() > 0


def _dummy_type(name: str) -> type:
    def get_err_fn(is_init: bool):
        def err_fn(obj, *args, **kwargs):
            if is_init:
                class_name = obj.__class__.__name__
            else:
                class_name = obj.__name__
            raise RuntimeError(f"Tried to instantiate dummy base class {class_name}")

        return err_fn

    return type(
        name, (object,), {"__init__": get_err_fn(True), "__new__": get_err_fn(False)}
    )
