# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# mypy: allow-untyped-defs
import contextlib
import tempfile

import torch

from . import check_error, supart

__all__ = ["start", "stop", "profile"]

DEFAULT_FLAGS = [
    "gpustarttimestamp",
    "gpuendtimestamp",
    "gridsize3d",
    "threadblocksize",
    "streamid",
    "enableonstart 0",
    "conckerneltrace",
]

def start():
    r"""Starts supa profiler data collection.

    .. warning::
        Raises CudaError in case of it is unable to start the profiler.
    """
    check_error(supart().supaProfilerStart())


def stop():
    r"""Stops supa profiler data collection.

    .. warning::
        Raises CudaError in case of it is unable to stop the profiler.
    """
    check_error(supart().supaProfilerStop())


@contextlib.contextmanager
def profile():
    """
    Enable profiling.

    Context Manager to enabling profile collection by the active profiling tool from CUDA backend.
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> model = torch.nn.Linear(20, 30).supa()
        >>> inputs = torch.randn(128, 20).supa()
        >>> with torch_supa.supa.profiler.profile() as prof:
        ...     model(inputs)
    """
    try:
        start()
        yield
    finally:
        stop()