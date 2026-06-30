# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import functools
import os
import sys
import traceback
from torch.torch_version import TorchVersion
import torch_supa


def transfer_device_type(func):
    """Decorator to temporarily set device_type to True during function execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        status = torch_supa._C._transfer.device_type_status()
        torch_supa._C._transfer.device_type(True)
        try:
            return func(*args, **kwargs)
        finally:
            torch_supa._C._transfer.device_type(status)

    return wrapper


def get_torch_version():
    try:
        import torch
        import re

        return re.split(r"^([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2}).*", torch.__version__)[1]
    except:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)


def torch_version_ge(major: int, minor: int, patch: int):
    return TorchVersion(get_torch_version()) >= (major, minor, patch)


def torch_version_gt(major: int, minor: int, patch: int):
    return TorchVersion(get_torch_version()) > (major, minor, patch)


def torch_version_le(major: int, minor: int, patch: int):
    return TorchVersion(get_torch_version()) <= (major, minor, patch)


def torch_version_lt(major: int, minor: int, patch: int):
    return TorchVersion(get_torch_version()) < (major, minor, patch)


def torch_version_eq(major: int, minor: int, patch: int):
    return TorchVersion(get_torch_version()) == (major, minor, patch)
