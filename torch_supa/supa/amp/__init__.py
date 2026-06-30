# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

__all__ = [
    "get_amp_supported_dtype",
    "get_autocast_gpu_dtype",
    "autocast",
    "custom_fwd",
    "custom_bwd",
    "is_autocast_enabled",
    "set_autocast_enabled",
    "get_autocast_dtype",
    "set_autocast_dtype",
]

from .autocast_mode import get_amp_supported_dtype, autocast, custom_fwd, custom_bwd, get_autocast_gpu_dtype, \
                           is_autocast_enabled, set_autocast_enabled, get_autocast_dtype, set_autocast_dtype # noqa
from .grad_scaler import GradScaler # noqa
