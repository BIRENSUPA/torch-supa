# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch

from .storage import _rebuild_supa_tensor


def _add_serialization_methods():
    torch.serialization.add_safe_globals([_rebuild_supa_tensor])
