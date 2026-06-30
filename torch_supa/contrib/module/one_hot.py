# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import functools
import os

import torch
import torch.nn.functional as F


_ORIGINAL_ONE_HOT = F.one_hot


def _is_supa_tensor(tensor):
    return isinstance(tensor, torch.Tensor) and tensor.device.type == "supa"


def _one_hot_fused_enabled():
    return os.getenv("BRTB_ENABLE_NATIVE_OP", "0").lower() not in ("1", "true", "on", "yes")


def _can_use_one_hot_fused(tensor, num_classes):
    return (
        _one_hot_fused_enabled()
        and _is_supa_tensor(tensor)
        and tensor.dtype == torch.int64
        and isinstance(num_classes, int)
        and num_classes > 0
        and tensor.is_contiguous()
    )


@functools.wraps(_ORIGINAL_ONE_HOT)
def _one_hot_patch(tensor, num_classes=-1):
    if _can_use_one_hot_fused(tensor, num_classes):
        return torch.ops.custom.one_hot_fused(tensor, num_classes)
    return _ORIGINAL_ONE_HOT(tensor, num_classes)


def _apply_one_hot_patch():
    F.one_hot = _one_hot_patch


_apply_one_hot_patch()
