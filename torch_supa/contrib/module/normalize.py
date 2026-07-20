# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import functools
import os

import torch
import torch.nn.functional as F


_ORIGINAL_NORMALIZE = F.normalize


def _normalize_fused_enabled():
    return os.getenv("BRTB_ENABLE_NATIVE_OP", "0").lower() not in ("1", "true", "on", "yes")


def _can_use_normalize_fused(input, p, dim, eps, out):
    return (
        _normalize_fused_enabled()
        and input.is_supa
        and p == 2.0
        and dim == -1
        and out is None
        and input.dtype == torch.float32
        and input.dim() >= 1
        and input.is_contiguous()
        and isinstance(eps, float)
    )


@functools.wraps(_ORIGINAL_NORMALIZE)
def _normalize_patch(input, p=2.0, dim=1, eps=1e-12, out=None):
    if _can_use_normalize_fused(input, p, dim, eps, out):
        return torch.ops.custom.normalize_fused(input, eps)
    return _ORIGINAL_NORMALIZE(input, p=p, dim=dim, eps=eps, out=out)


def _apply_normalize_patch():
    F.normalize = _normalize_patch
    torch.normalize = _normalize_patch


_apply_normalize_patch()
