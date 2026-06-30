# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

__all__ = []

import torch
import torch_supa
from . import serialization as se

def _rebuild_supa_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks, supa_storage_info):
    warn_massages = (
        "Warning: The current version of the file storing weights is old,"
        "and in the future we will deprecate the loading support for this type of file,"
        "please use 2.1 and newer torch to re-store the weight file."
    )
    se._warn_legacy_serialization(warn_massages, "oldfile")
    tensor = torch.empty(
        (0,),
        dtype=storage.dtype,
        device=storage._untyped_storage.device,
        requires_grad=requires_grad,
    )
    tensor.set_(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    tensor._backward_hooks = backward_hooks
    target_device = torch.device("cpu") if se.RE_MAP_CPU else torch.device("supa")
    is_fake_mode = (
        hasattr(torch, "_guards")
        and torch._guards.detect_fake_mode(None) is not None
    )

    if is_fake_mode:
        tensor.fake_device = target_device
    elif not se.RE_MAP_CPU:
        if isinstance(supa_storage_info, bool):
            tensor = tensor.supa()
        else:
            tensor = torch_supa.supa_format_cast(tensor.supa(), supa_storage_info)
    return tensor

def _share_supa_(self, *args, **kwargs):
    return torch_supa._C._share_supa_(self, *args, **kwargs)

def _typed_storage_share_supa_(self, *args, **kwargs):
    return self._untyped_storage._share_supa_(*args, **kwargs)

def _new_shared_supa(*args, **kwargs):
    return torch_supa._C._new_shared_supa(*args, **kwargs)

def _typed_storage_new_shared_supa(*args, **kwargs):
    return torch.UntypedStorage._new_shared_supa(*args, **kwargs)

def _release_ipc_counter_supa(*args, **kwargs):
    return torch_supa._C._release_ipc_counter_supa(*args, **kwargs)

def _typed_storage_release_ipc_counter_supa(*args, **kwargs):
    return torch.UntypedStorage._release_ipc_counter_supa(*args, **kwargs)

def _add_storage_methods():
    setattr(torch.UntypedStorage, "_share_supa_", _share_supa_)
    setattr(torch.UntypedStorage, "_new_shared_supa", _new_shared_supa)
    setattr(torch.UntypedStorage, "_release_ipc_counter_supa", _release_ipc_counter_supa)
    setattr(torch.TypedStorage, "_share_supa_", _typed_storage_share_supa_)
    setattr(torch.TypedStorage, "_new_shared_supa", _typed_storage_new_shared_supa)
    setattr(torch.TypedStorage, "_release_ipc_counter_supa", _typed_storage_release_ipc_counter_supa)
