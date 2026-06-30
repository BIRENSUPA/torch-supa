# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import sys
import os
import types

# Disable autoloading before running 'import torch' to avoid circular dependencies
ORG_AUTOLOAD = os.getenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"


import torch
from torch.utils.checkpoint import DefaultDeviceType
import torch_supa
import torch_supa.backends
import torch_supa.supa
import torch_supa.supa.amp
import torch_supa.contrib.module
from torch_supa import profiler
from torch_supa.utils import (
    _add_collect_env_methods,
    _inductor_register_device_op_overrides,
    _add_dynamo_methods,
    _apply_dlpack_patch,
    _add_serialization_methods,
    _add_storage_methods,
)
from torch_supa.utils.storage import _rebuild_supa_tensor
from torch_supa.multiprocessing.reductions import _add_reductions_methods
from torch_supa.utils._dynamo_device import _dynamo_register_interface_for_device

from .version import __supa_version__ as __supa_version__
from .version import __version__ as __version__

all_monkey_patches = [
    ["profiler", torch_supa.profiler.profiler],
    ["autograd.profiler", torch_supa.profiler.profiler.prof],
]

def _apply_patches(monkey_patches):

    def _getattr(module_list, root_module=torch):
        if len(module_list) <= 1:
            return root_module

        if hasattr(root_module, module_list[0]):
            return _getattr(module_list[1:], getattr(root_module, module_list[0]))
        else:
            empty_module_name = f"{root_module.__name__}.{module_list[0]}"
            sys.modules[empty_module_name] = types.ModuleType(empty_module_name)
            setattr(root_module, module_list[0], sys.modules.get(empty_module_name))
            return _getattr(module_list[1:], getattr(root_module, module_list[0]))

    for patch_pair in monkey_patches:
        dest, patch = patch_pair
        dest_module = _getattr(dest.split("."), root_module=torch)
        last_module_level = dest.split(".")[-1]
        if not isinstance(patch, types.ModuleType):
            setattr(dest_module, last_module_level, patch)
            continue

        if not hasattr(dest_module, last_module_level) or not hasattr(patch, "__all__"):
            setattr(dest_module, last_module_level, patch)
            sys.modules[f"{dest_module.__name__}.{last_module_level}"] = patch
            continue

        assert hasattr(patch, "__all__"), "Patch module must have __all__ definition."
        dest_module = getattr(dest_module, last_module_level)
        for attr in patch.__all__:
            setattr(dest_module, attr, getattr(patch, attr))

_apply_patches(all_monkey_patches)

# Bridge PrivateUse1 to SUPA.
torch.utils.rename_privateuse1_backend("supa")
# rename device name to 'supa' and register funcs
torch._register_device_module("supa", torch_supa.supa)

unsupported_dtype = [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True, for_module=True, for_storage=True, unsupported_dtype=unsupported_dtype
)

def _apply_patches(monkey_patches):
    def _getattr(module_list, root_module=torch):
        if len(module_list) <= 1:
            return root_module

        if hasattr(root_module, module_list[0]):
            return _getattr(module_list[1:], getattr(root_module, module_list[0]))
        else:
            empty_module_name = f"{root_module.__name__}.{module_list[0]}"
            sys.modules[empty_module_name] = types.ModuleType(empty_module_name)
            setattr(root_module, module_list[0], sys.modules.get(empty_module_name))
            return _getattr(module_list[1:], getattr(root_module, module_list[0]))

    for patch_pair in monkey_patches:
        dest, patch = patch_pair
        dest_module = _getattr(dest.split("."), root_module=torch)
        last_module_level = dest.split(".")[-1]
        if not isinstance(patch, types.ModuleType):
            setattr(dest_module, last_module_level, patch)
            continue

        if not hasattr(dest_module, last_module_level) or not hasattr(patch, "__all__"):
            setattr(dest_module, last_module_level, patch)
            sys.modules[f"{dest_module.__name__}.{last_module_level}"] = patch
            continue

        assert hasattr(patch, "__all__"), "Patch module must have __all__ definition."
        dest_module = getattr(dest_module, last_module_level)
        for attr in patch.__all__:
            setattr(dest_module, attr, getattr(patch, attr))


def _apply_distributed_patches():
    import torch_supa.distributed


def apply_class_patches():
    _apply_distributed_patches()
    _add_storage_methods()
    _add_serialization_methods()
    _add_collect_env_methods()
    _add_dynamo_methods()
    _apply_dlpack_patch()
    _add_reductions_methods()


torch_supa._C._initExtension()

apply_class_patches()

_inductor_register_device_op_overrides()

# register supa device interface for dynamo
_dynamo_register_interface_for_device()

# This function is an entrypoint called by PyTorch
# when running 'import torch'. There is no need to do anything.
def _autoload():
    import os
    os.environ["IS_CUSTOM_DEVICE_BACKEND_IMPORTED"] = "1"
    if os.getenv("TORCH_SUPA_EXT", "0") == "1":
        import torch_supa_ext
    from torch_supa.contrib import transfer_to_supa
    # We should restore this switch as sub processes need to inherit its value
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = ORG_AUTOLOAD
