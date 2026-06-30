# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# Copyright © 2023-2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
import torch_supa

all = ["is_available", "dump_bccl_trace_json", "dump_bccl_trace"]


def is_available() -> bool:
    """
    Returns ``True`` if the distributed package is available. Otherwise,
    ``torch.distributed`` does not expose any other APIs. Currently,
    ``torch.distributed`` is available on Linux, MacOS and Windows. Set
    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
    ``USE_DISTRIBUTED=0`` for MacOS.
    """
    return hasattr(torch_supa._C, "_c10d_supa_init")


if is_available():
    if not torch_supa._C._c10d_supa_init():
        raise RuntimeError("Failed to initialize torch_supa.distributed")

    from .distributed_c10d import *  # noqa
    from .fsdp import *  # noqa
    from .rpc import *  # noqa
    patch_distributed()
    patch_fsdp()
    patch_rpc()

else:
    print("bccl is not available")
