# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch
import torch.distributed.fsdp._fully_shard._fsdp_param  # noqa: F401


def patch_fully_shard_fsdp_param():
    patch_copy_()


def patch_copy_():
    lib = torch.library.Library("fsdp", "FRAGMENT")
    @torch.library.impl(lib, "copy_", "PrivateUse1")
    def copy_(tensor, data):
        tensor.copy_(data)

