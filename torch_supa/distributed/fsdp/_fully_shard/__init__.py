# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from ._fsdp_param import patch_fully_shard_fsdp_param


def patch_fully_shard():
    patch_fully_shard_fsdp_param()
