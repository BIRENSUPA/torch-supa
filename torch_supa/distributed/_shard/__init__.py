# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from .sharded_tensor import patch_sharded_tensor

def patch__shard():
    patch_sharded_tensor()
