# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from ._runtime_utils import patch_fsdp__runtime_utils
from ._flat_param import patch_fsdp__flat_param
from ._fully_shard import patch_fully_shard


def patch_fsdp():
    patch_fsdp__runtime_utils()
    patch_fsdp__flat_param()
    patch_fully_shard()
