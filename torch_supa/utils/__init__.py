# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from ._inductor import _inductor_register_device_op_overrides
from .collect_env import _add_collect_env_methods
from ._dynamo import _add_dynamo_methods
from .dlpack import _apply_dlpack_patch
from .serialization import _add_serialization_methods
from .storage import _add_storage_methods
from .utils import (
    get_torch_version,
    torch_version_ge,
    torch_version_gt,
    torch_version_le,
    torch_version_lt,
    torch_version_eq,
    transfer_device_type,
)
