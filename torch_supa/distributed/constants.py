# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from datetime import timedelta
from typing import Optional

try:
    from torch_supa._C._distributed_c10d import _DEFAULT_PG_BCCL_TIMEOUT

    default_pg_bccl_timeout: Optional[timedelta] = _DEFAULT_PG_BCCL_TIMEOUT
except ImportError:
    # if C++ BCCL support is not compiled, we don't have access to the default bccl value.
    # if anyone is actually trying to use bccl in this state, it should error.
    default_pg_bccl_timeout = None
