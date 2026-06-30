# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from typing import Any

try:
    import triton
except ImportError:
    triton = None

if triton is not None:
    try:
        from triton.runtime.cache import triton_key  # type: ignore[attr-defined]
    except ImportError:
        from triton.compiler.compiler import (
            triton_key,  # type: ignore[attr-defined,no-redef]
        )

    HAS_TRITON = True
else:
    def _raise_error(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("triton package is not installed")

    triton_key = _raise_error
    HAS_TRITON = False
