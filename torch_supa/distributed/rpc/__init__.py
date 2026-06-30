# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from .backend_registry import patch_backend_registry


def patch_rpc():
    patch_backend_registry()
