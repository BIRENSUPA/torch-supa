# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch


def patch_common_utils():
    patch_get_cycles_per_ms()


def patch_get_cycles_per_ms():
    def get_cycles_per_ms() -> float:
        return 2.4

    torch.testing._internal.common_utils.get_cycles_per_ms = get_cycles_per_ms
