# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from torch.nn.attention import flex_attention
from torch import Tensor

def patch_validate_device():
    # patch function _validate_device with validate_device, add logic to support supa
    def validate_device(query: Tensor, key: Tensor, value: Tensor):
        if query.device.type != "cuda" and query.device.type != "cpu" and query.device.type != "supa":
            raise ValueError(
                "FlexAttention is only supported on CUDA, CPU, or SUPA devices. "
                f"Found input tensors on {query.device.type} device."
            )

    flex_attention._validate_device = validate_device

patch_validate_device()
