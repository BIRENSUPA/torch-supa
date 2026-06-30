# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch_supa.testing._internal.distributed  # noqa

from .common_fsdp import patch_common_fsdp
from .common_distributed import patch_common_distributed
from .common_utils import patch_common_utils


patch_common_fsdp()
patch_common_distributed()
patch_common_utils()
