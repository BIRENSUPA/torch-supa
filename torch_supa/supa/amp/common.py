# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# Copyright © 2025 Shanghai Biren Technology Co., Ltd. All rights reserved.

import torch_supa


def amp_definitely_not_available():
    return not torch_supa.supa.is_available()
