# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.


import os
import torch
from typing import Optional
from collections import namedtuple


SUPA_RUNTIME_PATH = os.environ.get("SUPA_PATH", "/usr/local/birensupa/sdk/latest/supa")

def _get_supa_version() -> Optional[str]:
    with open(os.path.join(SUPA_RUNTIME_PATH, "version.txt")) as fi:
        return fi.read().strip()

def _add_collect_env_methods():
    torch.version.supa = _get_supa_version()

