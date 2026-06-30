# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.


import os


def _autoload():
    # Set the environment variable to true in this entrypoint
    os.environ["IS_CUSTOM_DEVICE_BACKEND_IMPORTED"] = "1"
