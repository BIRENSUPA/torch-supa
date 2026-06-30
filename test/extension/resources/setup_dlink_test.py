# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import os
import sys

from setuptools import setup

from torch_supa.utils.cpp_extension import (
    SupaExtension,
    BuildExtension,
)


CXX_FLAGS = ["-g"]
USE_NINJA = True

pwd = os.getcwd()

ext_modules = [
    SupaExtension(
        "torch_test_dlink.supa",
        [
            "vector_add_kernel.su",
            "vector_add.cpp",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "brcc": ["-O2", "-fgpu-rdc"],
            "brcc_dlink": ["-fgpu-rdc"],
        },
    )
]

usage = """
unit test on brcc_dlink feature.
usage:
1. build: python3 setup_dlink_test.py bdist_wheel
2. install: pip install dist/torch_test_dlink-0.1.0-cp310-cp310-linux_x86_64.whl
3. check: python3 -c "import torch_test_dlink.supa; help(torch_test_dlink.supa)"

"""
print(sys.argv)

if __name__ == "__main__":
    setup(
        name="torch_test_dlink",
        description=usage,
        version="0.1.0",
        packages=[],
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
    )
