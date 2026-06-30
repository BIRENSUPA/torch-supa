# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import os
import sys

from setuptools import setup

from torch_supa.utils.cpp_extension import CppExtension, BuildExtension, SupaExtension, SudaExtension


CXX_FLAGS = ["-g"]
USE_NINJA = os.getenv("USE_NINJA") == "1"

ext_modules = [
    CppExtension(
        "torch_test_cpp_extension.cpp", ["extension.cpp"], extra_compile_args=CXX_FLAGS, define_macros=[("DEBUG", None)]
    ),
    SupaExtension(
        "torch_test_cpp_extension.supa",
        [
            "vector_add.cpp",
            "vector_add_kernel.su",
        ],
        extra_compile_args={"cxx": CXX_FLAGS, "brcc": ["-O2"]},
    ),
    SudaExtension(
        "torch_test_cpp_extension.suda",
        [
            "vector_add.cpp",
            "vector_add_kernel.cu",
        ],
        # -D_SUPA_CUDA_ is only for reuse code of vector_add.cpp.
        extra_compile_args={"cxx": ["-D_SUPA_CUDA_"], "nvcc": ["-O2"]},
        libraries=["torch_supa_op"],
    ),
]

usage = """
unit test on cpp_extension.
usage:
1. build: python3 setup.py bdist_wheel
2. install: pip install dist/torch_test_cpp_extension-0.1.0-cp310-cp310-linux_x86_64.whl
3. check: python3 -c "import torch, torch_test_cpp_extension.supa; help(torch_test_cpp_extension.supa)

"""
print(sys.argv)

setup(
    name="torch_test_cpp_extension",
    description=usage,
    version="0.1.0",
    packages=["torch_test_cpp_extension"],
    ext_modules=ext_modules,
    # include_dirs="",
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    entry_points={
        "torch.backends": [
            "device_backend = torch_test_cpp_extension:_autoload",
        ],
    },
)
