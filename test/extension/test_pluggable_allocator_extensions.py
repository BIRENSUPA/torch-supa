# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import os

import pytest
import torch

import torch_supa
from torch.testing._internal.common_utils import TestCase
from torch_supa.utils.cpp_extension import load


PYTORCH_INSTALL_PATH = os.path.dirname(os.path.realpath(torch.__file__))
PYTORCH_SUPA_INSTALL_PATH = os.path.dirname(os.path.realpath(torch_supa.__file__))
cwd = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.single
@pytest.mark.pt20600
class TestPluggableAllocator(TestCase):
    module = None
    module_dir = os.path.dirname(__file__)
    build_directory = os.path.join(module_dir, "build")
    os.makedirs(build_directory, exist_ok=True)

    @classmethod
    def setUpClass(cls):
        extra_ldflags = []
        extra_ldflags.append("-lc10")
        extra_ldflags.append(f"-L{PYTORCH_INSTALL_PATH}")
        extra_ldflags.append(f"-L{PYTORCH_SUPA_INSTALL_PATH}/lib -ltorch_supa")
        extra_include_paths = ["cpp_extensions"]
        extra_include_paths.append(os.path.join(PYTORCH_SUPA_INSTALL_PATH, "include"))
        extra_include_paths.append(os.path.join(PYTORCH_SUPA_INSTALL_PATH, "include"))
        for i_path in os.getenv("CMAKE_INCLUDE_PATH", "/usr/local/include").split(":"):
            extra_include_paths.append(i_path)
        cls.module = load(
            name="pluggable_allocator_extensions",
            sources=[f"{cwd}/resources/pluggable_allocator_extensions.cpp"],
            extra_include_paths=extra_include_paths,
            extra_cflags=["-g", "-O0"],
            extra_supa_cflags=["-g"],
            extra_ldflags=extra_ldflags,
            build_directory=cls.build_directory,
            verbose=True,
        )

    def test_pluggable_allocator(self):
        pool = torch_supa.supa.MemPool()
        # MemPool doesn't have an allocator by default
        self.assertEqual(pool.allocator, None)

        os_path = os.path.join(
            TestPluggableAllocator.build_directory, "pluggable_allocator_extensions.so"
        )
        # Load the allocator
        pluggableallocator = torch_supa.supa.memory.SUPAPluggableAllocator(
            os_path, "my_malloc", "my_free"
        )

        pool = torch_supa.supa.MemPool(pluggableallocator.allocator())
        # pool should point to the same allocator as the one passed into it
        self.assertEqual(pluggableallocator.allocator(), pool.allocator)

        # pool's use count should be 1 at this point as MemPool object
        # holds a reference
        self.assertEqual(pool.use_count(), 1)

        with torch_supa.supa.use_mem_pool(pool):
            nelem_1mb = 1024 * 1024 // 4
            out_0 = torch.randn(nelem_1mb, device="supa")

            # pool's use count should be 2 at this point as use_mem_pool
            # holds a reference
            self.assertEqual(pool.use_count(), 2)

        del out_0
        del pool
