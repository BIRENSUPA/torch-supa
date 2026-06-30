# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import os
import torch
from torch._inductor.codegen.common import register_backend_for_device, get_scheduling_for_device
from torch_supa.utils import torch_version_ge
from .codecache import patch_cache_base_get_system, patch_aot_code_compiler_compile
from .decomposition import _register_supa_inductor_decompositions, patch_opoverload_decompose_for_privateuse1
from .triton_heuristics import patch_triton_heuristic_device_check
from .codegen.wrapper import SUPAWrapperCodeGen
from .graph_utils import patch_graph_device_check
from .kernel import patch_tuned_kernel, patch_template_heuristic_device_type

def _inductor_register_backend_for_device():
    if get_scheduling_for_device("supa") is None:
        from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
        from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling

        if torch_version_ge(2, 4, 0):
            register_backend_for_device('supa', CUDACombinedScheduling, SUPAWrapperCodeGen, CppWrapperCpu)
            register_backend_for_device('cuda', CUDACombinedScheduling, SUPAWrapperCodeGen, CppWrapperCpu)
        else:
            register_backend_for_device('supa', CUDACombinedScheduling, SUPAWrapperCodeGen)
            register_backend_for_device('cuda', CUDACombinedScheduling, SUPAWrapperCodeGen)

_inductor_register_backend_for_device()

patch_cache_base_get_system()
# patch_aot_code_compiler_compile()

patch_graph_device_check()

patch_opoverload_decompose_for_privateuse1()
_register_supa_inductor_decompositions()

patch_tuned_kernel()
patch_template_heuristic_device_type()

# we do not support static supa launcher now, remove patch currently
# patch_triton_heuristic_device_check()
