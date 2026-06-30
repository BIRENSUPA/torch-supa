# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from torch_supa.utils import torch_version_ge
import torch

def patch_triton_heuristic_device_check():
    if torch_version_ge(2, 8, 0):
        from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult
        from torch._inductor.runtime.triton_heuristics import (
            CannotStaticallyLaunchKernel,
            HeuristicType,
            StaticallyLaunchedCudaKernel,
            triton_cache_dir,
            triton_hash_to_path_key,
        )
        import os
        from torch._inductor.runtime import triton_heuristics

        # Patch can_statically_launch to support "supa" device type
        original_can_statically_launch = StaticTritonCompileResult.can_statically_launch

        @staticmethod
        def patched_can_statically_launch(
            kernel,
            inductor_meta,
            triton_meta,
            heuristic_type,
        ):
            # Monkey-patch the check_can_launch function to allow "supa" device type
            if not torch._inductor.config.use_static_cuda_launcher:
                return None

            def check_can_launch():
                device_type = triton_meta.get("device_type", None)
                if device_type not in ("cuda", "supa"):
                    # Only cuda or supa kernels
                    raise CannotStaticallyLaunchKernel(f"Unsupported device type: {device_type}")

                if torch._inductor.config.cpp_wrapper:
                    raise CannotStaticallyLaunchKernel("Cpp wrapper enabled")

                if (
                    heuristic_type == HeuristicType.USER_AUTOTUNE
                    and not torch._inductor.config.static_launch_user_defined_triton_kernels
                ):
                    raise CannotStaticallyLaunchKernel("User defined triton kernel")

                if inductor_meta.get("store_cubin", None):
                    raise CannotStaticallyLaunchKernel("store_cubin is enabled")

                cubin_location = os.path.join(
                    triton_cache_dir(triton_meta.get("device", 0)),
                    triton_hash_to_path_key(kernel.hash),
                    f"{kernel.src.fn.__name__}.cubin",
                )

                if not os.path.exists(cubin_location):
                    raise CannotStaticallyLaunchKernel(
                        f"Cubin path not found: {cubin_location}"
                    )
                else:
                    kernel._cubin_path = cubin_location

                try:
                    static_kernel = StaticallyLaunchedCudaKernel(kernel)
                except NotImplementedError as e:
                    raise CannotStaticallyLaunchKernel(f"NotImplemented: {str(e)}") from e

                return static_kernel

            try:
                result = check_can_launch()
                return result
            except CannotStaticallyLaunchKernel as e:
                triton_heuristics.log.info("Bypassing StaticallyLaunchedCudaKernel due to %s", str(e))
                if torch._inductor.config.strict_static_cuda_launcher:
                    raise e
                return None

        StaticTritonCompileResult.can_statically_launch = patched_can_statically_launch
