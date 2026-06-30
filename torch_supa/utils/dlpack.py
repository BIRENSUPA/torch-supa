# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch
import torch.utils.dlpack as torch_dlpack
from torch_supa._C import _supa_from_dlpack, _supa_to_dlpack
from .utils import transfer_device_type, torch_version_ge
from torch.utils.dlpack import DLDeviceType

def _to_dlpack(tensor):
    return _supa_to_dlpack(tensor)


@transfer_device_type
def _from_dlpack(ext_tensor) -> 'torch.Tensor':
    if hasattr(ext_tensor, '__dlpack__'):
        device = ext_tensor.__dlpack_device__()
        # device is either CUDA or ROCm, we need to pass the current
        # stream

        if torch_version_ge(2, 9, 0):
            gpu_type = DLDeviceType.kDLCUDA
        else:
            gpu_type = DLDeviceType.kDLGPU

        if device[0] in (gpu_type, DLDeviceType.kDLROCM):
            stream = torch.cuda.current_stream(f'cuda:{device[1]}')
            # cuda_stream is the pointer to the stream and it is a public
            # attribute, but it is not documented
            # The array API specify that the default legacy stream must be passed
            # with a value of 1 for CUDA
            # https://data-apis.org/array-api/latest/API_specification/array_object.html?dlpack-self-stream-none#dlpack-self-stream-none
            is_cuda = device[0] == gpu_type
            # Since pytorch is not using PTDS by default, lets directly pass
            # the legacy stream
            stream_ptr = 1 if is_cuda and stream.cuda_stream == 0 else stream.cuda_stream
            dlpack = ext_tensor.__dlpack__(stream=stream_ptr)
        else:
            dlpack = ext_tensor.__dlpack__()
    else:
        # Old versions just call the converter
        dlpack = ext_tensor
    return _supa_from_dlpack(dlpack)


def _apply_dlpack_patch():
    """Patch torch.utils.dlpack and torch.utils to use torch_supa implementation for SUPA tensors"""
    # Store original functions
    _original_to_dlpack = torch_dlpack.to_dlpack
    _original_from_dlpack = torch_dlpack.from_dlpack

    def create_patched_to_dlpack(module_name):
        """Create a patched to_dlpack function with proper __module__ attribute"""

        def patched_to_dlpack(tensor):
            """Patched to_dlpack that uses torch_supa implementation for SUPA tensors"""
            if hasattr(tensor, "device") and tensor.device.type in ("supa", "cuda"):
                return _to_dlpack(tensor)
            return _original_to_dlpack(tensor)

        patched_to_dlpack.__module__ = module_name
        return patched_to_dlpack

    def create_patched_from_dlpack(module_name):
        """Create a patched from_dlpack function with proper __module__ attribute"""

        def patched_from_dlpack(ext_tensor):
            """Patched from_dlpack that uses torch_supa implementation when appropriate"""
            # For SUPA tensors or when torch_supa is available, use our implementation
            try:
                return _from_dlpack(ext_tensor)
            except Exception:
                # Fallback to original implementation
                return _original_from_dlpack(ext_tensor)

        patched_from_dlpack.__module__ = module_name
        return patched_from_dlpack

    # Apply patches to torch.utils.dlpack
    torch_dlpack.to_dlpack = create_patched_to_dlpack("torch.utils.dlpack")
    torch_dlpack.from_dlpack = create_patched_from_dlpack("torch.utils.dlpack")

    # Also patch torch.utils.to_dlpack and torch.utils.from_dlpack if they exist
    if hasattr(torch.utils, "to_dlpack"):
        _original_torch_utils_to_dlpack = torch.utils.to_dlpack
        torch.utils.to_dlpack = create_patched_to_dlpack("torch.utils")

    if hasattr(torch.utils, "from_dlpack"):
        _original_torch_utils_from_dlpack = torch.utils.from_dlpack
        torch.utils.from_dlpack = create_patched_from_dlpack("torch.utils")

    # Also patch torch.from_dlpack and torch.to_dlpack if they exist
    if hasattr(torch, "from_dlpack"):
        _original_torch_from_dlpack = torch.from_dlpack
        torch.from_dlpack = create_patched_from_dlpack("torch")

    if hasattr(torch, "to_dlpack"):
        _original_torch_to_dlpack = torch.to_dlpack
        torch.to_dlpack = create_patched_to_dlpack("torch")

        # Add to_dlpack to torch.__all__ if it exists, otherwise create it
        if not hasattr(torch, "__all__"):
            torch.__all__ = []
        if "to_dlpack" not in torch.__all__:
            torch.__all__.append("to_dlpack")

    # Also ensure from_dlpack is in torch.__all__ if it exists
    if hasattr(torch, "from_dlpack"):
        if not hasattr(torch, "__all__"):
            torch.__all__ = []
        if "from_dlpack" not in torch.__all__:
            torch.__all__.append("from_dlpack")
