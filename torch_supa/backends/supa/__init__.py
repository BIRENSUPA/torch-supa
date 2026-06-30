# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import os
from typing import Union

import torch

__all__ = [
    "is_built",
    "preferred_blas_library"
]

def is_built():
    r"""
    Return whether PyTorch is built with SUPA support.

    Note that this doesn't necessarily mean SUPA is available; just that if this PyTorch
    binary were run on a machine with working SUPA drivers and devices, we would be able to use it.
    """
    return True


_BlasBackends = {
    "cublas": "Sublas",
    "cublaslt": "Sublaslt",
    "sublas": "Sublas",  # alias
    "sublaslt": "Sublaslt",
}
_BlasBackends_str = ", ".join(_BlasBackends.keys())


def preferred_blas_library(
    backend: Union[None, str] = None,
):
    r"""
    Override the library PyTorch uses for BLAS operations. Choose between cuBLAS, cuBLASLt, and CK [ROCm-only].

    .. warning:: This flag is experimental and subject to change.

    When PyTorch runs a CUDA BLAS operation it defaults to cuBLAS even if both cuBLAS and cuBLASLt are available.
    For PyTorch built for ROCm, hipBLAS, hipBLASLt, and CK may offer different performance.
    This flag (a :class:`str`) allows overriding which BLAS library to use.

    * If `"cublas"` is set then cuBLAS will be used wherever possible.
    * If `"cublaslt"` is set then cuBLASLt will be used wherever possible.
    * If `"ck"` is set then CK will be used wherever possible.
    * If `"default"` (the default) is set then heuristics will be used to pick between the other options.
    * When no input is given, this function returns the currently preferred library.
    * User may use the environment variable TORCH_BLAS_PREFER_CUBLASLT=1 to set the preferred library to cuBLASLt
      globally.
      This flag only sets the initial value of the preferred library and the preferred library
      may still be overridden by this function call later in your script.

    Note: When a library is preferred other libraries may still be used if the preferred library
    doesn't implement the operation(s) called.
    This flag may achieve better performance if PyTorch's library selection is incorrect
    for your application's inputs.

    """
    if backend is None:
        pass
    elif isinstance(backend, str):
        if backend not in _BlasBackends:
            raise RuntimeError(
                f"Unknown input value. Choose from: {_BlasBackends_str}."
            )
        os.environ["BRTB_SUBLAS_PREFERRED_BACKEND"] = _BlasBackends[backend]
    else:
        raise RuntimeError("Unknown input value type.")
