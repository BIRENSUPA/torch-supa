/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once
#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/core/supa/SUPAContext.h"
#include <ATen/core/TensorBase.h>

namespace at::detail {

TORCH_SUPA_API TensorBase
empty_supa(IntArrayRef size, ScalarType dtype, c10::optional<Device> device_opt,
           c10::optional<c10::MemoryFormat> memory_format_opt);

TORCH_SUPA_API TensorBase
empty_supa(IntArrayRef size, c10::optional<ScalarType> dtype_opt,
           c10::optional<Layout> layout_opt, c10::optional<Device> device_opt,
           c10::optional<bool> pin_memory_opt,
           c10::optional<c10::MemoryFormat> memory_format_opt);

TORCH_SUPA_API TensorBase empty_supa(IntArrayRef size,
                                     const TensorOptions &options);

TORCH_SUPA_API TensorBase empty_strided_supa(IntArrayRef size,
                                             IntArrayRef stride,
                                             ScalarType dtype,
                                             c10::optional<Device> device_opt);

TORCH_SUPA_API TensorBase empty_strided_supa(
    IntArrayRef size, IntArrayRef stride, c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt, c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt);

TORCH_SUPA_API TensorBase empty_strided_supa(IntArrayRef size,
                                             IntArrayRef stride,
                                             const TensorOptions &options);

} // namespace at::detail