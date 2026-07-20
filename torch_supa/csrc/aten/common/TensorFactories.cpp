/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/Context.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorFactories.h>
#include <c10/core/TensorOptions.h>

// register op
#include <torch/library.h>
#include "torch_supa/csrc/aten/common/EmptyTensor.h"
#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAGuard.h"

namespace at::supa {

at::Tensor SUPANativeFunctions::empty(
    IntArrayRef size,
    optional<ScalarType> dtype_opt,
    optional<Layout> layout_opt,
    optional<Device> device_opt,
    optional<bool> pin_memory_opt,
    optional<MemoryFormat> memory_format_opt) {
  at::Tensor result =
      at::detail::empty_supa(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
  // See Note [Enabling Deterministic Operations] in PyTorch TensorFactories.
  if (C10_UNLIKELY(
          at::globalContext().deterministicAlgorithms() &&
          at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_empty_deterministic_(result);
  }
  return result;
}

at::Tensor SUPANativeFunctions::empty_strided(
    IntArrayRef size,
    IntArrayRef stride,
    optional<ScalarType> dtype_opt,
    optional<Layout> layout_opt,
    optional<Device> device_opt,
    optional<bool> pin_memory_opt) {
  at::Tensor result = at::detail::empty_strided_supa(size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  // See Note [Enabling Deterministic Operations] in PyTorch TensorFactories.
  if (C10_UNLIKELY(
          at::globalContext().deterministicAlgorithms() &&
          at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_empty_deterministic_(result);
  }
  return result;
}

int64_t SUPANativeFunctions::_get_data_ptr(const at::Tensor& self) {
  return (int64_t)(self.data_ptr());
}

} // namespace at::supa
