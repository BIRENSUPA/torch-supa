/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/DispatchKeyExtractor.h>
#include <c10/core/Storage.h>
#include <torch/library.h>

#include "torch_supa/csrc/core/supa/CachingHostAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/core/supa/WarningUtils.h"

namespace at::supa {
namespace native {

bool is_pinned(const at::Tensor& self, c10::optional<at::Device> device) {
  // Only CPU tensors can be pinned
  if (!self.is_cpu()) {
    return false;
  }
  c10::DispatchKeySet _dk = c10::DispatchKeySet(
      c10::computeDispatchKey(c10::nullopt, self.layout(), device.value_or(c10::DeviceType::PrivateUse1)));
  return at::_ops::is_pinned::redispatch(_dk, self, device);
}

at::Tensor _pin_memory(const at::Tensor& self, c10::optional<at::Device> device) {
  TORCH_CHECK(self.device().is_cpu(), "cannot pin '", self.toString(), "' only dense CPU tensors can be pinned");
  c10::DispatchKeySet _dk = c10::DispatchKeySet(
      c10::computeDispatchKey(c10::nullopt, self.layout(), device.value_or(c10::DeviceType::PrivateUse1)));
  return at::_ops::_pin_memory::redispatch(_dk, self, device);
}

WITH_IGNORE_WARNING_OVERRIDE_OPERATOR(pin_memory, true,
  TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
    m.impl(TORCH_SELECTIVE_NAME("aten::is_pinned"), TORCH_FN(is_pinned));
    m.impl(TORCH_SELECTIVE_NAME("aten::_pin_memory"), TORCH_FN(_pin_memory));
  })

} // namespace native
} // namespace at::supa
