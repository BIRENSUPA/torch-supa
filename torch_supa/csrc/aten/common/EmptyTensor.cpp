/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/aten/common/EmptyTensor.h"
#include <ATen/EmptyTensor.h>
#include "torch_supa/csrc/core/supa/SUPAContext.h"
#include "torch_supa/csrc/core/supa/SUPAGuard.h"

namespace at::detail {

TensorBase empty_supa(IntArrayRef size, ScalarType dtype,
                      c10::optional<Device> device_opt,
                      c10::optional<c10::MemoryFormat> memory_format_opt) {
  c10::supa::SupaSysCtrl::GetInstance().supaInit();
  const auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_privateuseone());
  c10::supa::SUPAGuard device_guard(device);
  auto *allocator = at::supa::getSUPADeviceAllocator();
  constexpr c10::DispatchKeySet privateuse1_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, allocator, privateuse1_dks, dtype,
                                   memory_format_opt);
}

TensorBase empty_supa(IntArrayRef size, c10::optional<ScalarType> dtype_opt,
                      c10::optional<Layout> layout_opt,
                      c10::optional<Device> device_opt,
                      c10::optional<bool> pin_memory_opt,
                      c10::optional<c10::MemoryFormat> memory_format_opt) {
  TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt,
              "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) ==
                                   Layout::Strided);

  const auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_supa(size, dtype, device_opt, memory_format_opt);
}

TensorBase empty_supa(IntArrayRef size, const TensorOptions &options) {
  return at::detail::empty_supa(
      size, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(),
      options.device_opt(), options.pinned_memory_opt(),
      options.memory_format_opt());
}

TensorBase empty_strided_supa(IntArrayRef size, IntArrayRef stride,
                              ScalarType dtype,
                              c10::optional<Device> device_opt) {
  c10::supa::SupaSysCtrl::GetInstance().supaInit();
  const auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_privateuseone());
  c10::supa::SUPAGuard device_guard(device);
  auto *allocator = at::supa::getSUPADeviceAllocator();
  constexpr c10::DispatchKeySet privateuse1_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(size, stride, allocator,
                                           privateuse1_dks, dtype);
}

TensorBase empty_strided_supa(IntArrayRef size, IntArrayRef stride,
                              c10::optional<ScalarType> dtype_opt,
                              c10::optional<Layout> layout_opt,
                              c10::optional<Device> device_opt,
                              c10::optional<bool> pin_memory_opt) {
  TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt,
              "Only dense CPU tensors can be pinned");
#ifndef NDEBUG
  const auto layout = layout_or_default(layout_opt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout == Layout::Strided);
#endif

  const auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_strided_supa(size, stride, dtype, device_opt);
}

TensorBase empty_strided_supa(IntArrayRef size, IntArrayRef stride,
                              const TensorOptions &options) {
  return at::detail::empty_strided_supa(
      size, stride, optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}

} // namespace at::detail