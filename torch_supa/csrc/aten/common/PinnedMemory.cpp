/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/CPUFunctions.h>
#include <ATen/Context.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/Tensor.h>
#include <c10/core/TensorOptions.h>
#include "supa_runtime.h"

#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/core/supa/PinnedMemoryAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAHooks.h"

namespace at::supa {

bool SUPANativeFunctions::is_pinned(const Tensor& self, c10::optional<c10::Device> device) {
  // Only CPU tensors can be pinned
  if (!self.is_cpu()) {
    return false;
  }
  return CachingHostAllocator_isPinned(self.storage().data());
}

Tensor SUPANativeFunctions::_pin_memory(const Tensor& self, c10::optional<c10::Device> device) {
  auto* allocator = getPinnedMemoryAllocator();
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      static_cast<int64_t>(detail::computeStorageNbytes(self.sizes(), self.strides(), self.dtype().itemsize())),
      allocator,
      /*resizable=*/false);
  auto tensor = at::cpu::empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

} // namespace at::supa
