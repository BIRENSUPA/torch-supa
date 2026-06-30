/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/aten/common/Resize.h"
#include "torch_supa/csrc/core/supa/SUPAContext.h"

namespace at::supa {

void resize_bytes_supa(StorageImpl* storage, size_t size_bytes) {
  TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");
  auto* allocator = storage->allocator();
  TORCH_CHECK(allocator != nullptr, "Trying to resize storage without an allocator");

  c10::Device device = storage->device();

  if (size_bytes == 0) {
    storage->set_data_ptr_noswap(at::DataPtr(nullptr, device));
    storage->set_nbytes(0);
    return;
  }

  c10::supa::SUPAGuard guard(device.index());
  at::DataPtr data = allocator->allocate(size_bytes);
  if (storage->data_ptr()) {
    at::globalContext().lazyInitPrivateUse1();

    C10_SUPA_CHECK(supaMemcpyAsync(
        data.get(),
        storage->data(),
        std::min(storage->nbytes(), size_bytes),
        supaMemcpyDeviceToDevice,
        c10::supa::getCurrentSUPAStream()));
  }

  // Destructively overwrite data_ptr
  storage->set_data_ptr_noswap(std::move(data));
  storage->set_nbytes(size_bytes);
}

} // namespace at::supa
