/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/core/SUPAStorageImpl.h"
#include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"

namespace torch_supa {

SUPAStorageImpl::SUPAStorageImpl(
    use_byte_size_t use_byte_size,
    c10::SymInt size_bytes,
    at::DataPtr data_ptr,
    at::Allocator* allocator,
    bool resizable)
    : c10::StorageImpl(use_byte_size, size_bytes, at::DataPtr(std::move(data_ptr)), allocator, resizable) {}

void SUPAStorageImpl::release_resources() {
  StorageImpl::release_resources();
}

#if TORCH_VER >= TORCH_2_3_0
c10::intrusive_ptr<c10::StorageImpl> make_supa_storage_impl(
    c10::StorageImpl::use_byte_size_t use_byte_size,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable) {
  if (data_ptr == nullptr) {
    data_ptr = allocator->allocate(size_bytes.as_int_unchecked());
  }
  // Correctly create SUPAStorageImpl object.
  c10::intrusive_ptr<c10::StorageImpl> SUPA_storage_impl = c10::make_intrusive<SUPAStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes.as_int_unchecked(), std::move(data_ptr), allocator, resizable);
  return SUPA_storage_impl;
}
#else
c10::intrusive_ptr<c10::StorageImpl> make_supa_storage_impl(
    c10::StorageImpl::use_byte_size_t use_byte_size,
    c10::SymInt size_bytes,
    c10::Allocator* allocator,
    bool resizable) {
  // Correctly create SUPAStorageImpl object.
  c10::intrusive_ptr<c10::StorageImpl> SUPA_storage_impl = c10::make_intrusive<SUPAStorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes.as_int_unchecked(),
      allocator->allocate(size_bytes.as_int_unchecked()),
      allocator,
      resizable);
  return SUPA_storage_impl;
}
#endif

} // namespace torch_supa
