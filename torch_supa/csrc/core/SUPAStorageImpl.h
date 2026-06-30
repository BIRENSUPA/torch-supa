/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/Tensor.h>
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/order_preserving_flat_hash_map.h>
#include <c10/util/typeid.h>
#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/core/supa/TorchVersion.h"
#pragma once

namespace torch_supa {

struct C10_SUPA_API SUPAStorageImpl : public c10::StorageImpl {
  explicit SUPAStorageImpl(
      use_byte_size_t use_byte_size,
      c10::SymInt size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable);
  ~SUPAStorageImpl() override = default;
  SUPAStorageImpl(const SUPAStorageImpl&) = delete;
  SUPAStorageImpl(SUPAStorageImpl&&) = delete;
  SUPAStorageImpl& operator=(const SUPAStorageImpl&) = delete;
  SUPAStorageImpl& operator=(SUPAStorageImpl&&) = delete;

  void release_resources() override;
};

c10::intrusive_ptr<c10::StorageImpl> make_supa_storage_impl(
    c10::StorageImpl::use_byte_size_t use_byte_size,
    c10::SymInt size_bytes,
#if TORCH_VER >= TORCH_2_3_0
    c10::DataPtr data_ptr,
#endif
    c10::Allocator* allocator,
    bool resizable);

} // namespace torch_supa