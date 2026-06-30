/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/core/supa/TorchVersion.h"
#include "torch_supa/csrc/supa/SUPAPluggableAllocator.h"

#if TORCH_VER >= TORCH_2_9_0
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

namespace {
using namespace c10d::symmetric_memory;

// Alloc functor for MemPool
void* supa_symm_alloc(size_t size, int device, void* stream) {
  static auto allocator = get_allocator(c10::DeviceType::PrivateUse1);
  // Note: the group info is now specified at the time of rendezvous instead of
  // allocation. We thus pass `nullopt` for group here.
  return allocator->alloc(size, device, /*group_name=*/std::nullopt);
}

// Free functor for MemPool
void supa_symm_free(void* ptr, size_t size, int device, void* stream) {
  static auto allocator = get_allocator(c10::DeviceType::PrivateUse1);
  allocator->free(ptr);
}

// Register allocator for SUPA MemPool
struct RegisterSUPAMemPoolAllocator {
  RegisterSUPAMemPoolAllocator() {
    std::shared_ptr<c10::supa::SUPACachingAllocator::SUPAAllocator> allocator =
        torch_supa::supa::SUPAPluggableAllocator::createCustomAllocator(supa_symm_alloc, supa_symm_free);
    register_mempool_allocator(c10::DeviceType::PrivateUse1, allocator);
  }
};

RegisterSUPAMemPoolAllocator register_supa_mempool_allocator_;

} // namespace

#endif
