/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <c10/core/Allocator.h>
#include <c10/util/SmallVector.h>

#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/core/supa/SUPAStream.h"

namespace at::supa {

struct TORCH_SUPA_API HostAllocator : public at::Allocator {
  // Associates the pinned memory allocation with a stream to track
  // dependencies. This ensures the memory won't be reused until the stream's
  // operations complete
  virtual bool record_event(void* ptr, void* ctx, c10::Stream stream) = 0;

  // Frees all cached pinned memory and returns it to the system, clearing the
  // allocator's internal cache
  virtual void empty_cache() = 0;
};

TORCH_SUPA_API HostAllocator* getCachingHostAllocator();

TORCH_SUPA_API bool CachingHostAllocator_recordEvent(void* ptr, void* ctx, c10::supa::SUPAStream stream);

// Releases cached pinned memory allocations via supaHostFree
TORCH_SUPA_API void CachingHostAllocator_emptyCache();

TORCH_SUPA_API bool CachingHostAllocator_isPinned(const void* ptr);

} // namespace at::supa
