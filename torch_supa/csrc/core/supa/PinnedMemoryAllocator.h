/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include "torch_supa/csrc/core/supa/CachingHostAllocator.h"

namespace at::supa {

inline TORCH_SUPA_API HostAllocator* getPinnedMemoryAllocator() {
  return getCachingHostAllocator();
}

} // namespace at::supa
