/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <c10/core/Allocator.h>

// Preserved for BC, as many files depend on these includes
#include <ATen/Context.h>
#include <torch_supa/csrc/core/supa/SUPAException.h>
#include <torch_supa/csrc/core/supa/SUPAFunctions.h>
#include <torch_supa/csrc/core/supa/SUPAStream.h>
#include <torch_supa/csrc/core/supa/sys_ctrl/SupaSysCtrl.h>

namespace at::supa {

inline int64_t getNumGPUs() {
  return c10::supa::device_count();
}

TORCH_SUPA_API supaDeviceProp* getCurrentDeviceProperties();

TORCH_SUPA_API int warp_size();

TORCH_SUPA_API supaDeviceProp* getDeviceProperties(c10::DeviceIndex device);

TORCH_SUPA_API bool canDeviceAccessPeer(
    c10::DeviceIndex device,
    c10::DeviceIndex peer_device);

TORCH_SUPA_API c10::Allocator* getSUPADeviceAllocator();

} // namespace at::supa
