/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/core/supa/SUPAContext.h"
#include <c10/util/CallOnce.h>
#include <deque>
#include <vector>
#include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAVersion.h"

namespace at::supa {

namespace {

DeviceIndex num_gpus = -1;
c10::once_flag init_flag;
std::deque<c10::once_flag> device_flags;
std::vector<supaDeviceProp> device_properties;

void initSUPAContextVectors() {
  num_gpus = c10::supa::device_count();
  device_flags.resize(num_gpus);
  device_properties.resize(num_gpus);
}

void initDeviceProperty(DeviceIndex device_index) {
  supaDeviceProp device_prop{};
  C10_SUPA_CHECK(supaGetDeviceProperties(&device_prop, device_index));

  // Compatible with third-party application compute capability version numbers
  if (device_prop.major == 2) {
    device_prop.major = COMPUTE_MAJOR_VERSION_9;
  }
  device_properties[device_index] = device_prop;
}

} // anonymous namespace

int warp_size() {
  return getCurrentDeviceProperties()->warpSize;
}

supaDeviceProp* getCurrentDeviceProperties() {
  auto device = c10::supa::current_device();
  return getDeviceProperties(device);
}

supaDeviceProp* getDeviceProperties(c10::DeviceIndex device) {
  c10::call_once(init_flag, initSUPAContextVectors);
  if (device == -1) {
    device = c10::supa::current_device();
  }
  AT_ASSERT(device >= 0 && device < num_gpus, "device=", static_cast<int>(device), ", num_gpus=", num_gpus);
  c10::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

bool canDeviceAccessPeer(c10::DeviceIndex device, c10::DeviceIndex peer_device) {
  c10::call_once(init_flag, initSUPAContextVectors);
  if (device == -1) {
    device = c10::supa::current_device();
  }
  AT_ASSERT(device >= 0 && device < num_gpus, "device=", static_cast<int>(device), ", num_gpus=", num_gpus);
  AT_ASSERT(
      peer_device >= 0 && peer_device < num_gpus,
      "peer_device=",
      static_cast<int>(peer_device),
      ", num_gpus=",
      num_gpus);
  int can_access = 0;
  C10_SUPA_CHECK(supaDeviceCanAccessPeer(&can_access, device, peer_device));
  return can_access != 0;
}

Allocator* getSUPADeviceAllocator() {
  return c10::supa::SUPACachingAllocator::get();
}

} // namespace at::supa
