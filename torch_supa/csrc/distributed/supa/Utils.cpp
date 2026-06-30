/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/distributed/Utils.hpp"
#include <supa_runtime.h>
#include "torch_supa/csrc/core/supa/DriverAPI.h"

namespace c10d::supa {

bool deviceSupportsMulticast(int device_idx) {
  auto* driver_api = c10::supa::DriverAPI::get();
  int multicast_supported = 0;
  C10_SUPA_DRIVER_CHECK(
      driver_api->suDeviceGetAttribute_(&multicast_supported, SU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device_idx));
  return driver_api->suMulticastCreate_ != nullptr && multicast_supported;
}

} // namespace c10d::supa
