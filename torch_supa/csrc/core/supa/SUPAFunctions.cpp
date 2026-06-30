/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#include <limits>
#include <optional>

#ifndef BESU_NOT_FOUND
#include "besu.h"
#endif
#if TORCH_VER >= TORCH_2_5_0
#include <c10/util/WaitCounter.h>
#endif

#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAGraphsUtils.h"
#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/utils/logger/Logger.h"

namespace c10::supa {

namespace {

// returns -1 on failure
int32_t driver_version() {
  int driver_version = -1;
  supaError_t err = supaDriverGetVersion(&driver_version);
  if (err != supaSuccess) {
    [[maybe_unused]] supaError_t last_err = supaGetLastError();
  }
  return driver_version;
}

int device_count_impl(bool fail_if_no_driver) {
  int count = -1;
  auto err = supaGetDeviceCount(&count);
  if (err == supaSuccess) {
    return count;
  }
  // Clear out the error state, so we don't spuriously trigger someone else.
  // (This shouldn't really matter, since we won't be running very much SUPA
  // code in this regime.)
  [[maybe_unused]] supaError_t last_err = supaGetLastError();
  switch (err) {
    case supaErrorNoDevice:
      // Zero devices is ok here
      count = 0;
      break;
    case supaErrorInsufficientDriver: {
      auto version = driver_version();
      if (version <= 0) {
        if (!fail_if_no_driver) {
          // No SUPA driver means no devices
          count = 0;
          break;
        }
        TORCH_CHECK(
            false,
            "Found no SUPA driver on your system. Please check that you "
            "have an SUPA GPU and installed a driver.");
      } else {
        TORCH_CHECK(
            false,
            "The SUPA driver on your system is too old (found version ",
            version,
            "). Please update your GPU driver by downloading and installing "
            "a new version.");
      }
    } break;
    case supaErrorUnknown:
      TORCH_CHECK(
          false,
          "SUPA unknown error - this may be due to an "
          "incorrectly set up environment, e.g. changing env "
          "variable SUPA_VISIBLE_DEVICES after program start. "
          "Setting the available devices to be zero.");
      break;
    case supaErrorMemoryAllocation:
      TORCH_CHECK(false, "Got 'out of memory' error while trying to initialize SUPA");
      break;
    default:
      TORCH_CHECK(false, "Unexpected error from supaGetDeviceCount(). {}: {} ", err, supaGetErrorString(err));
  }
  return count;
}

} // namespace

DeviceIndex device_count() noexcept {
  // initialize number of devices only once
  static int count = device_count_impl(/*fail_if_no_driver=*/false);
  TORCH_INTERNAL_ASSERT(
      count <= std::numeric_limits<DeviceIndex>::max(), "Too many SUPA devices, DeviceIndex overflowed");
  return static_cast<DeviceIndex>(count);
}

DeviceIndex device_count_ensure_non_zero() {
  // Call the implementation every time to throw the exception
  int count = device_count_impl(/*fail_if_no_driver=*/true);
  // Zero gpus doesn't produce a warning in `device_count` but we fail here
  TORCH_CHECK(count, "No BIREN GPUs are available");
  return static_cast<DeviceIndex>(count);
}

DeviceIndex current_device() {
  DeviceIndex cur_device = -1;
  C10_SUPA_CHECK(c10::supa::GetDevice(&cur_device));
  return cur_device;
}

void set_device(DeviceIndex device, const bool force) {
  C10_SUPA_CHECK(c10::supa::SetDevice(device, force));
}

void device_synchronize() {
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_device_synchronization(c10::kPrivateUse1);
  }
#if TORCH_VER >= TORCH_2_5_0
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.supa_device_synchronize);
#endif
  C10_SUPA_CHECK(supaDeviceSynchronize());
}

// this function has to be called from callers performing supa synchronizing
// operations, to raise proper error or warning
void warn_or_error_on_sync() {
  if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_ERROR) {
    TORCH_CHECK(false, "called a synchronizing SUPA operation");
  } else if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_WARN) {
    TORCH_SUPA_WARN("called a synchronizing SUPA operation");
  }
}

c10::optional<DeviceIndex> getDeviceIndexWithPrimaryContext() {
  // check current device first
  auto current_device_index = current_device();
  if (current_device_index >= 0) {
    if (hasPrimaryContext(current_device_index)) {
      return current_device_index;
    }
  }
  for (const auto device_index : c10::irange(at::supa::device_count())) {
    if (device_index == current_device_index) {
      continue;
    }
    if (hasPrimaryContext(device_index)) {
      return device_index;
    }
  }
  return c10::nullopt;
}

bool hasPrimaryContext(DeviceIndex device_index) {
  TORCH_CHECK(
      device_index >= 0 && device_index < c10::supa::device_count(),
      "hasPrimaryContext expects a valid device index, but got device_index=",
      device_index);
  unsigned int ctx_flags = 0;
  int ctx_is_active = 0;
  AT_SUPA_DRIVER_CHECK(suDevicePrimaryCtxGetState(device_index, &ctx_flags, &ctx_is_active));
  return ctx_is_active == 1;
}

void begin_vector_dump(supaStream_t stream) {
#ifdef BESU_NOT_FOUND
  TORCH_CHECK(false, "Not found Besu for vector dump");
#else
  BesuResult status = besuStreamBeginVectorDump(reinterpret_cast<BesuStream>(stream));
  TORCH_CHECK(status == BESU_SUCCESS, "besuStreamBeginVectorDump failed with status {}", status);
#endif
}

void end_vector_dump(supaStream_t stream) {
#ifdef BESU_NOT_FOUND
  TORCH_CHECK(false, "Not found Besu for vector dump");
#else
  BesuResult status = besuStreamEndVectorDump(reinterpret_cast<BesuStream>(stream));
  TORCH_CHECK(status == BESU_SUCCESS, "besuStreamEndVectorDump failed with status {}", status);
#endif
}

bool canDeviceAccessPeer(DeviceIndex device, DeviceIndex peer_device) {
  if (device == -1) {
    device = c10::supa::current_device();
  }
  auto num_gpus = c10::supa::device_count();
  TORCH_CHECK(
      device >= 0 && device < num_gpus, "device=", static_cast<int>(device), ", num_gpus=", static_cast<int>(num_gpus));
  TORCH_CHECK(
      peer_device >= 0 && peer_device < num_gpus,
      "peer_device=",
      static_cast<int>(peer_device),
      ", num_gpus=",
      static_cast<int>(num_gpus));
  int can_access = 0;
  C10_SUPA_CHECK(supaDeviceCanAccessPeer(&can_access, device, peer_device));
  return can_access != 0;
}

// Wrappers for raw SUPA device management functions
supaError_t GetDeviceCount(int* dev_count) {
  return supaGetDeviceCount(dev_count);
}

static thread_local DeviceIndex k_device = -1;

supaError_t GetDevice(DeviceIndex* device) {
  if (k_device >= 0) {
    *device = k_device;
    return supaSuccess;
  }

  int tmp_device = -1;
  auto error = supaGetDevice(&tmp_device);
  if (error == supaSuccess) {
    *device = static_cast<DeviceIndex>(tmp_device);
  }

  return error;
}

supaError_t SetDevice(DeviceIndex device, const bool force) {
  TORCH_CHECK(device >= 0, "device id must be positive!", device);
  k_device = -1;
  if (force) {
    return supaSetDevice(device);
  }
  int cur_device = -1;
  C10_SUPA_CHECK(supaGetDevice(&cur_device));
  if (device == cur_device) {
    return supaSuccess;
  }
  return supaSetDevice(device);
}

supaError_t MaybeSetDevice(DeviceIndex device) {
  if (hasPrimaryContext(device)) {
    return c10::supa::SetDevice(device);
  }
  k_device = device;
  return supaSuccess;
}

// This function always initializes the SUPA context
// on to_device
DeviceIndex ExchangeDevice(DeviceIndex device) {
  auto cur_device = k_device;
  k_device = -1;
  if (cur_device < 0) {
    int tmp_device = -1;
    C10_SUPA_CHECK(supaGetDevice(&tmp_device));
    cur_device = static_cast<DeviceIndex>(tmp_device);
    if (device == cur_device) {
      return cur_device;
    }
  }
  C10_SUPA_CHECK(supaSetDevice(device));
  return cur_device;
}

// This function does not initialize the SUPA context
// on to_device if it does not already exist
DeviceIndex MaybeExchangeDevice(DeviceIndex to_device) {
  int tmp_cur_device = -1;
  C10_SUPA_CHECK(supaGetDevice(&tmp_cur_device));
  TORCH_INTERNAL_ASSERT(
      tmp_cur_device >= 0 && tmp_cur_device <= std::numeric_limits<DeviceIndex>::max(),
      "supaGetDevice returns invalid device ",
      tmp_cur_device);
  auto cur_device = static_cast<DeviceIndex>(tmp_cur_device);
  if (to_device == tmp_cur_device) {
    return cur_device;
  }
  if (hasPrimaryContext(to_device)) {
    C10_SUPA_CHECK(supaSetDevice(to_device));
  } else {
    k_device = to_device;
  }
  return cur_device;
}

void SetTargetDevice() {
  if (k_device >= 0) {
    C10_SUPA_CHECK(c10::supa::SetDevice(k_device));
  }
}

} // namespace c10::supa
