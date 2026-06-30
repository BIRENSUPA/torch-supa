/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once

// This header provides C++ wrappers around commonly used SUPA API functions.
// The benefit of using C++ here is that we can raise an exception in the
// event of an error, rather than explicitly pass around error codes.  This
// leads to more natural APIs.
//
// The naming convention used here matches the naming convention of torch.supa

#include <c10/core/Device.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/util/irange.h>
#include <supa_runtime.h>
#include <torch_supa/csrc/core/supa/SUPAException.h>
#include <torch_supa/csrc/core/supa/SUPAMacros.h>
#include <torch_supa/csrc/core/supa/TorchVersion.h>
#include <optional>

namespace c10::supa {

// NB: In the past, we were inconsistent about whether or not this reported
// an error if there were driver problems are not.  Based on experience
// interacting with users, it seems that people basically ~never want this
// function to fail; it should just return zero if things are not working.
// Oblige them.
// It still might log a warning for user first time it's invoked
C10_SUPA_API DeviceIndex device_count() noexcept;

// Version of device_count that throws is no devices are detected
C10_SUPA_API DeviceIndex device_count_ensure_non_zero();

C10_SUPA_API DeviceIndex current_device();

C10_SUPA_API void set_device(DeviceIndex device, bool force = false);

C10_SUPA_API void device_synchronize();

C10_SUPA_API void warn_or_error_on_sync();

C10_SUPA_API supaError_t GetDeviceCount(int* dev_count);

C10_SUPA_API supaError_t GetDevice(DeviceIndex* device);

C10_SUPA_API supaError_t SetDevice(DeviceIndex device, bool force = false);

C10_SUPA_API supaError_t MaybeSetDevice(DeviceIndex device);

C10_SUPA_API DeviceIndex ExchangeDevice(DeviceIndex device);

C10_SUPA_API DeviceIndex MaybeExchangeDevice(DeviceIndex device);

C10_SUPA_API void SetTargetDevice();

enum class SyncDebugMode { L_DISABLED = 0, L_WARN, L_ERROR };

// this is a holder for c10 global state (similar to at GlobalContext)
// currently it's used to store supa synchronization warning state,
// but can be expanded to hold other related global state, e.g. to
// record stream usage
class WarningState {
 public:
  void set_sync_debug_mode(SyncDebugMode l) {
    sync_debug_mode = l;
  }

  SyncDebugMode get_sync_debug_mode() {
    return sync_debug_mode;
  }

 private:
  SyncDebugMode sync_debug_mode = SyncDebugMode::L_DISABLED;
};

C10_SUPA_API __inline__ WarningState& warning_state() {
  static WarningState warning_state_;
  return warning_state_;
}
// the subsequent functions are defined in the header because for performance
// reasons we want them to be inline
C10_SUPA_API void __inline__ memcpy_and_sync(
    void* dst,
    const void* src,
    size_t nbytes,
    supaMemcpyKind kind,
    supaStream_t stream) {
  if (C10_UNLIKELY(warning_state().get_sync_debug_mode() != SyncDebugMode::L_DISABLED)) {
    warn_or_error_on_sync();
  }
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
#if TORCH_VER >= TORCH_2_4_0
    (*interp)->trace_gpu_stream_synchronization(c10::kPrivateUse1, reinterpret_cast<uintptr_t>(stream));
#else
    (*interp)->trace_gpu_stream_synchronization(reinterpret_cast<uintptr_t>(stream));
#endif
  }
  C10_SUPA_CHECK(supaMemcpyAsync(dst, src, nbytes, kind, stream));
  C10_SUPA_CHECK(supaStreamSynchronize(stream));
}

C10_SUPA_API void __inline__ stream_synchronize(supaStream_t stream) {
  if (C10_UNLIKELY(warning_state().get_sync_debug_mode() != SyncDebugMode::L_DISABLED)) {
    warn_or_error_on_sync();
  }
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
#if TORCH_VER >= TORCH_2_4_0
    (*interp)->trace_gpu_stream_synchronization(c10::kPrivateUse1, reinterpret_cast<uintptr_t>(stream));
#else
    (*interp)->trace_gpu_stream_synchronization(reinterpret_cast<uintptr_t>(stream));
#endif
  }
  C10_SUPA_CHECK(supaStreamSynchronize(stream));
}

void begin_vector_dump(supaStream_t stream);
void end_vector_dump(supaStream_t stream);

C10_SUPA_API bool canDeviceAccessPeer(DeviceIndex device, DeviceIndex peer_device);

C10_SUPA_API bool hasPrimaryContext(DeviceIndex device_index);

C10_SUPA_API c10::optional<DeviceIndex> getDeviceIndexWithPrimaryContext();

inline bool is_available() {
  return c10::supa::device_count() > 0;
}

} // namespace c10::supa
