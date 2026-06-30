/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <c10/core/impl/GPUTrace.h>

#include "torch_supa/csrc/core/supa/SUPAEvent.h"
#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/SUPAGuard.h"
#include "torch_supa/csrc/core/supa/TorchVersion.h"
#include "torch_supa/csrc/core/supa/sys_ctrl/SupaSysCtrl.h"

namespace c10::supa {

void SUPAEvent::moveHelper(SUPAEvent&& other) {
  std::swap(flags_, other.flags_);
  std::swap(is_created_, other.is_created_);
  std::swap(was_recorded_, other.was_recorded_);
  std::swap(device_index_, other.device_index_);
  std::swap(event_, other.event_);
}

SUPAEvent::SUPAEvent(DeviceIndex device_index, const supaIpcEventHandle_t* handle) : device_index_(device_index) {
  SUPAGuard guard(device_index_);

  C10_SUPA_CHECK(supaIpcOpenEventHandle(&event_, *handle));
  is_created_ = true;
}

SUPAEvent::~SUPAEvent() {
  try {
    if (is_created_) {
      SUPAGuard guard(device_index_);
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
#if TORCH_VER >= TORCH_2_4_0
        (*interp)->trace_gpu_event_deletion(c10::kPrivateUse1, reinterpret_cast<uintptr_t>(event_));
#else
        (*interp)->trace_gpu_event_deletion(reinterpret_cast<uintptr_t>(event_));
#endif
      }
      C10_SUPA_CHECK(supaEventDestroy(event_));
    }
  } catch (...) { /* No throw */
  }
}

SUPAEvent::SUPAEvent(SUPAEvent&& other) noexcept {
  moveHelper(std::move(other));
}

SUPAEvent& SUPAEvent::operator=(SUPAEvent&& other) noexcept {
  if (this != &other) {
    moveHelper(std::move(other));
  }
  return *this;
}

bool SUPAEvent::query() const {
  if (!is_created_) {
    return true;
  }

  supaError_t err = supaEventQuery(event_);
  if (err == supaSuccess) {
    return true;
  }
  if (err != supaErrorNotReady) {
    C10_SUPA_CHECK(err);
  }

  // ignore and clear the error if not ready
  (void)supaGetLastError();

  return false;
}

void SUPAEvent::record() {
  record(getCurrentSUPAStream());
}

void SUPAEvent::recordOnce(const SUPAStream& stream) {
  if (!was_recorded_) {
    record(stream);
  }
}

void SUPAEvent::record(const SUPAStream& stream) {
  if (!is_created_) {
    createEvent(stream.device_index());
  }

  TORCH_CHECK(
      device_index_ == stream.device_index(),
      "Event device ",
      device_index_,
      " does not match recording stream's device ",
      stream.device_index(),
      ".");
  SUPAGuard guard(device_index_);
  C10_SUPA_CHECK(supaEventRecord(event_, stream));
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
#if TORCH_VER >= TORCH_2_4_0
    (*interp)->trace_gpu_event_record(
        c10::kPrivateUse1, reinterpret_cast<uintptr_t>(event_), reinterpret_cast<uintptr_t>(stream.stream()));
#else
    (*interp)->trace_gpu_event_record(
        reinterpret_cast<uintptr_t>(event_), reinterpret_cast<uintptr_t>(stream.stream()));
#endif
  }
  was_recorded_ = true;
}

// Note: supaStreamWaitEvent must be called on the same device as the stream.
// The event has no actual GPU resources associated with it.
void SUPAEvent::block(const SUPAStream& stream) {
  if (is_created_) {
    SUPAGuard guard(stream.device_index());
    C10_SUPA_CHECK(supaStreamWaitEvent(stream, event_, 0));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
#if TORCH_VER >= TORCH_2_4_0
      (*interp)->trace_gpu_event_wait(
          c10::kPrivateUse1, reinterpret_cast<uintptr_t>(event_), reinterpret_cast<uintptr_t>(stream.stream()));
#else
      (*interp)->trace_gpu_event_wait(
          reinterpret_cast<uintptr_t>(event_), reinterpret_cast<uintptr_t>(stream.stream()));
#endif
    }
  }
}

// Note: supaEventElapsedTime can be safely called from any device
float SUPAEvent::elapsed_time(const SUPAEvent& other) const {
  TORCH_CHECK(is_created_ && other.isCreated(), "Both events must be recorded before calculating elapsed time.");
  float time_ms = 0;
  // raise supaErrorNotReady if either event is recorded but not yet completed
  C10_SUPA_CHECK(supaEventElapsedTime(&time_ms, event_, other.event_));
  return time_ms;
}

// Note: supaEventSynchronize can be safely called from any device
void SUPAEvent::synchronize() const {
  if (is_created_) {
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
#if TORCH_VER >= TORCH_2_4_0
      (*interp)->trace_gpu_event_synchronization(c10::kPrivateUse1, reinterpret_cast<uintptr_t>(event_));
#else
      (*interp)->trace_gpu_event_synchronization(reinterpret_cast<uintptr_t>(event_));
#endif
    }
    C10_SUPA_CHECK(supaEventSynchronize(event_));
  }
}

void SUPAEvent::ipc_handle(supaIpcEventHandle_t* handle) {
  if (!is_created_) {
    // this SUPAEvent object was initially constructed from flags but event_
    // is not created yet.
    createEvent(getCurrentSUPAStream().device_index());
  }
  SUPAGuard guard(device_index_);
  C10_SUPA_CHECK(supaIpcGetEventHandle(handle, event_));
}

void SUPAEvent::createEvent(DeviceIndex device_index) {
  device_index_ = device_index;
  SUPAGuard guard(device_index_);
  C10_SUPA_CHECK(supaEventCreateWithFlags(&event_, flags_));
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
#if TORCH_VER >= TORCH_2_4_0
    (*interp)->trace_gpu_event_creation(c10::kPrivateUse1, reinterpret_cast<uintptr_t>(event_));
#else
    (*interp)->trace_gpu_event_creation(reinterpret_cast<uintptr_t>(event_));
#endif
  }
  is_created_ = true;
}

} // namespace c10::supa
