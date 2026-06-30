/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <c10/core/impl/GPUTrace.h>

#include "torch_supa/csrc/core/SUPAStorageImpl.h"
#include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAHooks.h"
#include "torch_supa/csrc/core/supa/SUPAStream.h"
#include "torch_supa/csrc/core/supa/impl/SUPAGuardImpl.h"

namespace c10::supa {
namespace impl {

SUPAGuardImpl::SUPAGuardImpl(c10::DeviceType t) {
  TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1);
}

c10::Device SUPAGuardImpl::exchangeDevice(c10::Device d) const {
  TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1);
  auto old_device_index = c10::supa::ExchangeDevice(d.index());
  return Device(c10::DeviceType::PrivateUse1, old_device_index);
}

c10::Device SUPAGuardImpl::getDevice() const {
  c10::DeviceIndex device = 0;
  C10_SUPA_CHECK(c10::supa::GetDevice(&device));
  return c10::Device(c10::DeviceType::PrivateUse1, device);
}

c10::optional<c10::Device> SUPAGuardImpl::uncheckedGetDevice() const noexcept {
  c10::DeviceIndex device{-1};
  const auto err = c10::supa::GetDevice(&device);
  C10_SUPA_CHECK_WARN(err);
  if (err != supaSuccess) {
    return c10::nullopt;
  }
  return c10::Device(c10::DeviceType::PrivateUse1, device);
}

void SUPAGuardImpl::setDevice(c10::Device d) const {
  TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1);
  C10_SUPA_CHECK(c10::supa::SetDevice(d.index()));
}

void SUPAGuardImpl::uncheckedSetDevice(c10::Device d) const noexcept {
  C10_SUPA_CHECK_WARN(c10::supa::SetDevice(d.index()));
}

c10::Stream SUPAGuardImpl::getStream(c10::Device d) const noexcept {
  return getCurrentSUPAStream(d.index()).unwrap();
}

c10::Stream SUPAGuardImpl::getDefaultStream(c10::Device d) const {
  return getCurrentSUPAStream(d.index());
}

c10::Stream SUPAGuardImpl::getStreamFromGlobalPool(c10::Device d, bool isHighPriority) const {
  return getStreamFromPool(isHighPriority, d.index());
}

#if TORCH_VER >= TORCH_2_4_0
c10::Stream SUPAGuardImpl::getNewStream(c10::Device d, int priority) const {
  return getStreamFromPool(priority, d.index());
}
#endif

c10::Stream SUPAGuardImpl::exchangeStream(Stream s) const noexcept {
  SUPAStream cs(s);
  auto old_stream = getCurrentSUPAStream(s.device().index());
  setCurrentSUPAStream(cs);
  return old_stream.unwrap();
}

c10::DeviceIndex SUPAGuardImpl::deviceCount() const noexcept {
  return device_count();
}

// Event-related functions
void SUPAGuardImpl::createEvent(void** supa_event, const c10::EventFlag /*flag*/) const {
  // Maps PyTorch's Event::Flag to supa flag
  // TODO: add suEventBlockingSync flag
  auto supa_flag = supaEventDefault;
  supaEvent_t* supa_event_tmp = reinterpret_cast<supaEvent_t*>(supa_event);
  C10_SUPA_CHECK(supaEventCreateWithFlags(supa_event_tmp, supa_flag));
}

void SUPAGuardImpl::destroyEvent(void* event, const c10::DeviceIndex device_index) const noexcept {
  if (!event) {
    return;
  }
  auto* supa_event = static_cast<supaEvent_t>(event);
  DeviceIndex orig_device{-1};
  C10_SUPA_CHECK_WARN(c10::supa::GetDevice(&orig_device));
  C10_SUPA_CHECK_WARN(c10::supa::SetDevice(device_index));
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
#if TORCH_VER >= TORCH_2_4_0
    (*interp)->trace_gpu_event_deletion(c10::kPrivateUse1, reinterpret_cast<uintptr_t>(supa_event));
#else
    (*interp)->trace_gpu_event_deletion(reinterpret_cast<uintptr_t>(supa_event));
#endif
  }
  C10_SUPA_CHECK_WARN(supaEventDestroy(supa_event));
  C10_SUPA_CHECK_WARN(c10::supa::SetDevice(orig_device));
}

void SUPAGuardImpl::record(
    void** event,
    const c10::Stream& stream,
    const c10::DeviceIndex device_index,
    const c10::EventFlag flag) const {
  TORCH_CHECK(
      device_index == -1 || device_index == stream.device_index(),
      "Event device index ",
      device_index,
      " does not match recording stream's device index ",
      stream.device_index(),
      ".");
  supaEvent_t supa_event = static_cast<supaEvent_t>(*event);
  SUPAStream supa_stream{stream};

  // Moves to stream's device to record
  const auto orig_device = getDevice();
  setDevice(stream.device());

  // Creates the event (lazily)
  if (!supa_event) {
    createEvent(event, flag);
  }

  supa_event = reinterpret_cast<supaEvent_t>(*event);
  C10_SUPA_CHECK(supaEventRecord(supa_event, supa_stream));
  // Makes the void* point to the (possibly just allocated) SUPA event
  *event = supa_event;
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
#if TORCH_VER >= TORCH_2_4_0
    (*interp)->trace_gpu_event_record(
        c10::kPrivateUse1, reinterpret_cast<uintptr_t>(supa_event), reinterpret_cast<uintptr_t>(supa_stream.stream()));
#else
    (*interp)->trace_gpu_event_record(
        reinterpret_cast<uintptr_t>(supa_event), reinterpret_cast<uintptr_t>(supa_stream.stream()));
#endif
  }

  // Resets device
  setDevice(orig_device);
}

void SUPAGuardImpl::block(void* event, const c10::Stream& stream) const {
  if (!event) {
    return;
  }
  supaEvent_t supa_event = static_cast<supaEvent_t>(event);
  SUPAStream supa_stream{stream};
  const auto orig_device = getDevice();
  setDevice(stream.device());
  C10_SUPA_CHECK(supaStreamWaitEvent(
      supa_stream,
      supa_event,
      /*flags (must be zero)=*/0));
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
#if TORCH_VER >= TORCH_2_4_0
    (*interp)->trace_gpu_event_wait(
        c10::kPrivateUse1, reinterpret_cast<uintptr_t>(supa_event), reinterpret_cast<uintptr_t>(supa_stream.stream()));
#else
    (*interp)->trace_gpu_event_wait(
        reinterpret_cast<uintptr_t>(supa_event), reinterpret_cast<uintptr_t>(supa_stream.stream()));
#endif
  }
  setDevice(orig_device);
}

bool SUPAGuardImpl::queryEvent(void* event) const {
  if (!event) {
    return true;
  }
  supaEvent_t supa_event = static_cast<supaEvent_t>(event);
  const supaError_t err = C10_SUPA_ERROR_HANDLED(supaEventQuery(supa_event));
  if (err != supaErrorNotReady) {
    C10_SUPA_CHECK(err);
  } else {
    // ignore and clear the error if not ready
    (void)supaGetLastError();
  }
  return (err == supaSuccess);
}

void SUPAGuardImpl::recordDataPtrOnStream(const c10::DataPtr& data_ptr, const c10::Stream& stream) const {
  SUPAStream supa_stream{stream};
  SUPACachingAllocator::recordStream(data_ptr, supa_stream);
}

bool SUPAGuardImpl::queryStream(const Stream& stream) const {
  SUPAStream supa_stream{stream};
  return supa_stream.query();
}

void SUPAGuardImpl::synchronizeStream(const Stream& stream) const {
  SUPAStream supa_stream{stream};
  supa_stream.synchronize();
}

#if TORCH_VER >= TORCH_2_4_0
void SUPAGuardImpl::synchronizeEvent(void* event) const {
  if (!event) {
    return;
  }
  supaEvent_t supa_event = static_cast<supaEvent_t>(event);
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_event_synchronization(c10::kPrivateUse1, reinterpret_cast<uintptr_t>(supa_event));
  }
  // Note: supaEventSynchronize can be safely called from any device
  C10_SUPA_CHECK(supaEventSynchronize(supa_event));
}

double SUPAGuardImpl::elapsedTime(void* event1, void* event2, const DeviceIndex device_index) const {
  TORCH_CHECK(event1 && event2, "Both events must be recorded before calculating elapsed time.");
  // Even though supaEventElapsedTime can be safely called from any device, if
  // the current device is not initialized, it will create a new supa context,
  // which will consume a lot of memory.
  DeviceIndex orig_device{-1};
  C10_SUPA_CHECK(c10::supa::GetDevice(&orig_device));
  C10_SUPA_CHECK(c10::supa::SetDevice(device_index));
  supaEvent_t supa_event1 = static_cast<supaEvent_t>(event1);
  supaEvent_t supa_event2 = static_cast<supaEvent_t>(event2);
  float time_ms = 0;
  // raise supaErrorNotReady if either event is recorded but not yet completed
  C10_SUPA_CHECK(supaEventElapsedTime(&time_ms, supa_event1, supa_event2));
  C10_SUPA_CHECK(c10::supa::SetDevice(orig_device));
  return static_cast<double>(time_ms);
}
#endif

#if TORCH_VER >= TORCH_2_6_0
// Note: synchronizeDevice can be safely called from any device
void SUPAGuardImpl::synchronizeDevice(const c10::DeviceIndex device_index) const {
  DeviceIndex orig_device{-1};
  C10_SUPA_CHECK(c10::supa::GetDevice(&orig_device));
  C10_SUPA_CHECK(c10::supa::SetDevice(device_index));
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_device_synchronization(c10::kPrivateUse1);
  }
  C10_SUPA_CHECK(supaDeviceSynchronize());
  C10_SUPA_CHECK(c10::supa::SetDevice(orig_device));
}
#endif

C10_REGISTER_GUARD_IMPL(PrivateUse1, SUPAGuardImpl);

#define REGISTER_PRIVATEUSE1_BACKEND(name)                                                        \
  int rename_privateuse1_backend() {                                                              \
    c10::register_privateuse1_backend(#name);                                                     \
    c10::SetStorageImplCreate(c10::DeviceType::PrivateUse1, &torch_supa::make_supa_storage_impl); \
    at::RegisterPrivateUse1HooksInterface(c10::supa::get_supa_hooks());                           \
    return 0;                                                                                     \
  }                                                                                               \
  static const int _temp_##name = rename_privateuse1_backend();

REGISTER_PRIVATEUSE1_BACKEND(supa)

} // namespace impl

} // namespace c10::supa
