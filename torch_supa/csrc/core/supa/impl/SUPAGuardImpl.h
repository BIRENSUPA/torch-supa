/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <cassert>

#include <torch_supa/csrc/core/supa/SUPAMacros.h>
#include "torch_supa/csrc/core/supa/TorchVersion.h"

namespace c10::supa {
namespace impl {

struct C10_SUPA_API SUPAGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  SUPAGuardImpl() {}
  explicit SUPAGuardImpl(c10::DeviceType t);
  c10::DeviceType type() const override {
    return c10::DeviceType::PrivateUse1;
  }
  c10::Device exchangeDevice(c10::Device d) const override;
  c10::Device getDevice() const override;
  c10::optional<c10::Device> uncheckedGetDevice() const noexcept;
  void setDevice(c10::Device d) const override;
  void uncheckedSetDevice(c10::Device d) const noexcept override;

  c10::Stream getStream(c10::Device d) const noexcept override;
  c10::Stream getDefaultStream(c10::Device d) const override;
  c10::Stream getStreamFromGlobalPool(c10::Device d, bool isHighPriority = false) const override;
  // NB: These do NOT set the current device
  c10::Stream exchangeStream(c10::Stream s) const noexcept override;
  c10::DeviceIndex deviceCount() const noexcept override;

  // Event-related functions
  void createEvent(void** supa_event, [[maybe_unused]] c10::EventFlag flag) const;
  void destroyEvent(void* event, c10::DeviceIndex device_index) const noexcept override;
  void record(void** event, const c10::Stream& stream, c10::DeviceIndex device_index, c10::EventFlag flag)
      const override;
  void block(void* event, const c10::Stream& stream) const override;
  // May be called from any device
  bool queryEvent(void* event) const override;
  // Stream-related functions
  bool queryStream(const Stream& stream) const override;
  void synchronizeStream(const Stream& stream) const override;

#if TORCH_VER >= TORCH_2_4_0
  c10::Stream getNewStream(c10::Device d, int priority = 0) const override;
  void synchronizeEvent(void* event) const override;
  double elapsedTime(void* event1, void* event2, DeviceIndex device_index) const override;
#endif

#if TORCH_VER >= TORCH_2_6_0
  // Note: synchronizeDevice can be safely called from any device
  void synchronizeDevice(c10::DeviceIndex device_index) const override;
#endif
  void recordDataPtrOnStream(const c10::DataPtr& data_ptr, const c10::Stream& stream) const override;
};

} // namespace impl
} // namespace c10::supa
