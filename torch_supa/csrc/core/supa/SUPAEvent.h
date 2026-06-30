/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <supa_runtime.h>

#include <cstdint>
#include <utility>
#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/core/supa/SUPAStream.h"

namespace c10::supa {
/*
 * SUPAEvents are movable not copyable wrappers around SUPA's events.
 * SUPAEvents are constructed lazily when first recorded.
 */
struct C10_SUPA_API SUPAEvent {
  // Default value for `flags` is specified below - it's supaEventDisableTiming
  SUPAEvent() noexcept = default;
  SUPAEvent(unsigned int flags) noexcept : flags_{flags} {}
  SUPAEvent(DeviceIndex device_index, const supaIpcEventHandle_t* handle);

  ~SUPAEvent();

  SUPAEvent(const SUPAEvent&) = delete;
  SUPAEvent& operator=(const SUPAEvent&) = delete;

  SUPAEvent(SUPAEvent&& other) noexcept;
  SUPAEvent& operator=(SUPAEvent&& other) noexcept;

  operator supaEvent_t() const {
    return event();
  }

  // aclrtEvent do not support Less than operator until now

  c10::optional<at::Device> device() const {
    if (is_created_) {
      return at::Device(c10::DeviceType::PrivateUse1, device_index_);
    }
    return {};
  }

  bool isCreated() const {
    return is_created_;
  }
  c10::DeviceIndex device_index() const {
    return device_index_;
  }
  supaEvent_t event() const {
    return event_;
  }

  bool query() const;
  void record();
  void recordOnce(const SUPAStream& stream);
  void record(const SUPAStream& stream);
  void block(const SUPAStream& stream);
  float elapsed_time(const SUPAEvent& other) const;
  void synchronize() const;
  void ipc_handle(supaIpcEventHandle_t* handle);

  // supa do not support IpcEventHandle until now

 private:
  unsigned int flags_ = supaEventDisableTiming;
  bool is_created_ = false;
  bool was_recorded_ = false;
  c10::DeviceIndex device_index_ = -1;
  supaEvent_t event_ = nullptr;

  void createEvent(c10::DeviceIndex device_index);
  void moveHelper(SUPAEvent&& other);
};

} // namespace c10::supa
