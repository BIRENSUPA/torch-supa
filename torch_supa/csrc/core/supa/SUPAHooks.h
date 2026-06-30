/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include "torch_supa/csrc/core/supa/TorchVersion.h"

namespace c10::supa {

struct TORCH_API SUPAHooks : public at::PrivateUse1HooksInterface {
  ~SUPAHooks() override = default;
  SUPAHooks() = default;
  SUPAHooks(const SUPAHooks&) = delete;
  SUPAHooks(SUPAHooks&&) = delete;
  SUPAHooks& operator=(const SUPAHooks&) = delete;
  SUPAHooks& operator=(SUPAHooks&&) = delete;

#if TORCH_VER >= TORCH_2_8_0
  bool isBuilt() const override {
    return true;
  }

  bool isAvailable() const override;
#endif

#if TORCH_VER >= TORCH_2_5_0
  const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) const override;
#else
  const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) override;
#endif
  Device getDeviceFromPtr(void* data) const override;
#if TORCH_VER >= TORCH_2_3_0
  Allocator* getPinnedMemoryAllocator() const override;
  bool hasPrimaryContext(c10::DeviceIndex device_index) const override;
  void resizePrivateUse1Bytes(const c10::Storage& storage, size_t newsize) const override;
#endif

#if TORCH_VER >= TORCH_2_5_0
  bool isPinnedPtr(const void* data) const override;
#endif

#if TORCH_VER >= TORCH_2_6_0
  void init() const override;
#elif TORCH_VER >= TORCH_2_3_0
  void initPrivateUse1() const override;
#endif
};

struct TORCH_API SUPAHooksArgs : public at::PrivateUse1HooksArgs {};

// register to PrivateUse1HooksInterface
at::PrivateUse1HooksInterface* get_supa_hooks();
} // namespace c10::supa
