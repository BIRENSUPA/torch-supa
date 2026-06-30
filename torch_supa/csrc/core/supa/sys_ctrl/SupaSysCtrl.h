/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once

#include <c10/core/Device.h>
#include <mutex>
#include <vector>
#include "torch_supa/csrc/core/supa/SUPAMacros.h"

namespace c10::supa {

class SupaSysCtrl {
 public:
  C10_SUPA_API static SupaSysCtrl& GetInstance();

  C10_SUPA_API void supaInit();
  C10_SUPA_API static void supaInit(c10::DeviceIndex device_index);

 private:
  SupaSysCtrl() {
    supaInit();
  };

  SupaSysCtrl(const SupaSysCtrl&) = delete;
  SupaSysCtrl(SupaSysCtrl&&) = delete;
  SupaSysCtrl& operator=(const SupaSysCtrl&) = delete;
  SupaSysCtrl& operator=(SupaSysCtrl&&) = delete;
  ~SupaSysCtrl() = default;

  std::once_flag init_flag_;
  bool init_label_ = false;
  int numDevices_ = 0;
  std::vector<std::vector<int>> p2pAccessEnabled_;
};

} // namespace c10::supa