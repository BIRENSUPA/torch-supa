/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once

#include <supa_driver.h>
#ifndef NO_SUPA_RT_HEADER
#include <torch_supa/csrc/core/supa/SUPAException.h>
#include <torch_supa/csrc/core/supa/SUPAFunctions.h>
#endif
#include <torch_supa/csrc/core/supa/SUPAMacros.h>

namespace c10 {
namespace supa {

using CaptureId_t = unsigned long long;
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

#ifndef NO_SUPA_RT_HEADER
using CaptureStatus = supaStreamCaptureStatus;
C10_SUPA_API CaptureStatus currentStreamCaptureStatusMayInitCtx();

inline CaptureStatus currentStreamCaptureStatus() {
  // don't create a context if we don't have to
  if (hasPrimaryContext(c10::supa::current_device())) {
    return currentStreamCaptureStatusMayInitCtx();
  }
  return CaptureStatus::supaStreamCaptureStatusNone;
}

inline void assertNotCapturing(std::string attempt) {
  auto status = currentStreamCaptureStatus();
  TORCH_CHECK(
      status == CaptureStatus::supaStreamCaptureStatusNone,
      attempt,
      " during SUPA graph capture. If you need this call to be captured, "
      "please file an issue. "
      "Current supaStreamCaptureStatus: ",
      status);
}

inline void errorIfCapturingSudnnBenchmark(std::string version_specific) {
  auto status = currentStreamCaptureStatus();
  TORCH_CHECK(
      status == CaptureStatus::supaStreamCaptureStatusNone,
      "Current supaStreamCaptureStatus: ",
      status,
      "\nCapturing ",
      version_specific,
      "is prohibited. Possible causes of this error:\n"
      "1. No warmup iterations occurred before capture.\n"
      "2. The convolutions you're trying to capture use dynamic shapes, "
      "in which case capturing them is generally prohibited.");
}

struct C10_SUPA_API SUPAStreamCaptureModeGuard {
  SUPAStreamCaptureModeGuard(supaStreamCaptureMode desired) : strictness_(desired) {
    C10_SUPA_CHECK(supaThreadExchangeStreamCaptureMode(&strictness_));
  }
  SUPAStreamCaptureModeGuard(const SUPAStreamCaptureModeGuard&) = delete;
  SUPAStreamCaptureModeGuard(SUPAStreamCaptureModeGuard&&) = delete;
  SUPAStreamCaptureModeGuard& operator=(const SUPAStreamCaptureModeGuard&) = delete;
  SUPAStreamCaptureModeGuard& operator=(SUPAStreamCaptureModeGuard&&) = delete;
  ~SUPAStreamCaptureModeGuard() {
    C10_SUPA_CHECK_WARN(supaThreadExchangeStreamCaptureMode(&strictness_));
  }

 private:
  supaStreamCaptureMode strictness_;
};

#endif

} // namespace supa
} // namespace c10
