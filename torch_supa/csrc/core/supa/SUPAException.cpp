/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/*
 * Copyright (c) 2023-2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/SUPADeviceAssertionHost.h"

#include <functional>

namespace c10::supa {

void c10_supa_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const int line_number,
    const bool include_device_assertions) {
  const auto supa_error = static_cast<supaError_t>(err);
  const auto supa_kernel_failure =
      include_device_assertions ? c10::supa::SUPAKernelLaunchRegistry::get_singleton_ref().has_failed() : false;

  if (C10_LIKELY(supa_error == supaSuccess && !supa_kernel_failure)) {
    return;
  }

  [[maybe_unused]] auto error_unused = supaGetLastError();

  std::string check_message;
#ifndef STRIP_ERROR_MESSAGES
  check_message.append("SUPA error: ");
  check_message.append(supaGetErrorString(supa_error));
  check_message.append(c10::supa::get_supa_check_suffix());
  check_message.append("\n");
  if (include_device_assertions) {
    check_message.append(c10_retrieve_device_side_assertion_info());
  } else {
    check_message.append(
        "Device-side assertions were explicitly omitted for this error check; the error probably arose while initializing the DSA handlers.");
  }
#endif

  TORCH_CHECK(false, check_message);
}

} // namespace c10::supa