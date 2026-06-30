/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/*
 * Copyright (c) 2023-2024 Shanghai Biren Technology Co., Ltd. All rights
 * reserved.
 */

#pragma once

#include <fmt/format.h>

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <sudnn/sudnn.h>
#include <supa.h>
#include <supa_runtime.h>
#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/core/supa/SUPAMiscFunctions.h"
#include "torch_supa/csrc/utils/logger/Logger.h"

namespace c10 {
class C10_SUPA_API SUPAError : public c10::Error {
  using Error::Error;
};

class SuDNNError : public c10::Error {
  using Error::Error;
};

} // namespace c10

#define C10_SUPA_CHECK(EXPR)                                        \
  do {                                                              \
    const supaError_t __err = EXPR;                                 \
    c10::supa::c10_supa_check_implementation(                       \
        static_cast<int32_t>(__err),                                \
        __FILE__,                                                   \
        __func__, /* Line number data type not well-defined between \
                      compilers, so we perform an explicit cast */  \
        static_cast<uint32_t>(__LINE__),                            \
        true);                                                      \
  } while (0)
// Indicates that a SUPA error is handled in a non-standard way
#define C10_SUPA_ERROR_HANDLED(EXPR) EXPR

// Intentionally ignore a SUPA error
#define C10_SUPA_IGNORE_ERROR(EXPR)                                   \
  do {                                                                \
    const supaError_t __err = EXPR;                                   \
    if (C10_UNLIKELY(__err != supaSuccess)) {                         \
      [[maybe_unused]] supaError_t error_unused = supaGetLastError(); \
    }                                                                 \
  } while (0)

// This should be used directly after every kernel launch to ensure
// the launch happened correctly and provide an early, close-to-source
// diagnostic if it didn't.
#define C10_SUPA_KERNEL_LAUNCH_CHECK() C10_SUPA_CHECK(supaGetLastError())

#define C10_SUPA_CHECK_WARN(EXPR)                                   \
  do {                                                              \
    supaError_t __err = EXPR;                                       \
    if (__err != supaSuccess) {                                     \
      auto error_unused C10_UNUSED = supaGetLastError();            \
      TORCH_SUPA_WARN("SUPA warning: ", supaGetErrorString(__err)); \
    }                                                               \
  } while (0)

#define C10_SUPA_DRIVER_CHECK(EXPR)                                                    \
  do {                                                                                 \
    SUresult __err = EXPR;                                                             \
    if (__err != SUPA_SUCCESS) {                                                       \
      const char* err_str;                                                             \
      SUresult get_error_str_err [[maybe_unused]] = suGetErrorString(__err, &err_str); \
      if (get_error_str_err != SUPA_SUCCESS) {                                         \
        TORCH_CHECK(false, "SUPA driver error: unknown error");                        \
      } else {                                                                         \
        TORCH_CHECK(false, "SUPA driver error: ", err_str);                            \
      }                                                                                \
    }                                                                                  \
  } while (0)

#define AT_SUPA_DRIVER_CHECK(EXPR)                                        \
  do {                                                                    \
    SUresult __err = EXPR;                                                \
    if (__err != SUPA_SUCCESS) {                                          \
      TORCH_CHECK(false, "SUPA driver error: ", static_cast<int>(__err)); \
    }                                                                     \
  } while (0)

#define AT_SUDNN_FRONTEND_CHECK(EXPR, ...)                                                       \
  do {                                                                                           \
    auto error_object = EXPR;                                                                    \
    if (!error_object.is_good()) {                                                               \
      TORCH_CHECK_WITH(SuDNNError, false, "suDNN Frontend error: ", error_object.get_message()); \
    }                                                                                            \
  } while (0)

#define AT_SUDNN_CHECK(EXPR, ...)                                                                         \
  do {                                                                                                    \
    sudnnStatus_t status = EXPR;                                                                          \
    if (status != SUDNN_STATUS_SUCCESS) {                                                                 \
      if (status == SUDNN_STATUS_NOT_SUPPORTED) {                                                         \
        TORCH_CHECK_WITH(                                                                                 \
            SuDNNError,                                                                                   \
            false,                                                                                        \
            "suDNN error: ",                                                                              \
            sudnnGetErrorString(status),                                                                  \
            ". This error may appear if you passed in a non-contiguous input.",                           \
            ##__VA_ARGS__);                                                                               \
      } else {                                                                                            \
        TORCH_CHECK_WITH(SuDNNError, false, "suDNN error: ", sudnnGetErrorString(status), ##__VA_ARGS__); \
      }                                                                                                   \
    }                                                                                                     \
  } while (0)

namespace c10::supa {

/// In the event of a SUPA failure, formats a nice error message about that
/// failure and also checks for device-side assertion failures
C10_SUPA_API void c10_supa_check_implementation(
    int32_t err,
    const char* filename,
    const char* function_name,
    int line_number,
    bool include_device_assertions);

} // namespace c10::supa
