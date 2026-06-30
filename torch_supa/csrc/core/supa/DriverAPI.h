/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once
#include <supa_driver.h>
#define BRML_NO_UNVERSIONED_FUNC_DEFS
#include <brml.h>

#include <c10/util/Exception.h>

#define C10_SUPA_DRIVER_CHECK(EXPR)                                                                                  \
  do {                                                                                                               \
    SUresult __err = EXPR;                                                                                           \
    if (__err != SUPA_SUCCESS) {                                                                                     \
      const char* err_str;                                                                                           \
      SUresult get_error_str_err [[maybe_unused]] = c10::supa::DriverAPI::get()->suGetErrorString_(__err, &err_str); \
      if (get_error_str_err != SUPA_SUCCESS) {                                                                       \
        TORCH_CHECK(false, "SUPA driver error: unknown error");                                                      \
      } else {                                                                                                       \
        TORCH_CHECK(false, "SUPA driver error: ", err_str);                                                          \
      }                                                                                                              \
    }                                                                                                                \
  } while (0)

#define C10_SUPA_DRIVER_CHECK_GOTO(EXPR, NEXT)                                                                       \
  do {                                                                                                               \
    SUresult __err = EXPR;                                                                                           \
    if (__err != SUPA_SUCCESS) {                                                                                     \
      const char* err_str;                                                                                           \
      SUresult get_error_str_err [[maybe_unused]] = c10::supa::DriverAPI::get()->suGetErrorString_(__err, &err_str); \
      if (get_error_str_err != SUPA_SUCCESS) {                                                                       \
        TORCH_WARN("SUPA driver error: unknown error");                                                              \
      } else {                                                                                                       \
        TORCH_WARN("SUPA driver error: ", err_str);                                                                  \
      }                                                                                                              \
      goto NEXT;                                                                                                     \
    }                                                                                                                \
  } while (0)

// The integer in the second column specifies the requested SUPA Driver API
// version. The dynamic loader will accept a driver with a newer version, but it
// ensures that the requested symbol exists in *at least* the specified version
// or earlier.

// Keep these requested versions as low as possible to maximize compatibility
// across different driver versions.

// Why do we pin to an older version instead of using the latest?
// If a user installs a newer driver, blindly resolving the symbol may bind to a
// newer version of the function with different behavior, potentially breaking
// PyTorch.

#define C10_LIBSUPA_DRIVER_API_REQUIRED(_) \
  _(suDeviceGetAttribute, 12000)           \
  _(suMemAddressReserve, 12000)            \
  _(suMemRelease, 12000)                   \
  _(suMemMap, 12000)                       \
  _(suMemAddressFree, 12000)               \
  _(suMemSetAccess, 12000)                 \
  _(suMemUnmap, 12000)                     \
  _(suMemCreate, 12000)                    \
  _(suMemGetAllocationGranularity, 12000)  \
  _(suMemExportToShareableHandle, 12000)   \
  _(suMemImportFromShareableHandle, 12000) \
  _(suMemsetD32Async, 12000)               \
  _(suStreamWriteValue32, 12000)           \
  _(suGetErrorString, 12000)

#define C10_LIBSUPA_DRIVER_API_OPTIONAL(_) \
  _(suCtxFromGreenCtx, 12080)              \
  _(suCtxGetCurrent, 12080)                \
  _(suCtxPopCurrent, 12080)                \
  _(suCtxPushCurrent, 12080)               \
  _(suCtxSetCurrent, 12080)                \
  _(suGreenCtxCreate, 12080)               \
  _(suGreenCtxDestroy, 12080)              \
  _(suDevSmResourceSplitByCount, 12080)    \
  _(suDeviceGet, 12080)                    \
  _(suDeviceGetDevResource, 12080)         \
  _(suDevResourceGenerateDesc, 12080)      \
  _(suMulticastAddDevice, 12030)           \
  _(suMulticastBindMem, 12030)             \
  _(suMulticastCreate, 12030)              \
  _(suMulticastUnbind, 12030)

#define C10_BRML_DRIVER_API(_)            \
  _(brmlInit_v2)                          \
  _(brmlDeviceGetHandleByPciBusId_v2)     \
  _(brmlDeviceGetBLinkRemoteDeviceType)   \
  _(brmlDeviceGetBLinkRemotePciInfo_v2)   \
  _(brmlDeviceGetComputeRunningProcesses) \
  _(brmlSystemGetSupaDriverVersion_v2)

#define C10_BRML_DRIVER_API_OPTIONAL(_) _(brmlDeviceGetGpuFabricInfoV)

namespace c10::supa {

struct DriverAPI {
#define CREATE_MEMBER_VERSIONED(name, version) decltype(&name) name##_;
#define CREATE_MEMBER(name) decltype(&name) name##_;
  C10_LIBSUPA_DRIVER_API_REQUIRED(CREATE_MEMBER_VERSIONED)
  C10_LIBSUPA_DRIVER_API_OPTIONAL(CREATE_MEMBER_VERSIONED)
  C10_BRML_DRIVER_API(CREATE_MEMBER)
  C10_BRML_DRIVER_API_OPTIONAL(CREATE_MEMBER)
#undef CREATE_MEMBER_VERSIONED
#undef CREATE_MEMBER

  static DriverAPI* get();
  static void* get_brml_handle();
};

} // namespace c10::supa
