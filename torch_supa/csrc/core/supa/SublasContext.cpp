/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/core/supa/SublasContext.h"

namespace at::sublas {

C10_SUPA_EXPORT const char* _sublasGetErrorEnum(sublasStatus_t error) {
  if (error == SUBLAS_STATUS_SUCCESS) {
    return "SUBLAS_STATUS_SUCCESS";
  }
  if (error == SUBLAS_STATUS_NOT_INITIALIZED) {
    return "SUBLAS_STATUS_NOT_INITIALIZED";
  }
  if (error == SUBLAS_STATUS_ALLOC_FAILED) {
    return "SUBLAS_STATUS_ALLOC_FAILED";
  }
  if (error == SUBLAS_STATUS_INVALID_VALUE) {
    return "SUBLAS_STATUS_INVALID_VALUE";
  }
  if (error == SUBLAS_STATUS_ARCH_MISMATCH) {
    return "SUBLAS_STATUS_ARCH_MISMATCH";
  }
  if (error == SUBLAS_STATUS_MAPPING_ERROR) {
    return "SUBLAS_STATUS_MAPPING_ERROR";
  }
  if (error == SUBLAS_STATUS_EXECUTION_FAILED) {
    return "SUBLAS_STATUS_EXECUTION_FAILED";
  }
  if (error == SUBLAS_STATUS_INTERNAL_ERROR) {
    return "SUBLAS_STATUS_INTERNAL_ERROR";
  }
  if (error == SUBLAS_STATUS_NOT_SUPPORTED) {
    return "SUBLAS_STATUS_NOT_SUPPORTED";
  }
#ifdef SUBLAS_STATUS_LICENSE_ERROR
  if (error == SUBLAS_STATUS_LICENSE_ERROR) {
    return "SUBLAS_STATUS_LICENSE_ERROR";
  }
#endif
  return "<unknown>";
}

} // namespace at::sublas