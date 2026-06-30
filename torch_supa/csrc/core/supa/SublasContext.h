/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <c10/core/Allocator.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <sublasLt.h>
#include <torch_supa/csrc/core/supa/SUPAMacros.h>
#include <map>

namespace at::sublas {
C10_SUPA_EXPORT const char* _sublasGetErrorEnum(sublasStatus_t error);
} // namespace at::sublas

#define AT_SUBLAS_CHECK(EXPR)                   \
  do {                                          \
    sublasStatus_t __err = EXPR;                \
    TORCH_CHECK(                                \
        __err == SUBLAS_STATUS_SUCCESS,         \
        "SUPA error: ",                         \
        at::sublas::_sublasGetErrorEnum(__err), \
        " when calling `" #EXPR "`");           \
  } while (0)

namespace at::supa {
/* Handles */
TORCH_SUPA_API sublasHandle_t getCurrentSuBlasHandle();
TORCH_SUPA_API sublasLtHandle_t getCurrentSuBlasLtHandle();
TORCH_SUPA_API void clearSublasWorkspaces();
TORCH_SUPA_API std::map<std::tuple<void*, void*>, at::DataPtr>& sublas_handle_stream_to_workspace();
TORCH_SUPA_API size_t getChosenWorkspaceSize();
} // namespace at::supa