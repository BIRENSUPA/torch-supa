/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#ifndef THNP_SUPA_MODULE_INC
#define THNP_SUPA_MODULE_INC
#include <Python.h>
#include "torch_supa/csrc/core/supa/SUPAMacros.h"

TORCH_SUPA_API void InitSupaModuleBindings(PyObject *module);
TORCH_SUPA_API void RegisterSupaDeviceProperties(PyObject *module);
TORCH_SUPA_API void RegisterSupaPluggableAllocator(PyObject *module);

#endif // THBP_EVENT_INC
