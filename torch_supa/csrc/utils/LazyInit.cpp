/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#include <mutex>

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/python_headers.h"
#include "torch/csrc/utils/object_ptr.h"

#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/core/supa/sys_ctrl/SupaSysCtrl.h"
#include "torch_supa/csrc/utils/LazyInit.h"

namespace torch_supa {
namespace utils {

static bool supa_run_yet = false;

std::once_flag thb_init;

bool is_call_from_python() {
  return Py_IsInitialized();
}

void supa_lazy_init() {
  pybind11::gil_scoped_acquire g;

  // Protected by the GIL.  We don't use call_once because under ASAN it
  // has a buggy implementation that deadlocks if an instance throws an
  // exception.  In any case, call_once isn't necessary, because we
  // have taken a lock.
  if (!supa_run_yet) {
    auto module = THPObjectPtr(PyImport_ImportModule("torch_supa.supa"));
    if (!module) {
      throw python_error();
    }
    auto res = THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
    if (!res) {
      throw python_error();
    }
    supa_run_yet = true;
  }
}

void supa_set_run_yet_variable_to_false() {
  supa_run_yet = false;
}

void lazyInitSUPA() {
  c10::supa::SupaSysCtrl::GetInstance().supaInit();
}

} // namespace utils
} // namespace torch_supa
