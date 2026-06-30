/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */
#include <ATen/autocast_mode.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/utils/python_arg_parsing.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_stub.h>

#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/core/supa/TorchVersion.h"

namespace torch_supa {
namespace autocast {

static PyObject* set_autocast_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK_TYPE(PyBool_Check(arg), "enabled must be a bool (got ", Py_TYPE(arg)->tp_name, ")");
#if TORCH_VER >= TORCH_2_4_0
  at::autocast::set_autocast_enabled(at::kPrivateUse1, arg == Py_True);
#else
  at::autocast::set_privateuseone_enabled(arg == Py_True);
#endif
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* is_autocast_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
#if TORCH_VER >= TORCH_2_4_0
  if (at::autocast::is_autocast_enabled(at::kPrivateUse1))
#else
  if (at::autocast::is_privateuseone_enabled())
#endif
  {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;

  END_HANDLE_TH_ERRORS
}

static PyObject* set_autocast_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK_TYPE(THPDtype_Check(arg), "enabled must be a torch.dtype (got ", Py_TYPE(arg)->tp_name, ")");
  at::ScalarType targetType = reinterpret_cast<THPDtype*>(arg)->scalar_type;
#if TORCH_VER >= TORCH_2_4_0
  at::autocast::set_autocast_dtype(at::kPrivateUse1, targetType);
#else
  at::autocast::set_autocast_privateuseone_dtype(targetType);
#endif
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* get_autocast_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
#if TORCH_VER >= TORCH_2_4_0
  at::ScalarType current_dtype = at::autocast::get_autocast_dtype(at::kPrivateUse1);
#else
  at::ScalarType current_dtype = at::autocast::get_autocast_privateuseone_dtype();
#endif
  auto* dtype = (PyObject*)torch::getTHPDtype(current_dtype);
  Py_INCREF(dtype);
  return dtype;
  END_HANDLE_TH_ERRORS
}

/**
 * @brief rewrite torch.set_autocast_enable to support CUDA -> SUPA
 *
 * @return PyObject* return None.
 */
static PyObject* torch_set_autocast_enabled(PyObject* _unused, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {"set_autocast_enabled(std::string_view device_type, bool enabled)",
       "set_autocast_enabled(bool enabled)"}); // this signature is depracated.
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  // Set at::kPrivateUse1 as default value to prevent BC-breaking changes.
  at::DeviceType device_type = at::kPrivateUse1;
  int enabled_id = 0;
  if (r.idx == 0) {
    device_type = at::Device(r.string(0)).type();
    if (device_type == at::kCUDA) {
      device_type = at::kPrivateUse1;
    }
    enabled_id = 1;
  }
  auto enabled = r.toBool(enabled_id);
  at::autocast::set_autocast_enabled(device_type, enabled);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

/**
 * @brief rewrite torch.is_autocast_enable to support CUDA -> SUPA
 *
 * @return PyObject* return None.
 */
static PyObject* torch_is_autocast_enabled(PyObject* _unused, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {"is_autocast_enabled(std::string_view device_type)", "is_autocast_enabled()"}); // this signature is depracated.
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  // Set at::kPrivateUse1 as default value to prevent BC-breaking changes.
  at::DeviceType device_type = at::kPrivateUse1;
  if (r.idx == 0) {
    device_type = at::Device(r.string(0)).type();
    if (device_type == at::kCUDA) {
      device_type = at::kPrivateUse1;
    }
  }
  if (at::autocast::is_autocast_enabled(device_type)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;

  END_HANDLE_TH_ERRORS
}

// autocast methods on torch_supa._C
static PyMethodDef methods[] = { // NOLINT
    {"set_autocast_enabled", set_autocast_enabled, METH_O, nullptr},
    {"is_autocast_enabled", is_autocast_enabled, METH_NOARGS, nullptr},
    {"set_autocast_dtype", set_autocast_dtype, METH_O, nullptr},
    {"get_autocast_dtype", get_autocast_dtype, METH_NOARGS, nullptr},
    {"torch_set_autocast_enabled",
     castPyCFunctionWithKeywords(torch_set_autocast_enabled),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"torch_is_autocast_enabled",
     castPyCFunctionWithKeywords(torch_is_autocast_enabled),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};

} // namespace autocast
} // namespace torch_supa

TORCH_SUPA_API PyMethodDef* AutocastModeFunctions() {
  return torch_supa::autocast::methods;
}
