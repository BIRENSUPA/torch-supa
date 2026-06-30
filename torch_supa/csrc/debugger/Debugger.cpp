/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */
#include <malloc.h>
#include <torch/csrc/python_headers.h>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include "torch_supa/csrc/utils/EnvConfig.h"

namespace torch_supa {
namespace supa {

using namespace torch_supa::utils;

// Expand volatile entries according to environment variables defined in EnvConfig.h
#define VOLATILE_ENV_VAR(_)

#define DECLARE_AND_REGISTER_ENV_SET_FUNC(_C_FUNC, _PYTHON_FUNC)                             \
  PyObject* THCPModule_supaDebugSet##_C_FUNC(PyObject* self, PyObject* args) {               \
    HANDLE_TH_ERRORS                                                                         \
                                                                                             \
    if (!PyBool_Check(args)) {                                                               \
      THPUtils_invalidArguments(args, nullptr, "set_" _PYTHON_FUNC, 1, "(bool is_enable);"); \
      return nullptr;                                                                        \
    }                                                                                        \
                                                                                             \
    EnvConfig::Set##_C_FUNC(args == Py_True ? true : false);                                 \
                                                                                             \
    END_HANDLE_TH_ERRORS                                                                     \
    Py_RETURN_NONE;                                                                          \
  }

#define DECLARE_AND_REGISTER_ENV_GET_FUNC(_C_FUNC, _PYTHON_FUNC)                       \
  static PyObject* THCPModule_supaDebugIs##_C_FUNC(PyObject* self, PyObject* noargs) { \
    HANDLE_TH_ERRORS                                                                   \
    if (EnvConfig::Is##_C_FUNC()) {                                                    \
      Py_RETURN_TRUE;                                                                  \
    } else {                                                                           \
      Py_RETURN_FALSE;                                                                 \
    }                                                                                  \
    END_HANDLE_TH_ERRORS                                                               \
  }

#define DECLARE_AND_REGISTER_ENV_RESET_FUNC(_C_FUNC, _PYTHON_FUNC)               \
  PyObject* THCPModule_supaDebugReset##_C_FUNC(PyObject* self, PyObject* args) { \
    HANDLE_TH_ERRORS                                                             \
    EnvConfig::Reset##_C_FUNC();                                                 \
                                                                                 \
    END_HANDLE_TH_ERRORS                                                         \
    Py_RETURN_NONE;                                                              \
  }

VOLATILE_ENV_VAR(DECLARE_AND_REGISTER_ENV_SET_FUNC)
VOLATILE_ENV_VAR(DECLARE_AND_REGISTER_ENV_GET_FUNC)
VOLATILE_ENV_VAR(DECLARE_AND_REGISTER_ENV_RESET_FUNC)

#undef DECLARE_AND_REGISTER_ENV_SET_FUNC
#undef DECLARE_AND_REGISTER_ENV_GET_FUNC
#undef DECLARE_AND_REGISTER_ENV_RESET_FUNC

#define DECLARE_PYTHON_METHODS(_C_FUNC, _PYTHON_FUNC)                                                            \
  {"_supa_is_" _PYTHON_FUNC, (PyCFunction)THCPModule_supaDebugIs##_C_FUNC, METH_NOARGS | METH_STATIC, nullptr},  \
      {"_supa_set_" _PYTHON_FUNC, (PyCFunction)THCPModule_supaDebugSet##_C_FUNC, METH_O | METH_STATIC, nullptr}, \
      {"_supa_reset_" _PYTHON_FUNC,                                                                              \
       (PyCFunction)THCPModule_supaDebugReset##_C_FUNC,                                                          \
       METH_NOARGS | METH_STATIC,                                                                                \
       nullptr},

#undef VOLATILE_ENV_VAR
#undef DECLARE_PYTHON_METHODS

static PyTypeObject THBPDebugFunctions = {
    PyVarObject_HEAD_INIT(NULL, 0) "torch_supa._C._DebugFunctionsClass", /* tp_name */
    0, /* tp_basicsize */
    0, /* tp_itemsize */
    0, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    0, /* tp_getattr */
    0, /* tp_setattr */
    0, /* tp_reserved */
    0, /* tp_repr */
    0, /* tp_as_number */
    0, /* tp_as_sequence */
    0, /* tp_as_mapping */
    0, /* tp_hash  */
    0, /* tp_call */
    0, /* tp_str */
    0, /* tp_getattro */
    0, /* tp_setattro */
    0, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    NULL, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    NULL, /* tp_methods */
    0, /* tp_members */
    0, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    0, /* tp_init */
    0, /* tp_alloc */
    0 /* tp_new */
};

static PyObject* THBPDebugFunctionsModule = NULL;

void initDebugFunctions(PyObject* module) {
  if (PyType_Ready(&THBPDebugFunctions) < 0) {
    throw python_error();
  }
  Py_INCREF(&THBPDebugFunctions);

  // Steals
  Py_INCREF(&THBPDebugFunctions);
  if (PyModule_AddObject(module, "_Debugger", reinterpret_cast<PyObject*>(&THBPDebugFunctions)) < 0) {
    throw python_error();
  }
  // PyType_GenericNew returns a new reference
  THBPDebugFunctionsModule = PyType_GenericNew(&THBPDebugFunctions, Py_None, Py_None);
  // PyModule_AddObject steals a reference
  if (PyModule_AddObject(module, "_VariableFunctions", THBPDebugFunctionsModule) < 0) {
    throw python_error();
  }
}

} // namespace supa
} // namespace torch_supa
