/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/*Copyright © 2026 Shanghai Biren Technology Co., Ltd. All rights reserved.*/
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>

#include "torch_supa/csrc/aten/common/EmptyTensor.h"

namespace torch_supa {

namespace supa {

template <typename T>
static void unwrap_size_tuple(PyObject* obj, T& output) {
  TORCH_CHECK(PyTuple_CheckExact(obj));
  size_t len = PyTuple_GET_SIZE(obj);
  output.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(obj, i));
    TORCH_CHECK(result >= 0);
    output.emplace_back(result);
  }
}

template <typename T>
static void _parse_empty_strided_args(PyObject* args, T& sizes, T& strides, at::ScalarType& dtype) {
  TORCH_CHECK(PyTuple_CheckExact(args));
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 3);
  // note PyTuple_GET_ITEM returns a borrowed ref, so no need for refcounts
  unwrap_size_tuple(PyTuple_GET_ITEM(args, 0), sizes);
  unwrap_size_tuple(PyTuple_GET_ITEM(args, 1), strides);
  PyObject* py_dtype = PyTuple_GET_ITEM(args, 2);
  TORCH_CHECK(THPDtype_Check(py_dtype));
  dtype = reinterpret_cast<THPDtype*>(py_dtype)->scalar_type;
}

static PyObject* _empty_strided_supa(PyObject* dummy, PyObject* args) {
  // at::empty_strided is surprising slow.  This is lower-overhead.
  HANDLE_TH_ERRORS;
  at::SmallVector<int64_t, 8> sizes;
  at::SmallVector<int64_t, 8> strides;
  at::ScalarType dtype{at::ScalarType::Undefined};
  _parse_empty_strided_args(args, sizes, strides, dtype);
  return THPVariable_Wrap(at::detail::empty_strided_supa(sizes, strides, dtype, at::kPrivateUse1));
  END_HANDLE_TH_ERRORS;
}

static PyMethodDef DynamoFunctions[] = {
    {"_empty_strided_supa", _empty_strided_supa, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef DynamoModule =
    {PyModuleDef_HEAD_INIT, "torch_supa._C._dynamo", "api for speed up dynamo.", -1, DynamoFunctions};

void initDynamoFunctions(PyObject* module) {
  PyObject* dynamo_module = PyModule_Create(&DynamoModule);
  if (!dynamo_module) {
    throw python_error();
  }
  if (PyModule_AddObject(module, "_dynamo", dynamo_module) < 0) {
    Py_DECREF(dynamo_module);
    throw python_error();
  }
}
} // namespace supa
} // namespace torch_supa
