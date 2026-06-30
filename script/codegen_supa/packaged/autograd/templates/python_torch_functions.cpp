#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

// Python bindings for torch.* functions implemented through ATen.
//
// The functions are bound as static methods on a class
// torch._C._VariableFunctions which is also aliased as Variable._torch
// and also copied into 'torch' module.

#include <Python.h>

// Undefine the copysign macro so that at::copysign works as intended with MSVC
// https://github.com/python/cpython/blob/c60394c7fc9cc09b16e9675a3eeb5844b6d8523f/PC/pyconfig.h#L196
#ifdef _MSC_VER
#undef copysign
#endif // _MSC_VER

#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/device_lazy_init.h"

#include <ATen/core/Tensor.h>


#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/aten/core/TorchVersion.h"
/*
$ops_headers
*/

using at::Tensor;
using at::Device;
using at::Layout;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using at::OptionalDeviceGuard;
using at::DeviceGuard;
using at::TensorOptions;
using at::IntArrayRef;
using at::Generator;
using at::TensorList;
using at::Dimname;
using at::DimnameList;
using at::ArrayRef;

using namespace torch;
using namespace torch::autograd::utils;


// NOTE: See [Sharded File] comment in VariableType

// The symbol visibility of libtorch_pytorch is set to hidden, starting from version 2.7.0
#if defined(TORCH_2_7_0) && TORCH_VER >= TORCH_2_7_0
namespace torch {
// Combines self and args into one tuple.
static auto combine_self_args(PyObject* self, PyObject* args) -> py::tuple {
  if (args == nullptr) {
    return py::make_tuple(py::handle(self));
  } else if (self == nullptr) {
    return py::reinterpret_borrow<py::tuple>(args);
  }

  auto py_args = py::reinterpret_borrow<py::tuple>(args);
  size_t n = py_args.size();
  auto args_ = py::tuple(n + 1);
  args_[0] = py::handle(self);
  for (const auto i : c10::irange(n)) {
    args_[i + 1] = py_args[i];
  }
  return args_;
}

auto handle_torch_function(
    PythonArgs& r,
    PyObject* self,
    PyObject* args,
    PyObject* kwargs,
    PyObject* torch_api,
    const char* module_name,
    const char* func_name_override) -> PyObject* {
  py::object torch_api_function = PyObject_FastGetAttrString(
      torch_api,
      (char*)(func_name_override ? func_name_override
                                : r.get_func_name().c_str()));
  TORCH_INTERNAL_ASSERT(
      torch_api_function.ptr() != nullptr, "torch API function must exist");
  py::tuple args_ = combine_self_args(self, args);
  return handle_torch_function_no_python_arg_parser(
      r.overloaded_args,
      args_.ptr(),
      kwargs,
      r.get_func_name().c_str(),
      torch_api_function.ptr(),
      module_name);
}
} // namespace torch
#endif

namespace torch_supa::autograd {

// generated forward declarations start here

${py_forwards}

static PyMethodDef torch_functions_shard[] = {
  ${py_method_defs}
  {nullptr, nullptr, 0, nullptr},
};

PyObject* THPTorchSUPAFunctionsModule = nullptr;

/**
 * @brief init module for backend_agnostic OPs.
 *
 */
void initTorchFunctions${shard_id}(PyObject* module) {
  static PyTypeObject THPTorchFunctions = {
    PyVarObject_HEAD_INIT(NULL, 0) "torch_supa._C._BackendAgnosticClass", /* tp_name */
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
    torch_functions_shard, /* tp_methods */
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
  if (PyType_Ready(&THPTorchFunctions) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPTorchFunctions);

  Py_INCREF(&THPTorchFunctions);
  if (PyModule_AddObject(module, "_BackendAgnostic", reinterpret_cast<PyObject*>(&THPTorchFunctions)) < 0) {
    throw python_error();
  }

  // PyType_GenericNew returns a new reference
  THPTorchSUPAFunctionsModule =
      PyType_GenericNew(&THPTorchFunctions, Py_None, Py_None);
  // PyModule_AddObject steals a reference
  if (PyModule_AddObject(
          module, "_BackendAgnosticFunctions", THPTorchSUPAFunctionsModule) < 0) {
    throw python_error();
  }


}

// generated methods start here

${py_methods}

} // namespace torch::autograd
