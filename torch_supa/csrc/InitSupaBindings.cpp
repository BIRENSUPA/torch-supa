/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/Parallel.h>
#include <Python.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Generator.h>

#include "torch_supa/csrc/debugger/SignalHandler.h"
#include "torch_supa/csrc/profiler/init.h"
#include "torch_supa/csrc/supa/Module.h"
#ifdef USE_BCCL
#include "torch_supa/csrc/distributed/c10d.h"
#endif
#include <torch_supa/csrc/ipc/StorageSharing.h>

PyObject* module;

void AddPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods) {
  if (!vector.empty()) {
    // remove nullptr terminator
    vector.pop_back();
  }
  while (true) {
    vector.push_back(*methods);
    if (!methods->ml_name) {
      break;
    }
    methods++;
  }
}

void THBPStream_init(PyObject* module);
void THBPEvent_init(PyObject* module);
void THBPMemPool_init(PyObject* module);
void THBPGraph_init(PyObject* module);

PyMethodDef* THBPModule_get_methods();
PyMethodDef* SupaExtensionFunctions();
PyMethodDef* AutocastModeFunctions();
namespace torch_supa {

namespace supa {
void initSupartBindings(PyObject* module);
void initDebugFunctions(PyObject* module);
void initDynamoFunctions(PyObject* module);

namespace shared {
void InitBrtxBindings(PyObject* module);
}
} // namespace supa

namespace utils {
void InitTransferSupaBindings(PyObject* module);
}
} // namespace torch_supa

static std::vector<PyMethodDef> methods;
extern "C" PyObject* initModule() {
  AddPyMethodDefs(methods, THBPModule_get_methods());
  AddPyMethodDefs(methods, SupaExtensionFunctions());
  AddPyMethodDefs(methods, AutocastModeFunctions());
  AddPyMethodDefs(methods, torch_supa::profiler::profiler_functions());
  AddPyMethodDefs(methods, torch_supa::reductions::storage_ipc_functions());

#ifdef USE_BCCL
  AddPyMethodDefs(methods, c10d::supa::InitFunctions());
#endif

  static struct PyModuleDef torch_supa_module = {PyModuleDef_HEAD_INIT, "torch_supa._C", nullptr, -1, methods.data()};
  module = PyModule_Create(&torch_supa_module);

  THBPStream_init(module);
  THBPEvent_init(module);
  THBPMemPool_init(module);
  THBPGraph_init(module);
  RegisterSupaDeviceProperties(module);
  RegisterSupaPluggableAllocator(module);
  InitSupaModuleBindings(module);
  torch_supa::supa::initSupartBindings(module);
  torch_supa::supa::initDebugFunctions(module);
  torch_supa::supa::initDynamoFunctions(module);
  torch_supa::utils::initSignalHandler();
  torch_supa::utils::InitTransferSupaBindings(module);
  torch_supa::supa::shared::InitBrtxBindings(module);
  return module;
}

PyMODINIT_FUNC PyInit__C(void) {
  return initModule();
}
