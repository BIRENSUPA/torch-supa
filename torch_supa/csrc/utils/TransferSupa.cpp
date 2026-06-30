/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/*Copyright © 2026 Shanghai Biren Technology Co., Ltd. All rights reserved.*/

#include <string>

#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_strings.h>

namespace torch_supa::utils {
namespace {

thread_local bool enable_translate_type = false;

inline std::string type_name(const c10::Device& device) {
  auto type = device.type();
  return (enable_translate_type && type == at::kPrivateUse1) ? "cuda"
                                                             : c10::DeviceTypeName(type, /* lower case */ true);
}

PyObject* THPDevice_new_type(THPDevice* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(type_name(self->device));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDevice_slot_new_str(PyObject* self) {
  auto* device = reinterpret_cast<THPDevice*>(self);
  std::string str = type_name(device->device);
  if (device->device.has_index()) {
    str.push_back(':');
    str.append(std::to_string(device->device.index()));
  }
  return THPUtils_packString(str.c_str());
}

/**
 * @brief patch torch.device class
 *
 * @return true: patch successfully.
 * @return false
 */
bool patch_torch_device(void) {
  PyObject* torch_module = PyImport_ImportModule("torch._C");
  if (!torch_module) {
    return false;
  }

  PyObject* device_type_obj = PyObject_GetAttrString(torch_module, "device");
  Py_DECREF(torch_module);

  if (!device_type_obj || !PyType_Check(device_type_obj)) {
    Py_XDECREF(device_type_obj);
    return false;
  }

  PyTypeObject* device_type = (PyTypeObject*)device_type_obj;

  if (strcmp(device_type->tp_name, "torch.device") == 0) {
    // replace device.type and device.str()
    device_type->tp_str = (reprfunc)THPDevice_slot_new_str;

    static PyGetSetDef new_getset[] = {
        {"type", (getter)THPDevice_new_type, nullptr, nullptr, nullptr},
        {"index", (getter)device_type->tp_getset[1].get, nullptr, nullptr, nullptr},
        {nullptr}};
    device_type->tp_getset = new_getset;

    // reset flag and fields, call PyType_Ready() to rebuild its tp_dict.
    device_type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_VALID_VERSION_TAG;
    Py_XDECREF(device_type->tp_dict);
    device_type->tp_dict = nullptr;
    device_type->tp_getattro = nullptr;
    device_type->tp_setattro = nullptr;
    device_type->tp_init = nullptr;
    device_type->tp_base = nullptr;
    device_type->tp_mro = nullptr;
    PyType_Ready(device_type);
    PyType_Modified(device_type); // reset tp_version_tag for python cache.
  }
  Py_DECREF(device_type_obj);
  return true;
}
} // namespace
bool initDeviceWrap(void);
void transfer_device(void) {
  if (!initDeviceWrap()) {
    throw std::runtime_error("failed to patch ctor of c10::Device");
  }
  if (!patch_torch_device()) {
    throw std::runtime_error("failed to patch torch.device");
  }
}

void InitTransferSupaBindings(PyObject* module) {
  auto parent = py::handle(module).cast<py::module>();

  py::module m = parent.def_submodule("_transfer", "transfer_to_supa for device in cpp");

  m.def("device", &transfer_device, "transfer cuda to supa for torch.device()");
  m.def(
      "device_type", [](bool enable) { enable_translate_type = enable; }, "whether to transfer torch.device.type");
  m.def(
      "device_type_status", [](void) { return enable_translate_type; }, "whether device.type is tranferred.");
}
} // namespace torch_supa::utils
