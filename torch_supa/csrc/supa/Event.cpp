/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <pybind11/pybind11.h>
#include <structmember.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include "torch_supa/csrc/core/supa/SUPAGuard.h"
#include "torch_supa/csrc/supa/Event.h"
#include "torch_supa/csrc/supa/Stream.h"

PyObject* THBPEventClass = nullptr;

static PyObject* THBPEvent_pynew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  unsigned char enable_timing = 0;
  unsigned char blocking = 0;
  unsigned char interprocess = 0;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* kwlist[] = {"enable_timing", "blocking", "interprocess", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|bbb",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &enable_timing,
          &blocking,
          &interprocess)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THBPEvent* self = (THBPEvent*)ptr.get();
  unsigned int flags = (blocking ? supaEventBlockingSync : supaEventDefault) |
      (enable_timing ? supaEventDefault : supaEventDisableTiming) |
      (interprocess ? supaEventInterprocess : supaEventDefault);

  new (&self->supa_event) c10::supa::SUPAEvent(flags);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPEvent_from_ipc_handle(PyObject* _type, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto* type = (PyTypeObject*)_type;

  static torch::PythonArgParser parser({
      "from_ipc_handle(Device device, std::string ipc_handle)",
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  at::Device device = r.device(0);
  std::string handle_string = r.string(1);

  TORCH_CHECK(
      handle_string.size() == sizeof(supaIpcEventHandle_t),
      "supaIpcEventHandle_t expects byte-like object of size ",
      sizeof(supaIpcEventHandle_t),
      ", but got ",
      handle_string.size());
  TORCH_CHECK(
      device.type() == at::kPrivateUse1,
      "Event can only be created on "
      "SUPA devices, but got device type ",
      device.type())

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }
  THBPEvent* self = (THBPEvent*)ptr.get();

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  supaIpcEventHandle_t handle;
  std::memcpy(&handle, handle_string.c_str(), handle_string.size());
  new (&self->supa_event) at::supa::SUPAEvent(device.index(), &handle);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THBPEvent_dealloc(THBPEvent* self) {
  {
    pybind11::gil_scoped_release no_gil{};
    self->supa_event.~SUPAEvent();
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THBPEvent_get_supa_event(THBPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->supa_event.event());
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPEvent_get_device(THBPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  c10::optional<at::Device> device = self->supa_event.device();
  if (!device) {
    Py_RETURN_NONE;
  }
  return THPDevice_New(device.value());
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPEvent_record(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS {
    auto* self = (THBPEvent*)_self;
    auto* stream = (THBPStream*)_stream;
    pybind11::gil_scoped_release no_gil{};
    self->supa_event.record(stream->supa_stream);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPEvent_wait(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS {
    auto* self = (THBPEvent*)_self;
    auto* stream = (THBPStream*)_stream;
    pybind11::gil_scoped_release no_gil{};
    self->supa_event.block(stream->supa_stream);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPEvent_query(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto* self = (THBPEvent*)_self;
  return PyBool_FromLong(self->supa_event.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPEvent_elapsed_time(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  auto* self = (THBPEvent*)_self;
  auto* other = (THBPEvent*)_other;
  return PyFloat_FromDouble(self->supa_event.elapsed_time(other->supa_event));
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPEvent_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    auto* self = (THBPEvent*)_self;
    pybind11::gil_scoped_release no_gil{};
    self->supa_event.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPEvent_ipc_handle(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto* self = (THBPEvent*)_self;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  supaIpcEventHandle_t handle;
  self->supa_event.ipc_handle(&handle);
  return PyBytes_FromStringAndSize((const char*)&handle, sizeof(handle));
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*c-arrays*, *global-variables)
static struct PyGetSetDef THBPEvent_properties[] = {
    {"device", (getter)THBPEvent_get_device, nullptr, nullptr, nullptr},
    {"supa_event", (getter)THBPEvent_get_supa_event, nullptr, nullptr, nullptr},
    {nullptr}};

// NOLINTNEXTLINE(*c-arrays*, *global-variables)
static PyMethodDef THBPEvent_methods[] = {
    {(char*)"from_ipc_handle",
     castPyCFunctionWithKeywords(THBPEvent_from_ipc_handle),
     METH_CLASS | METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {(char*)"record", THBPEvent_record, METH_O, nullptr},
    {(char*)"wait", THBPEvent_wait, METH_O, nullptr},
    {(char*)"query", THBPEvent_query, METH_NOARGS, nullptr},
    {(char*)"elapsed_time", THBPEvent_elapsed_time, METH_O, nullptr},
    {(char*)"synchronize", THBPEvent_synchronize, METH_NOARGS, nullptr},
    {(char*)"ipc_handle", THBPEvent_ipc_handle, METH_NOARGS, nullptr},
    {nullptr}};

PyTypeObject THBPEventType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch_supa._C._SUPAEventBase", /* tp_name */
    sizeof(THBPEvent), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THBPEvent_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THBPEvent_methods, /* tp_methods */
    nullptr, /* tp_members */
    THBPEvent_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THBPEvent_pynew, /* tp_new */
};

void THBPEvent_init(PyObject* module) {
#if TORCH_VER >= TORCH_2_6_0
  TORCH_CHECK(THPEventClass, "THPEvent has not been initialized yet.");
  Py_INCREF(THPEventClass);
  THBPEventType.tp_base = THPEventClass;
#endif
  THBPEventClass = (PyObject*)&THBPEventType;
  if (PyType_Ready(&THBPEventType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THBPEventType);
  if (PyModule_AddObject(module, "_SUPAEventBase", (PyObject*)&THBPEventType) < 0) {
    throw python_error();
  }
}